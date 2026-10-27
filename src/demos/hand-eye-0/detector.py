#!/usr/bin/env python3
"""AprilTag detection and PnP pose estimation layer."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation

from config import TAG_SIZES_M, DEFAULT_TAG_SIZE_M

log = logging.getLogger("hand-eye-0.detector")


@dataclass
class DetectedTag:
    tag_id: int
    corners_px: np.ndarray          # (4, 2)
    center_px: np.ndarray           # (2,)
    T_cam_tag: np.ndarray           # (4, 4) SE(3) transform from camera to tag
    reprojection_error_px: float    # RMS reprojection error
    decision_margin: float          # Tag decode confidence
    corner_jitter_px: float = 0.0   # Repeatability uncertainty
    repeat_count: int = 1           # Number of fused repeats


@dataclass
class PoseObservation:
    pose_name: str
    T_base_tcp: np.ndarray          # 4x4 matrix from base_link to tcp
    joint_angles_deg: dict[str, float]
    d415_tags: dict[int, DetectedTag]
    usb_tags: dict[int, DetectedTag]
    joint_angles_rad: list[float] = field(default_factory=list)
    d415_path: Path | None = None
    usb_path: Path | None = None


class TagDetector:
    """Wraps pupil-apriltags detector with OpenCV solvePnP.
    Uses singleton caching to prevent C library double-free SIGSEGV in pupil_apriltags __del__.
    """
    _instances: dict[tuple[str, int], TagDetector] = {}

    def __new__(cls, family: str = "tag36h11", nthreads: int = 4):
        key = (family, nthreads)
        if key not in cls._instances:
            inst = super().__new__(cls)
            inst.detector = Detector(families=family, nthreads=nthreads, quad_decimate=1.0)
            cls._instances[key] = inst
        return cls._instances[key]

    def __init__(self, family: str = "tag36h11", nthreads: int = 4):
        pass

    def detect(self, image: np.ndarray, K: np.ndarray, D: np.ndarray, refine_subpix: bool = True) -> dict[int, DetectedTag]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h_img, w_img = gray.shape[:2]
        raw_dets = self.detector.detect(gray)
        results = {}

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.005)

        for d in raw_dets:
            tid = int(d.tag_id)
            nominal_size = TAG_SIZES_M.get(tid, DEFAULT_TAG_SIZE_M)
            h = nominal_size / 2.0

            c_px = np.array(d.corners, dtype=np.float32)

            # 1. Boundary rejection: reject tags truncated or distorted near image edges
            margin = 5.0
            if (c_px[:, 0] < margin).any() or (c_px[:, 0] > w_img - margin).any() or \
               (c_px[:, 1] < margin).any() or (c_px[:, 1] > h_img - margin).any():
                continue

            # 2. Minimum area rejection: ignore tags below 80 px^2 (excessive noise)
            area = float(cv2.contourArea(c_px))
            if area < 80.0:
                continue

            # 3. Subpixel corner refinement on grayscale image
            if refine_subpix:
                try:
                    c_ref = cv2.cornerSubPix(
                        gray, c_px.copy(), winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria
                    )
                    # Safety check: if corner moved > 4 px, keep original
                    if np.max(np.linalg.norm(c_ref - c_px, axis=1)) < 4.0:
                        c_px = c_ref
                except Exception:
                    pass

            obj_pts = np.array([
                [-h,  h, 0.0],
                [ h,  h, 0.0],
                [ h, -h, 0.0],
                [-h, -h, 0.0]
            ], dtype=np.float64)

            img_pts = np.array(c_px, dtype=np.float64)

            success, rvecs, tvecs, repr_errs = cv2.solvePnPGeneric(
                obj_pts,
                img_pts,
                K,
                D,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not success or len(rvecs) == 0:
                continue

            # 4. Resolve IPPE planar pose ambiguity:
            # Check positive depth and ensure tag surface normal points toward camera
            best_idx = 0
            best_score = 1e9
            for i in range(len(rvecs)):
                tz = float(tvecs[i][2, 0])
                if tz <= 0.05:
                    continue
                R_cand, _ = cv2.Rodrigues(rvecs[i])
                t_cand = tvecs[i].ravel()
                # Surface normal in camera frame (Z axis of tag)
                n_cam = R_cand[:, 2]
                v_cam = t_cand / (np.linalg.norm(t_cand) + 1e-9)
                cos_view = -float(np.dot(n_cam, v_cam))
                if cos_view < 0.05:  # Facing away from camera
                    continue

                err = float(repr_errs[i, 0]) if repr_errs is not None and len(repr_errs) > i else 0.0
                score = err / max(cos_view, 0.1)
                if score < best_score:
                    best_score = score
                    best_idx = i

            R, _ = cv2.Rodrigues(rvecs[best_idx])
            t = tvecs[best_idx].ravel()

            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = t

            results[tid] = DetectedTag(
                tag_id=tid,
                corners_px=img_pts,
                center_px=np.mean(img_pts, axis=0),
                T_cam_tag=T,
                reprojection_error_px=float(repr_errs[best_idx, 0]) if repr_errs is not None and len(repr_errs) > best_idx else 0.0,
                decision_margin=float(getattr(d, "decision_margin", 0.0)),
            )

        return results


def load_all_poses(
    dataset_dir: Path | str,
    K_d415: np.ndarray,
    D_d415: np.ndarray,
    K_usb: np.ndarray,
    D_usb: np.ndarray,
) -> list[PoseObservation]:
    """Loads all poses, joint angles, and detects tags in both d415 and usb feeds."""
    root = Path(dataset_dir)
    detector = TagDetector()
    observations: list[PoseObservation] = []

    pose_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    log.info(f"Scanning {len(pose_dirs)} pose directories in: {root}")

    for pdir in pose_dirs:
        ja_file = pdir / "joint_angles.yaml"
        if not ja_file.is_file():
            continue

        with open(ja_file, "r", encoding="utf-8") as f:
            ja_data = yaml.safe_load(f)

        eep = ja_data["end_effector_pose"]
        t_tcp = np.array([eep["x"], eep["y"], eep["z"]], dtype=np.float64)
        q_tcp = [eep["qx"], eep["qy"], eep["qz"], eep["qw"]]
        R_tcp = Rotation.from_quat(q_tcp).as_matrix()

        T_base_tcp = np.eye(4, dtype=np.float64)
        T_base_tcp[:3, :3] = R_tcp
        T_base_tcp[:3, 3] = t_tcp

        joint_angles_deg = ja_data.get("joint_angles_deg", {})
        joint_angles_rad = ja_data.get("positions_rad", [])
        if not joint_angles_rad and joint_angles_deg:
            joint_angles_rad = [
                np.radians(joint_angles_deg.get(f"joint{i}", 0.0)) for i in range(1, 7)
            ]

        # Multi-frame and single-frame image discovery
        d415_candidates = sorted([p for p in pdir.glob("front_*.png")])
        if not d415_candidates:
            d415_candidates = sorted([p for p in pdir.glob("d415_*.png") if "depth" not in p.name])
        if not d415_candidates:
            for fname in ["front.png", "d415.png"]:
                p = pdir / fname
                if p.is_file():
                    d415_candidates.append(p)
                    break

        usb_candidates = sorted([p for p in pdir.glob("see3_*.png")])
        if not usb_candidates:
            usb_candidates = sorted([p for p in pdir.glob("usb_*.png")])
        if not usb_candidates:
            usb_candidates = sorted([p for p in pdir.glob("down_*.png")])
        if not usb_candidates:
            for fname in ["see3.png", "usb.png", "down.png"]:
                p = pdir / fname
                if p.is_file():
                    usb_candidates.append(p)
                    break

        num_shots = max(len(d415_candidates), len(usb_candidates), 1)
        for shot_idx in range(num_shots):
            d415_path = d415_candidates[shot_idx] if shot_idx < len(d415_candidates) else (d415_candidates[0] if d415_candidates else None)
            usb_path = usb_candidates[shot_idx] if shot_idx < len(usb_candidates) else (usb_candidates[0] if usb_candidates else None)

            d415_tags = {}
            if d415_path is not None and d415_path.is_file():
                img_d415 = cv2.imread(str(d415_path))
                if img_d415 is not None:
                    d415_tags = detector.detect(img_d415, K_d415, D_d415)

            usb_tags = {}
            if usb_path is not None and usb_path.is_file():
                img_usb = cv2.imread(str(usb_path))
                if img_usb is not None:
                    usb_tags = detector.detect(img_usb, K_usb, D_usb)

            pose_label = f"{pdir.name}_shot{shot_idx+1:02d}" if num_shots > 1 else pdir.name
            obs = PoseObservation(
                pose_name=pose_label,
                T_base_tcp=T_base_tcp,
                joint_angles_deg=joint_angles_deg,
                d415_tags=d415_tags,
                usb_tags=usb_tags,
                joint_angles_rad=joint_angles_rad,
                d415_path=d415_path,
                usb_path=usb_path,
            )
            observations.append(obs)

    log.info(f"Loaded and processed {len(observations)} pose observations.")
    return observations


def cluster_and_fuse_repeated_poses(
    observations: list[PoseObservation],
    K_d: np.ndarray | None = None,
    D_d: np.ndarray | None = None,
    K_u: np.ndarray | None = None,
    D_u: np.ndarray | None = None,
    pos_tol_m: float = 0.003,      # 3 mm translation tolerance
    rot_tol_deg: float = 0.5,      # 0.5 deg rotation tolerance
) -> tuple[list[PoseObservation], dict[str, Any]]:
    """Groups repeated robot poses, rejects transient corner outliers, and fuses multi-observation
    subpixel corners to suppress image sensor noise by 1 / sqrt(N).
    
    Returns:
        fused_observations: List of observations with repeated captures merged into high-precision points.
        stats: Diagnostic telemetry on repeats detected, jitter suppressed, and outlier rejections.
    """
    if not observations:
        return [], {}

    clusters: list[list[PoseObservation]] = []
    for obs in observations:
        matched = False
        t_obs = obs.T_base_tcp[:3, 3]
        R_obs = obs.T_base_tcp[:3, :3]
        for cluster in clusters:
            rep = cluster[0]
            t_rep = rep.T_base_tcp[:3, 3]
            R_rep = rep.T_base_tcp[:3, :3]
            dist_m = float(np.linalg.norm(t_obs - t_rep))
            angle_deg = float(np.linalg.norm(Rotation.from_matrix(R_obs.T @ R_rep).as_rotvec()) * 180.0 / np.pi)
            if dist_m <= pos_tol_m and angle_deg <= rot_tol_deg:
                cluster.append(obs)
                matched = True
                break
        if not matched:
            clusters.append([obs])

    fused_observations: list[PoseObservation] = []
    total_repeats = 0
    jitter_estimates = []

    for cluster in clusters:
        if len(cluster) == 1:
            fused_observations.append(cluster[0])
            continue

        total_repeats += len(cluster)
        N = len(cluster)

        # Average TCP pose
        mean_t = np.mean([o.T_base_tcp[:3, 3] for o in cluster], axis=0)
        R_sum = np.sum([o.T_base_tcp[:3, :3] for o in cluster], axis=0)
        u, _, vt = np.linalg.svd(R_sum)
        mean_R = u @ vt
        if np.linalg.det(mean_R) < 0:
            u[:, -1] *= -1.0
            mean_R = u @ vt
        T_fused_tcp = np.eye(4)
        T_fused_tcp[:3, :3] = mean_R
        T_fused_tcp[:3, 3] = mean_t

        mean_deg = {}
        for k in cluster[0].joint_angles_deg.keys():
            mean_deg[k] = float(np.mean([o.joint_angles_deg.get(k, 0.0) for o in cluster]))
        mean_rad = [
            float(np.mean([o.joint_angles_rad[i] for o in cluster if i < len(o.joint_angles_rad)]))
            for i in range(len(cluster[0].joint_angles_rad))
        ]

        def fuse_camera_tags(cam_attr: str, K: np.ndarray | None, D: np.ndarray | None) -> dict[int, DetectedTag]:
            fused_tags = {}
            all_tids = sorted(list({tid for o in cluster for tid in getattr(o, cam_attr).keys()}))
            for tid in all_tids:
                cand_dets = [getattr(o, cam_attr)[tid] for o in cluster if tid in getattr(o, cam_attr)]
                cand_corners = [d.corners_px for d in cand_dets]
                if not cand_corners:
                    continue
                cand_arr = np.array(cand_corners)  # (M, 4, 2)
                M = len(cand_corners)
                if M >= 3:
                    med_corners = np.median(cand_arr, axis=0)
                    dists = np.max(np.linalg.norm(cand_arr - med_corners, axis=2), axis=1)
                    inliers = dists <= 2.0  # within 2 px of median
                    if np.sum(inliers) >= 1:
                        cand_arr = cand_arr[inliers]

                fused_c = np.mean(cand_arr, axis=0)
                jitter_px = float(np.mean(np.std(cand_arr, axis=0))) if len(cand_arr) > 1 else 0.0
                jitter_estimates.append(jitter_px)
                center_c = np.mean(fused_c, axis=0)

                tag_size = TAG_SIZES_M.get(tid, DEFAULT_TAG_SIZE_M)
                T_c_t = cand_dets[0].T_cam_tag
                reproj_err = cand_dets[0].reprojection_error_px
                if K is not None and D is not None:
                    half = tag_size / 2.0
                    obj_pts = np.array([
                        [-half,  half, 0.0],
                        [ half,  half, 0.0],
                        [ half, -half, 0.0],
                        [-half, -half, 0.0],
                    ], dtype=np.float64)
                    ok, rvec, tvec = cv2.solvePnP(obj_pts, fused_c, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if ok:
                        R_c_t, _ = cv2.Rodrigues(rvec)
                        T_c_t = np.eye(4)
                        T_c_t[:3, :3] = R_c_t
                        T_c_t[:3, 3] = tvec.ravel()
                        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
                        reproj_err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - fused_c, axis=1)))

                fused_tags[tid] = DetectedTag(
                    tag_id=tid,
                    corners_px=fused_c,
                    center_px=center_c,
                    T_cam_tag=T_c_t,
                    reprojection_error_px=reproj_err,
                    decision_margin=float(np.mean([d.decision_margin for d in cand_dets])),
                    corner_jitter_px=jitter_px,
                    repeat_count=M,
                )
            return fused_tags

        fused_d415 = fuse_camera_tags("d415_tags", K_d, D_d)
        fused_usb = fuse_camera_tags("usb_tags", K_u, D_u)

        fused_obs = PoseObservation(
            pose_name=f"{cluster[0].pose_name}_fused_{N}x",
            T_base_tcp=T_fused_tcp,
            joint_angles_deg=mean_deg,
            d415_tags=fused_d415,
            usb_tags=fused_usb,
            joint_angles_rad=mean_rad,
            d415_path=cluster[0].d415_path,
            usb_path=cluster[0].usb_path,
        )
        fused_observations.append(fused_obs)

    avg_jitter = float(np.mean(jitter_estimates)) if jitter_estimates else 0.0
    rep_clusters = sum(1 for c in clusters if len(c) > 1)
    stats = {
        "raw_poses": len(observations),
        "fused_poses": len(fused_observations),
        "repeated_clusters": rep_clusters,
        "total_repeated_frames": total_repeats,
        "avg_corner_jitter_px": round(avg_jitter, 3),
        "theoretical_noise_reduction_pct": round((1.0 - 1.0 / np.sqrt(max(2, total_repeats / max(1, rep_clusters)))) * 100.0, 1) if total_repeats > 0 else 0.0,
    }
    if rep_clusters > 0:
        log.info(f"Pose Repeat Fusion: {len(observations)} raw poses -> {len(fused_observations)} fused poses ({rep_clusters} clusters, corner jitter: {avg_jitter:.3f} px).")
    return fused_observations, stats
