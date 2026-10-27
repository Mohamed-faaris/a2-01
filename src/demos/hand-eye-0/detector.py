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

    def detect(self, image: np.ndarray, K: np.ndarray, D: np.ndarray) -> dict[int, DetectedTag]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        raw_dets = self.detector.detect(gray)
        results = {}

        for d in raw_dets:
            tid = int(d.tag_id)
            nominal_size = TAG_SIZES_M.get(tid, DEFAULT_TAG_SIZE_M)
            h = nominal_size / 2.0

            obj_pts = np.array([
                [-h,  h, 0.0],
                [ h,  h, 0.0],
                [ h, -h, 0.0],
                [-h, -h, 0.0]
            ], dtype=np.float64)

            img_pts = np.array(d.corners, dtype=np.float64)

            success, rvecs, tvecs, repr_errs = cv2.solvePnPGeneric(
                obj_pts,
                img_pts,
                K,
                D,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not success or len(rvecs) == 0:
                continue

            # Pick the solution with positive depth and lowest reprojection error
            best_idx = 0
            best_err = 1e9
            for i in range(len(rvecs)):
                tz = float(tvecs[i][2, 0])
                err = float(repr_errs[i, 0]) if repr_errs is not None and len(repr_errs) > i else 0.0
                if tz > 0.01 and err < best_err:
                    best_err = err
                    best_idx = i

            R, _ = cv2.Rodrigues(rvecs[best_idx])
            t = tvecs[best_idx].ravel()

            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = t

            results[tid] = DetectedTag(
                tag_id=tid,
                corners_px=img_pts,
                center_px=np.array(d.center, dtype=np.float64),
                T_cam_tag=T,
                reprojection_error_px=best_err,
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

        # D415 Image (accepts d415.png or front.png)
        d415_img_path = pdir / "d415.png"
        if not d415_img_path.is_file():
            d415_img_path = pdir / "front.png"
        d415_tags = {}
        if d415_img_path.is_file():
            img_d415 = cv2.imread(str(d415_img_path))
            if img_d415 is not None:
                d415_tags = detector.detect(img_d415, K_d415, D_d415)

        # USB / Secondary Image (accepts usb.png, see3.png, or down.png)
        usb_img_path = pdir / "usb.png"
        if not usb_img_path.is_file():
            usb_img_path = pdir / "see3.png"
        if not usb_img_path.is_file():
            usb_img_path = pdir / "down.png"
        usb_tags = {}
        if usb_img_path.is_file():
            img_usb = cv2.imread(str(usb_img_path))
            if img_usb is not None:
                usb_tags = detector.detect(img_usb, K_usb, D_usb)

        obs = PoseObservation(
            pose_name=pdir.name,
            T_base_tcp=T_base_tcp,
            joint_angles_deg=joint_angles_deg,
            d415_tags=d415_tags,
            usb_tags=usb_tags,
            joint_angles_rad=joint_angles_rad,
            d415_path=d415_img_path if d415_img_path.is_file() else None,
            usb_path=usb_img_path if usb_img_path.is_file() else None,
        )
        observations.append(obs)

    log.info(f"Loaded and processed {len(observations)} pose observations.")
    return observations
