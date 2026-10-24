#!/usr/bin/env python3
"""Mathematical solver for Hand-Eye Calibration, static tag mapping, and multi-camera registration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from detector import PoseObservation
from config import BASE_LINKED_TAGS, OBJECT_LINKED_TAGS, STATIC_TAGS, TOOL_TCP_TAGS, TAG_SIZES_M

log = logging.getLogger("hand-eye-0.solver")


def chordal_mean_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    """Computes the chordal L2 mean of a set of 3x3 rotation matrices via SVD."""
    if not rotations:
        return np.eye(3)
    if len(rotations) == 1:
        return rotations[0].copy()

    M = np.sum(rotations, axis=0)
    u, _, vt = np.linalg.svd(M)
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1.0
        R = u @ vt
    return R


@dataclass
class HandEyeCalibrationResult:
    T_base_d415: np.ndarray                 # 4x4 transform from base_link to d415
    T_base_usb: np.ndarray | None           # 4x4 transform from base_link to usb
    tool_mounts: dict[int, np.ndarray]      # tag_id -> 4x4 transform from tcp to tag
    static_tag_poses: dict[int, np.ndarray] # tag_id -> 4x4 transform from base_link to tag
    box_plane_normal: np.ndarray | None     # Normal vector of workpiece plane in base_link
    box_plane_center: np.ndarray | None     # Center point of workpiece plane in base_link
    box_coplanarity_rmse_mm: float = 0.0
    pos_rms_mm: float = 0.0
    pos_max_mm: float = 0.0
    rot_rms_deg: float = 0.0
    rot_max_deg: float = 0.0
    num_tool_sightings: int = 0
    per_pose_errors: list[dict] = field(default_factory=list)


def solve_closed_form_rotation(samples: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Closed-form rotation solve from pairs of relative motions: A * R_X = R_X * B."""
    # Build linear system on quaternion q_X: (L(q_A) - R(q_B)) q_X = 0
    eqs = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            T_b_tcp_1, T_c_t_1 = samples[i]
            T_b_tcp_2, T_c_t_2 = samples[j]

            # Relative motion in base frame: A = T_b_tcp_2 * T_b_tcp_1^-1
            # Relative motion in cam frame:  B = T_c_t_2 * T_c_t_1^-1
            R_A = T_b_tcp_2[:3, :3] @ T_b_tcp_1[:3, :3].T
            R_B = T_c_t_2[:3, :3] @ T_c_t_1[:3, :3].T

            angle_A = np.linalg.norm(Rotation.from_matrix(R_A).as_rotvec())
            angle_B = np.linalg.norm(Rotation.from_matrix(R_B).as_rotvec())
            if angle_A < np.radians(3.0) or angle_B < np.radians(3.0):
                continue

            qA = Rotation.from_matrix(R_A).as_quat()  # [x, y, z, w]
            qB = Rotation.from_matrix(R_B).as_quat()

            # Convert to [w, x, y, z] for quaternion algebra
            wA, xA, yA, zA = qA[3], qA[0], qA[1], qA[2]
            wB, xB, yB, zB = qB[3], qB[0], qB[1], qB[2]

            # L(qA) - R(qB)
            row1 = [ wA - wB, -xA + xB, -yA + yB, -zA + zB ]
            row2 = [ xA - xB,  wA - wB,  zA + zB, -yA - yB ]
            row3 = [ yA - yB, -zA - zB,  wA - wB,  xA + xB ]
            row4 = [ zA - zB,  yA + yB, -xA - xB,  wA - wB ]
            eqs.extend([row1, row2, row3, row4])

    if len(eqs) < 4:
        return np.eye(3)

    A_mat = np.array(eqs, dtype=np.float64)
    _, _, vt = np.linalg.svd(A_mat)
    q_sol = vt[-1]  # [w, x, y, z]
    q_sol = q_sol / np.linalg.norm(q_sol)
    return Rotation.from_quat([q_sol[1], q_sol[2], q_sol[3], q_sol[0]]).as_matrix()


def solve_handeye_pipeline(observations: list[PoseObservation]) -> HandEyeCalibrationResult:
    """Full optimization pipeline for Hand-Eye calibration."""
    # 1. Collect tool tag observations across all poses
    tool_samples: list[dict] = []
    tags_seen: set[int] = set()

    for obs in observations:
        for tid in TOOL_TCP_TAGS:
            if tid in obs.d415_tags:
                d = obs.d415_tags[tid]
                tool_samples.append({
                    "pose": obs.pose_name,
                    "tid": tid,
                    "T_b_tcp": obs.T_base_tcp,
                    "T_c_t": d.T_cam_tag,
                })
                tags_seen.add(tid)

    log.info(f"Collected {len(tool_samples)} tool tag sightings across {len(observations)} poses. Tags: {sorted(tags_seen)}")

    # 2. Closed-form initial solve
    # Separate by tag ID for rotation solve
    rot_pairs = []
    for tid in tags_seen:
        samples_tid = [(s["T_b_tcp"], s["T_c_t"]) for s in tool_samples if s["tid"] == tid]
        rot_pairs.extend(samples_tid)

    R_init = solve_closed_form_rotation(rot_pairs)

    # Initial tool tag rotations: R_Y = R_L^-1 * R_X * R_C
    init_mounts: dict[int, np.ndarray] = {}
    for tid in tags_seen:
        mount_rot_cands = []
        for s in tool_samples:
            if s["tid"] == tid:
                R_tcp = s["T_b_tcp"][:3, :3]
                R_tag = s["T_c_t"][:3, :3]
                R_Y = R_tcp.T @ R_init @ R_tag
                mount_rot_cands.append(R_Y)
        R_mount_mean = chordal_mean_rotation(mount_rot_cands)
        init_mounts[tid] = np.eye(4)
        init_mounts[tid][:3, :3] = R_mount_mean

    # Initial linear least squares for translation:
    # T_b_tcp * Y = X * T_c_t  =>  R_b_tcp * t_Y + t_b_tcp = R_X * t_C + t_X
    # t_X - R_b_tcp * t_Y = t_b_tcp - R_X * t_C
    A_rows = []
    b_rows = []
    tag_list = sorted(tags_seen)
    tag_to_idx = {tid: i for i, tid in enumerate(tag_list)}

    for s in tool_samples:
        tid = s["tid"]
        tid_idx = tag_to_idx[tid]
        R_b_tcp = s["T_b_tcp"][:3, :3]
        t_b_tcp = s["T_b_tcp"][:3, 3]
        t_c_tag = s["T_c_t"][:3, 3]

        row = np.zeros((3, 3 + 3 * len(tag_list)))
        row[:, 0:3] = np.eye(3)
        row[:, 3 + 3 * tid_idx : 3 + 3 * (tid_idx + 1)] = -R_b_tcp

        b_vec = t_b_tcp - R_init @ t_c_tag
        A_rows.append(row)
        b_rows.append(b_vec)

    A_mat = np.vstack(A_rows)
    b_mat = np.concatenate(b_rows)
    t_sol, _, _, _ = np.linalg.lstsq(A_mat, b_mat, rcond=None)

    t_cam_init = t_sol[0:3]
    for tid in tag_list:
        idx = tag_to_idx[tid]
        init_mounts[tid][:3, 3] = t_sol[3 + 3 * idx : 3 + 3 * (idx + 1)]

    # 3. Non-linear Joint Optimization (Levenberg-Marquardt with Cauchy Loss)
    # Parameter vector:
    # rvec_cam (3), tvec_cam (3), followed by (rvec_mount_i (3), tvec_mount_i (3)) for each tool tag
    r_cam0 = Rotation.from_matrix(R_init).as_rotvec()
    params_init = [r_cam0, t_cam_init]
    for tid in tag_list:
        r_m = Rotation.from_matrix(init_mounts[tid][:3, :3]).as_rotvec()
        t_m = init_mounts[tid][:3, 3]
        params_init.extend([r_m, t_m])

    x0 = np.concatenate(params_init)

    def residuals_fn(p):
        R_c = Rotation.from_rotvec(p[0:3]).as_matrix()
        t_c = p[3:6]
        T_c = np.eye(4); T_c[:3, :3] = R_c; T_c[:3, 3] = t_c

        cur_mounts = {}
        for idx, tid in enumerate(tag_list):
            base_p = 6 + 6 * idx
            R_m = Rotation.from_rotvec(p[base_p : base_p + 3]).as_matrix()
            t_m = p[base_p + 3 : base_p + 6]
            T_m = np.eye(4); T_m[:3, :3] = R_m; T_m[:3, 3] = t_m
            cur_mounts[tid] = T_m

        res = []
        for s in tool_samples:
            tid = s["tid"]
            T_b_tcp = s["T_b_tcp"]
            T_c_tag = s["T_c_t"]
            T_pred = T_b_tcp @ cur_mounts[tid]
            T_meas = T_c @ T_c_tag

            # Translation residual in mm
            pos_err = (T_pred[:3, 3] - T_meas[:3, 3]) * 1000.0
            # Rotation residual in degrees (scaled to balance position)
            R_diff = T_pred[:3, :3].T @ T_meas[:3, :3]
            angle_deg = np.linalg.norm(Rotation.from_matrix(R_diff).as_rotvec()) * 180.0 / np.pi
            res.extend(pos_err.tolist())
            res.append(angle_deg * 2.0)
        return np.array(res)

    opt = least_squares(residuals_fn, x0, loss="cauchy", f_scale=2.5)

    R_cam_opt = Rotation.from_rotvec(opt.x[0:3]).as_matrix()
    t_cam_opt = opt.x[3:6]
    T_base_d415 = np.eye(4)
    T_base_d415[:3, :3] = R_cam_opt
    T_base_d415[:3, 3] = t_cam_opt

    final_mounts = {}
    for idx, tid in enumerate(tag_list):
        base_p = 6 + 6 * idx
        R_m = Rotation.from_rotvec(opt.x[base_p : base_p + 3]).as_matrix()
        t_m = opt.x[base_p + 3 : base_p + 6]
        T_m = np.eye(4); T_m[:3, :3] = R_m; T_m[:3, 3] = t_m
        final_mounts[tid] = T_m

    # 4. Error Statistics
    pos_errs = []
    rot_errs = []
    per_pose_errors = []

    for s in tool_samples:
        tid = s["tid"]
        T_pred = s["T_b_tcp"] @ final_mounts[tid]
        T_meas = T_base_d415 @ s["T_c_t"]
        pe = float(np.linalg.norm(T_pred[:3, 3] - T_meas[:3, 3]) * 1000.0)
        re = float(np.linalg.norm(Rotation.from_matrix(T_pred[:3, :3].T @ T_meas[:3, :3]).as_rotvec()) * 180.0 / np.pi)
        pos_errs.append(pe)
        rot_errs.append(re)
        per_pose_errors.append({
            "pose": s["pose"],
            "tag_id": tid,
            "pos_err_mm": pe,
            "rot_err_deg": re,
            "predicted_pos_mm": (T_pred[:3, 3] * 1000.0).tolist(),
            "measured_pos_mm": (T_meas[:3, 3] * 1000.0).tolist(),
        })

    pos_rms = float(np.sqrt(np.mean(np.array(pos_errs) ** 2)))
    pos_max = float(np.max(pos_errs))
    rot_rms = float(np.sqrt(np.mean(np.array(rot_errs) ** 2)))
    rot_max = float(np.max(rot_errs))

    # 5. Static Tag Poses in base_link (from D415 observations across all poses)
    static_tag_poses: dict[int, np.ndarray] = {}
    all_static_ids = BASE_LINKED_TAGS | OBJECT_LINKED_TAGS

    for tid in all_static_ids:
        cands_cam = []
        for obs in observations:
            if tid in obs.d415_tags:
                cands_cam.append(obs.d415_tags[tid].T_cam_tag)
        if cands_cam:
            ts = np.array([T[:3, 3] for T in cands_cam])
            t_mean = np.mean(ts, axis=0)
            R_mean = chordal_mean_rotation([T[:3, :3] for T in cands_cam])
            T_c_tag = np.eye(4); T_c_tag[:3, :3] = R_mean; T_c_tag[:3, 3] = t_mean
            T_base_tag = T_base_d415 @ T_c_tag
            static_tag_poses[tid] = T_base_tag

    # 6. Workpiece / Box Plane Analysis (Tags 42, 43, 44, 45)
    box_pts = []
    box_normal = None
    box_center = None
    box_rmse = 0.0

    for tid in (42, 43, 44, 45):
        if tid in static_tag_poses:
            box_pts.append(static_tag_poses[tid][:3, 3])

    if len(box_pts) >= 3:
        pts_arr = np.array(box_pts, dtype=np.float64)
        box_center = np.mean(pts_arr, axis=0)
        _, _, vh = np.linalg.svd(pts_arr - box_center)
        normal = vh[2, :]
        normal = normal / np.linalg.norm(normal)
        if normal[2] < 0:
            normal = -normal
        box_normal = normal
        d_val = -float(np.dot(normal, box_center))
        resids = np.abs(pts_arr @ normal + d_val)
        box_rmse = float(np.sqrt(np.mean(resids ** 2)) * 1000.0)

    # 7. Solve Second Camera Pose: T_base_usb
    # Use static tags visible in usb feed registered against their solved 3D positions in base_link
    T_base_usb = None
    usb_obj_pts = []
    usb_img_pts = []

    for obs in observations:
        for tid, d in obs.usb_tags.items():
            if tid in static_tag_poses:
                T_b_tag = static_tag_poses[tid]
                half = TAG_SIZES_M.get(tid, 0.066) / 2.0
                local_corners = np.array([
                    [-half,  half, 0.0],
                    [ half,  half, 0.0],
                    [ half, -half, 0.0],
                    [-half, -half, 0.0],
                ], dtype=np.float64)
                corners_base = (T_b_tag[:3, :3] @ local_corners.T).T + T_b_tag[:3, 3]
                for p_b, p_px in zip(corners_base, d.corners_px):
                    usb_obj_pts.append(p_b)
                    usb_img_pts.append(p_px)

    if len(usb_obj_pts) >= 6:
        # PnP to find USB camera pose in base_link
        from config import DEFAULT_USB_CALIB, load_camera_intrinsics
        K_u, D_u, _ = load_camera_intrinsics(DEFAULT_USB_CALIB)
        ok, rvec_u, tvec_u = cv2.solvePnP(
            np.array(usb_obj_pts, dtype=np.float64),
            np.array(usb_img_pts, dtype=np.float64),
            K_u, D_u, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if ok:
            R_usb_base, _ = cv2.Rodrigues(rvec_u)
            t_usb_base = tvec_u.ravel()
            # T_usb_base -> invert to get T_base_usb
            T_u_b = np.eye(4); T_u_b[:3, :3] = R_usb_base; T_u_b[:3, 3] = t_usb_base
            T_base_usb = np.linalg.inv(T_u_b)
            log.info(f"Solved USB camera pose in base_link: t={T_base_usb[:3, 3].round(3).tolist()}")

    return HandEyeCalibrationResult(
        T_base_d415=T_base_d415,
        T_base_usb=T_base_usb,
        tool_mounts=final_mounts,
        static_tag_poses=static_tag_poses,
        box_plane_normal=box_normal,
        box_plane_center=box_center,
        box_coplanarity_rmse_mm=box_rmse,
        pos_rms_mm=pos_rms,
        pos_max_mm=pos_max,
        rot_rms_deg=rot_rms,
        rot_max_deg=rot_max,
        num_tool_sightings=len(tool_samples),
        per_pose_errors=per_pose_errors,
    )
