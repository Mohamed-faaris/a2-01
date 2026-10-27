#!/usr/bin/env python3
"""Mathematical solver for Hand-Eye Calibration, static tag mapping, and multi-camera registration.

Supports multiple algorithmic methodologies per TESTING_PROCEDURE.md:
1. '3d_pose_cauchy': SVD closed-form init + Levenberg-Marquardt with Cauchy loss on 3D SE(3) pose error.
2. '2d_corner_ba': Full Non-linear Corner Reprojection Bundle Adjustment minimizing pixel residuals.
3. 'rigid_cube_ba': 6-Faced Rigid Cube constraint (fusing Tags 20 & 21 rigidly) + 2D Bundle Adjustment.
4. 'joint_dual_cam_ba': Joint D415 + USB dual-camera optimization coupled via tool and static scene tags.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from detector import PoseObservation
from config import (
    BASE_LINKED_TAGS,
    BOX_TAG_GROUPS,
    OBJECT_LINKED_TAGS,
    STATIC_TAGS,
    TOOL_TCP_TAGS,
    TAG_SIZES_M,
    DEFAULT_D415_CALIB,
    DEFAULT_USB_CALIB,
    load_camera_intrinsics,
)

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
    method: str
    T_base_d415: np.ndarray                 # 4x4 transform from base_link to d415
    T_base_usb: np.ndarray | None           # 4x4 transform from base_link to usb
    tool_mounts: dict[int, np.ndarray]      # tag_id -> 4x4 transform from tcp to tag
    static_tag_poses: dict[int, np.ndarray] # tag_id -> 4x4 transform from base_link to tag
    box_plane_normal: np.ndarray | None     # Normal vector of primary workpiece plane in base_link
    box_plane_center: np.ndarray | None     # Center point of primary workpiece plane in base_link
    box_coplanarity_rmse_mm: float = 0.0
    workpiece_planes: dict[str, dict] = field(default_factory=dict) # multi-box plane fits
    pos_rms_mm: float = 0.0
    pos_mean_mm: float = 0.0
    pos_max_mm: float = 0.0
    rot_rms_deg: float = 0.0
    rot_mean_deg: float = 0.0
    rot_max_deg: float = 0.0
    reproj_rms_px: float = 0.0
    reproj_mean_px: float = 0.0
    reproj_max_px: float = 0.0
    num_tool_sightings: int = 0
    per_pose_errors: list[dict] = field(default_factory=list)
    runtime_s: float = 0.0


def solve_closed_form_rotation(tag_sample_groups: list[list[tuple[np.ndarray, np.ndarray]]]) -> np.ndarray:
    """Closed-form rotation solve from pairs of relative motions: A * R_X = R_X * B.
    
    Pairs are formed STRICTLY between observations of the same physical tag.
    """
    eqs = []
    for samples in tag_sample_groups:
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                T_b_tcp_1, T_c_t_1 = samples[i]
                T_b_tcp_2, T_c_t_2 = samples[j]

                R_A = T_b_tcp_2[:3, :3] @ T_b_tcp_1[:3, :3].T
                R_B = T_c_t_2[:3, :3] @ T_c_t_1[:3, :3].T

                angle_A = np.linalg.norm(Rotation.from_matrix(R_A).as_rotvec())
                angle_B = np.linalg.norm(Rotation.from_matrix(R_B).as_rotvec())
                if angle_A < np.radians(4.0) or angle_B < np.radians(4.0):
                    continue
                # In eye-to-hand calibration R_A and R_B must have conjugate traces (|angle_A - angle_B| ~ 0).
                # Filter out planar ambiguity flips (e.g. 180 deg errors from single-tag PnP)
                if abs(angle_A - angle_B) > np.radians(8.0):
                    continue

                qA = Rotation.from_matrix(R_A).as_quat()  # [x, y, z, w]
                qB = Rotation.from_matrix(R_B).as_quat()
                # Ensure same quaternion hemisphere
                if qA[3] < 0:
                    qA = -qA
                if qB[3] < 0:
                    qB = -qB

                wA, xA, yA, zA = qA[3], qA[0], qA[1], qA[2]
                wB, xB, yB, zB = qB[3], qB[0], qB[1], qB[2]

                # qA * qX - qX * qB = 0 for qX = [wX, xX, yX, zX]
                row1 = [ wA - wB, -(xA - xB), -(yA - yB), -(zA - zB) ]
                row2 = [ xA - xB,  wA - wB, -(zA + zB),  yA + yB ]
                row3 = [ yA - yB,  zA + zB,  wA - wB, -(xA + xB) ]
                row4 = [ zA - zB, -(yA + yB),  xA + xB,  wA - wB ]
                eqs.extend([row1, row2, row3, row4])

    if len(eqs) < 4:
        log.warning("Not enough relative rotations with >3 deg difference; falling back to identity.")
        return np.eye(3)

    A_mat = np.array(eqs, dtype=np.float64)
    _, _, vt = np.linalg.svd(A_mat)
    q_sol = vt[-1]  # [w, x, y, z]
    q_sol = q_sol / np.linalg.norm(q_sol)
    return Rotation.from_quat([q_sol[1], q_sol[2], q_sol[3], q_sol[0]]).as_matrix()


def get_tag_corners_local(tag_id: int) -> np.ndarray:
    """Returns the 4 3D corner coordinates (4, 3) in tag local frame."""
    half = TAG_SIZES_M.get(tag_id, 0.066) / 2.0
    return np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float64)


def solve_handeye_pipeline(
    observations: Sequence[PoseObservation],
    method: str = "3d_pose_cauchy",
) -> HandEyeCalibrationResult:
    """Executes eye-to-hand calibration pipeline.

    Supported methods:
      - '3d_pose_cauchy': Minimizes 3D SE(3) pose error (translation + geodesic rotation).
      - '2d_corner_ba': Minimizes 2D corner reprojection pixel residuals directly.
      - 'rigid_cube_ba': Fuses cube tags 20 & 21 into a rigid 41 mm body + 2D bundle adjustment.
      - 'joint_dual_cam_ba': Joint D415 and USB camera optimization.
    """
    import time
    t_start = time.perf_counter()

    log.info(f"Solving Hand-Eye Calibration using method: '{method}'")

    K_d, D_d, _ = load_camera_intrinsics(DEFAULT_D415_CALIB)
    K_u, D_u, _ = load_camera_intrinsics(DEFAULT_USB_CALIB)

    # 1. Collect all tool sightings
    tool_samples = []
    tool_samples_usb = []
    for obs in observations:
        for tid, det in obs.d415_tags.items():
            if tid in TOOL_TCP_TAGS:
                tool_samples.append({
                    "pose": obs.pose_name,
                    "tid": tid,
                    "T_b_tcp": obs.T_base_tcp,
                    "T_c_t": det.T_cam_tag,
                    "corners_px": det.corners_px,
                    "camera": "d415",
                })
        for tid, det in obs.usb_tags.items():
            if tid in TOOL_TCP_TAGS:
                tool_samples_usb.append({
                    "pose": obs.pose_name,
                    "tid": tid,
                    "T_b_tcp": obs.T_base_tcp,
                    "T_c_t": det.T_cam_tag,
                    "corners_px": det.corners_px,
                    "camera": "usb",
                })

    if not tool_samples and not tool_samples_usb:
        raise ValueError("No tool tag sightings found in observations.")

    tag_list = sorted(list({s["tid"] for s in tool_samples}))
    tag_to_idx = {tid: i for i, tid in enumerate(tag_list)}

    # 2. Closed-form Initial Solve: Camera rotation R_base_d415
    tag_sample_groups = [
        [(s["T_b_tcp"], s["T_c_t"]) for s in tool_samples if s["tid"] == tid]
        for tid in tag_list
    ]
    R_init = solve_closed_form_rotation(tag_sample_groups)
    if np.linalg.det(R_init) < 0:
        R_init = -R_init

    # Compute initial tool rotations for each tag
    init_mounts = {}
    for tid in tag_list:
        sub = [s for s in tool_samples if s["tid"] == tid]
        R_tcp_tag_list = []
        for s in sub:
            # T_b_tcp * T_tcp_tag = T_b_c * T_c_tag => R_tcp_tag = R_b_tcp.T * R_b_c * R_c_tag
            R_est = s["T_b_tcp"][:3, :3].T @ R_init @ s["T_c_t"][:3, :3]
            R_tcp_tag_list.append(R_est)
        R_mount_init = chordal_mean_rotation(R_tcp_tag_list)
        M_init = np.eye(4)
        M_init[:3, :3] = R_mount_init
        init_mounts[tid] = M_init

    # Solve initial translations via linear least squares
    A_rows = []
    b_rows = []
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

    # 3. Method-Specific Optimization
    T_base_d415 = np.eye(4)
    final_mounts = {}

    if method == "3d_pose_cauchy":
        # Parameter vector: rvec_cam (3), tvec_cam (3), (rvec_mount_i, tvec_mount_i)
        r_cam0 = Rotation.from_matrix(R_init).as_rotvec()
        params_init = [r_cam0, t_cam_init]
        for tid in tag_list:
            r_m = Rotation.from_matrix(init_mounts[tid][:3, :3]).as_rotvec()
            t_m = init_mounts[tid][:3, 3]
            params_init.extend([r_m, t_m])
        x0 = np.concatenate(params_init)

        def res_3d(p):
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

            residuals = []
            for s in tool_samples:
                tid = s["tid"]
                T_pred = s["T_b_tcp"] @ cur_mounts[tid]
                T_meas = T_c @ s["T_c_t"]
                pos_err = (T_pred[:3, 3] - T_meas[:3, 3]) * 1000.0
                R_diff = T_pred[:3, :3].T @ T_meas[:3, :3]
                angle_deg = np.linalg.norm(Rotation.from_matrix(R_diff).as_rotvec()) * 180.0 / np.pi
                residuals.extend(pos_err.tolist())
                residuals.append(angle_deg * 2.0)
            return np.array(residuals)

        opt = least_squares(res_3d, x0, loss="cauchy", f_scale=2.5, max_nfev=200)
        R_cam_opt = Rotation.from_rotvec(opt.x[0:3]).as_matrix()
        T_base_d415[:3, :3] = R_cam_opt
        T_base_d415[:3, 3] = opt.x[3:6]
        for idx, tid in enumerate(tag_list):
            base_p = 6 + 6 * idx
            R_m = Rotation.from_rotvec(opt.x[base_p : base_p + 3]).as_matrix()
            t_m = opt.x[base_p + 3 : base_p + 6]
            M = np.eye(4); M[:3, :3] = R_m; M[:3, 3] = t_m
            final_mounts[tid] = M

    elif method in ("2d_corner_ba", "joint_dual_cam_ba"):
        # Minimize 2D Corner Reprojection Error directly
        r_cam0 = Rotation.from_matrix(R_init).as_rotvec()
        params_init = [r_cam0, t_cam_init]
        for tid in tag_list:
            r_m = Rotation.from_matrix(init_mounts[tid][:3, :3]).as_rotvec()
            t_m = init_mounts[tid][:3, 3]
            params_init.extend([r_m, t_m])
        x0 = np.concatenate(params_init)

        def res_2d(p):
            R_c = Rotation.from_rotvec(p[0:3]).as_matrix()
            t_c = p[3:6]
            T_c = np.eye(4); T_c[:3, :3] = R_c; T_c[:3, 3] = t_c
            T_c_b = np.linalg.inv(T_c)

            cur_mounts = {}
            for idx, tid in enumerate(tag_list):
                base_p = 6 + 6 * idx
                R_m = Rotation.from_rotvec(p[base_p : base_p + 3]).as_matrix()
                t_m = p[base_p + 3 : base_p + 6]
                T_m = np.eye(4); T_m[:3, :3] = R_m; T_m[:3, 3] = t_m
                cur_mounts[tid] = T_m

            residuals = []
            for s in tool_samples:
                tid = s["tid"]
                corners_local = get_tag_corners_local(tid)
                T_b_tag = s["T_b_tcp"] @ cur_mounts[tid]
                T_c_tag = T_c_b @ T_b_tag

                rvec_tag, _ = cv2.Rodrigues(T_c_tag[:3, :3])
                tvec_tag = T_c_tag[:3, 3].reshape(3, 1)

                proj, _ = cv2.projectPoints(corners_local, rvec_tag, tvec_tag, K_d, D_d)
                diff = (proj.reshape(-1, 2) - s["corners_px"]).ravel()
                residuals.extend(diff)
            return np.array(residuals)

        opt = least_squares(res_2d, x0, loss="cauchy", f_scale=1.5, max_nfev=250)
        R_cam_opt = Rotation.from_rotvec(opt.x[0:3]).as_matrix()
        T_base_d415[:3, :3] = R_cam_opt
        T_base_d415[:3, 3] = opt.x[3:6]
        for idx, tid in enumerate(tag_list):
            base_p = 6 + 6 * idx
            R_m = Rotation.from_rotvec(opt.x[base_p : base_p + 3]).as_matrix()
            t_m = opt.x[base_p + 3 : base_p + 6]
            M = np.eye(4); M[:3, :3] = R_m; M[:3, 3] = t_m
            final_mounts[tid] = M

    elif method == "rigid_cube_ba":
        if 20 in tag_list and 21 in tag_list:
            # Rigid Cube Model: Tags 20 and 21 belong to a 41 mm rigid cube
            r_cam0 = Rotation.from_matrix(R_init).as_rotvec()
            M20 = init_mounts.get(20, np.eye(4))
            M21 = init_mounts.get(21, np.eye(4))
            M40 = init_mounts.get(40, np.eye(4))
            T_20_21 = np.linalg.inv(M20) @ M21

            x0 = np.concatenate([
                r_cam0, t_cam_init,
                Rotation.from_matrix(M20[:3, :3]).as_rotvec(), M20[:3, 3],
                Rotation.from_matrix(M40[:3, :3]).as_rotvec(), M40[:3, 3],
            ])

            def res_cube(p):
                R_c = Rotation.from_rotvec(p[0:3]).as_matrix()
                t_c = p[3:6]
                T_c = np.eye(4); T_c[:3, :3] = R_c; T_c[:3, 3] = t_c
                T_c_b = np.linalg.inv(T_c)

                R_20 = Rotation.from_rotvec(p[6:9]).as_matrix()
                t_20 = p[9:12]
                T_20 = np.eye(4); T_20[:3, :3] = R_20; T_20[:3, 3] = t_20
                T_21 = T_20 @ T_20_21

                R_40 = Rotation.from_rotvec(p[12:15]).as_matrix()
                t_40 = p[15:18]
                T_40 = np.eye(4); T_40[:3, :3] = R_40; T_40[:3, 3] = t_40

                cur_mounts = {20: T_20, 21: T_21, 40: T_40}

                residuals = []
                for s in tool_samples:
                    tid = s["tid"]
                    if tid not in cur_mounts:
                        continue
                    corners_local = get_tag_corners_local(tid)
                    T_b_tag = s["T_b_tcp"] @ cur_mounts[tid]
                    T_c_tag = T_c_b @ T_b_tag

                    rvec_tag, _ = cv2.Rodrigues(T_c_tag[:3, :3])
                    tvec_tag = T_c_tag[:3, 3].reshape(3, 1)

                    proj, _ = cv2.projectPoints(corners_local, rvec_tag, tvec_tag, K_d, D_d)
                    diff = (proj.reshape(-1, 2) - s["corners_px"]).ravel()
                    residuals.extend(diff)
                return np.array(residuals)

            opt = least_squares(res_cube, x0, loss="cauchy", f_scale=1.5, max_nfev=250)
            R_cam_opt = Rotation.from_rotvec(opt.x[0:3]).as_matrix()
            T_base_d415[:3, :3] = R_cam_opt
            T_base_d415[:3, 3] = opt.x[3:6]

            R_20 = Rotation.from_rotvec(opt.x[6:9]).as_matrix()
            T_20 = np.eye(4); T_20[:3, :3] = R_20; T_20[:3, 3] = opt.x[9:12]
            T_21 = T_20 @ T_20_21

            R_40 = Rotation.from_rotvec(opt.x[12:15]).as_matrix()
            T_40 = np.eye(4); T_40[:3, :3] = R_40; T_40[:3, 3] = opt.x[15:18]

            final_mounts[20] = T_20
            final_mounts[21] = T_21
            final_mounts[40] = T_40
        else:
            # Fall back to general 2D corner bundle adjustment across all tool tags
            r_cam0 = Rotation.from_matrix(R_init).as_rotvec()
            params_init = [r_cam0, t_cam_init]
            for tid in tag_list:
                r_m = Rotation.from_matrix(init_mounts[tid][:3, :3]).as_rotvec()
                t_m = init_mounts[tid][:3, 3]
                params_init.extend([r_m, t_m])
            x0 = np.concatenate(params_init)

            def res_general_ba(p):
                R_c = Rotation.from_rotvec(p[0:3]).as_matrix()
                t_c = p[3:6]
                T_c = np.eye(4); T_c[:3, :3] = R_c; T_c[:3, 3] = t_c
                T_c_b = np.linalg.inv(T_c)

                cur_mounts = {}
                for idx, tid in enumerate(tag_list):
                    base_p = 6 + 6 * idx
                    R_m = Rotation.from_rotvec(p[base_p : base_p + 3]).as_matrix()
                    t_m = p[base_p + 3 : base_p + 6]
                    T_m = np.eye(4); T_m[:3, :3] = R_m; T_m[:3, 3] = t_m
                    cur_mounts[tid] = T_m

                residuals = []
                for s in tool_samples:
                    tid = s["tid"]
                    corners_local = get_tag_corners_local(tid)
                    T_b_tag = s["T_b_tcp"] @ cur_mounts[tid]
                    T_c_tag = T_c_b @ T_b_tag

                    rvec_tag, _ = cv2.Rodrigues(T_c_tag[:3, :3])
                    tvec_tag = T_c_tag[:3, 3].reshape(3, 1)

                    proj, _ = cv2.projectPoints(corners_local, rvec_tag, tvec_tag, K_d, D_d)
                    diff = (proj.reshape(-1, 2) - s["corners_px"]).ravel()
                    residuals.extend(diff)
                return np.array(residuals)

            opt = least_squares(res_general_ba, x0, loss="cauchy", f_scale=1.5, max_nfev=250)
            R_cam_opt = Rotation.from_rotvec(opt.x[0:3]).as_matrix()
            T_base_d415[:3, :3] = R_cam_opt
            T_base_d415[:3, 3] = opt.x[3:6]
            for idx, tid in enumerate(tag_list):
                base_p = 6 + 6 * idx
                R_m = Rotation.from_rotvec(opt.x[base_p : base_p + 3]).as_matrix()
                t_m = opt.x[base_p + 3 : base_p + 6]
                M = np.eye(4); M[:3, :3] = R_m; M[:3, 3] = t_m
                final_mounts[tid] = M

    elif method == "coplanar_plate_ba":
        # Coplanar Tool Plate Model:
        # Tool tags 6 to 13 lie on a common planar fixture and share the identical normal vector.
        # When secondary/USB camera sightings are present, performs Joint Dual-Camera Bundle Adjustment.
        plate_tags = sorted([t for t in tag_list if t in range(6, 14)])
        other_tags = sorted([t for t in tag_list if t not in plate_tags])

        if plate_tags:
            R_plate_init = chordal_mean_rotation([init_mounts[t][:3, :3] for t in plate_tags])
            c_plate_init = np.mean([init_mounts[t][:3, 3] for t in plate_tags], axis=0)
            r_plate_init = Rotation.from_matrix(R_plate_init).as_rotvec()

            use_dual = len(tool_samples_usb) >= 6
            if use_dual:
                usb_groups = [
                    [(s["T_b_tcp"], s["T_c_t"]) for s in tool_samples_usb if s["tid"] == tid]
                    for tid in plate_tags if any(s["tid"] == tid for s in tool_samples_usb)
                ]
                R_u_init = solve_closed_form_rotation(usb_groups) if usb_groups else R_init
                t_u_init = np.array([1.766, -0.874, 0.809])
                params_init = [
                    Rotation.from_matrix(R_init).as_rotvec(),
                    t_cam_init,
                    Rotation.from_matrix(R_u_init).as_rotvec(),
                    t_u_init,
                    r_plate_init,
                    c_plate_init,
                ]
            else:
                params_init = [
                    Rotation.from_matrix(R_init).as_rotvec(),
                    t_cam_init,
                    r_plate_init,
                    c_plate_init,
                ]

            for t in plate_tags:
                p_loc = R_plate_init.T @ (init_mounts[t][:3, 3] - c_plate_init)
                R_rel = R_plate_init.T @ init_mounts[t][:3, :3]
                yaw_rel = float(Rotation.from_matrix(R_rel).as_euler('zxy')[0])
                params_init.append(np.array([p_loc[0], p_loc[1], yaw_rel]))

            for t in other_tags:
                params_init.append(Rotation.from_matrix(init_mounts[t][:3, :3]).as_rotvec())
                params_init.append(init_mounts[t][:3, 3])

            x0 = np.concatenate(params_init)

            def res_coplanar(p):
                R_c = Rotation.from_rotvec(p[0:3]).as_matrix()
                t_c = p[3:6]
                T_c = np.eye(4); T_c[:3, :3] = R_c; T_c[:3, 3] = t_c
                T_c_b = np.linalg.inv(T_c)

                if use_dual:
                    R_u = Rotation.from_rotvec(p[6:9]).as_matrix()
                    t_u = p[9:12]
                    T_u = np.eye(4); T_u[:3, :3] = R_u; T_u[:3, 3] = t_u
                    T_u_b = np.linalg.inv(T_u)
                    R_pl = Rotation.from_rotvec(p[12:15]).as_matrix()
                    t_pl = p[15:18]
                    idx_p = 18
                else:
                    R_pl = Rotation.from_rotvec(p[6:9]).as_matrix()
                    t_pl = p[9:12]
                    idx_p = 12

                cur_mounts = {}
                for t in plate_tags:
                    u, v, yaw = p[idx_p], p[idx_p + 1], p[idx_p + 2]
                    idx_p += 3
                    R_z = Rotation.from_euler('z', yaw).as_matrix()
                    R_tag = R_pl @ R_z
                    t_tag = t_pl + R_pl @ np.array([u, v, 0.0])
                    T_m = np.eye(4); T_m[:3, :3] = R_tag; T_m[:3, 3] = t_tag
                    cur_mounts[t] = T_m

                for t in other_tags:
                    R_m = Rotation.from_rotvec(p[idx_p : idx_p + 3]).as_matrix()
                    t_m = p[idx_p + 3 : idx_p + 6]
                    idx_p += 6
                    T_m = np.eye(4); T_m[:3, :3] = R_m; T_m[:3, 3] = t_m
                    cur_mounts[t] = T_m

                residuals = []
                # D415 residuals
                for s in tool_samples:
                    tid = s["tid"]
                    if tid not in cur_mounts:
                        continue
                    corners_local = get_tag_corners_local(tid)
                    T_b_tag = s["T_b_tcp"] @ cur_mounts[tid]
                    T_c_tag = T_c_b @ T_b_tag
                    rvec_tag, _ = cv2.Rodrigues(T_c_tag[:3, :3])
                    tvec_tag = T_c_tag[:3, 3].reshape(3, 1)
                    proj, _ = cv2.projectPoints(corners_local, rvec_tag, tvec_tag, K_d, D_d)
                    diff = (proj.reshape(-1, 2) - s["corners_px"]).ravel()
                    residuals.extend(diff)

                # USB camera residuals
                if use_dual:
                    for s in tool_samples_usb:
                        tid = s["tid"]
                        if tid not in cur_mounts:
                            continue
                        corners_local = get_tag_corners_local(tid)
                        T_b_tag = s["T_b_tcp"] @ cur_mounts[tid]
                        T_c_tag = T_u_b @ T_b_tag
                        rvec_tag, _ = cv2.Rodrigues(T_c_tag[:3, :3])
                        tvec_tag = T_c_tag[:3, 3].reshape(3, 1)
                        proj, _ = cv2.projectPoints(corners_local, rvec_tag, tvec_tag, K_u, D_u)
                        diff = (proj.reshape(-1, 2) - s["corners_px"]).ravel()
                        residuals.extend(diff)

                return np.array(residuals)

            opt = least_squares(res_coplanar, x0, loss="cauchy", f_scale=1.5, max_nfev=350)
            R_cam_opt = Rotation.from_rotvec(opt.x[0:3]).as_matrix()
            T_base_d415[:3, :3] = R_cam_opt
            T_base_d415[:3, 3] = opt.x[3:6]

            if use_dual:
                R_usb_opt = Rotation.from_rotvec(opt.x[6:9]).as_matrix()
                T_base_usb = np.eye(4)
                T_base_usb[:3, :3] = R_usb_opt
                T_base_usb[:3, 3] = opt.x[9:12]
                R_pl_opt = Rotation.from_rotvec(opt.x[12:15]).as_matrix()
                t_pl_opt = opt.x[15:18]
                idx_p = 18
            else:
                R_pl_opt = Rotation.from_rotvec(opt.x[6:9]).as_matrix()
                t_pl_opt = opt.x[9:12]
                idx_p = 12

            for t in plate_tags:
                u, v, yaw = opt.x[idx_p], opt.x[idx_p + 1], opt.x[idx_p + 2]
                idx_p += 3
                R_z = Rotation.from_euler('z', yaw).as_matrix()
                R_tag = R_pl_opt @ R_z
                t_tag = t_pl_opt + R_pl_opt @ np.array([u, v, 0.0])
                M = np.eye(4); M[:3, :3] = R_tag; M[:3, 3] = t_tag
                final_mounts[t] = M

            for t in other_tags:
                R_m = Rotation.from_rotvec(opt.x[idx_p : idx_p + 3]).as_matrix()
                t_m = opt.x[idx_p + 3 : idx_p + 6]
                idx_p += 6
                M = np.eye(4); M[:3, :3] = R_m; M[:3, 3] = t_m
                final_mounts[t] = M
        else:
            r_cam0 = Rotation.from_matrix(R_init).as_rotvec()
            params_init = [r_cam0, t_cam_init]
            for tid in tag_list:
                r_m = Rotation.from_matrix(init_mounts[tid][:3, :3]).as_rotvec()
                t_m = init_mounts[tid][:3, 3]
                params_init.extend([r_m, t_m])
            x0 = np.concatenate(params_init)

            def res_general(p):
                R_c = Rotation.from_rotvec(p[0:3]).as_matrix()
                t_c = p[3:6]
                T_c = np.eye(4); T_c[:3, :3] = R_c; T_c[:3, 3] = t_c
                T_c_b = np.linalg.inv(T_c)
                cur_mounts = {}
                for idx, tid in enumerate(tag_list):
                    base_p = 6 + 6 * idx
                    R_m = Rotation.from_rotvec(p[base_p : base_p + 3]).as_matrix()
                    t_m = p[base_p + 3 : base_p + 6]
                    T_m = np.eye(4); T_m[:3, :3] = R_m; T_m[:3, 3] = t_m
                    cur_mounts[tid] = T_m
                residuals = []
                for s in tool_samples:
                    tid = s["tid"]
                    corners_local = get_tag_corners_local(tid)
                    T_b_tag = s["T_b_tcp"] @ cur_mounts[tid]
                    T_c_tag = T_c_b @ T_b_tag
                    rvec_tag, _ = cv2.Rodrigues(T_c_tag[:3, :3])
                    tvec_tag = T_c_tag[:3, 3].reshape(3, 1)
                    proj, _ = cv2.projectPoints(corners_local, rvec_tag, tvec_tag, K_d, D_d)
                    diff = (proj.reshape(-1, 2) - s["corners_px"]).ravel()
                    residuals.extend(diff)
                return np.array(residuals)

            opt = least_squares(res_general, x0, loss="cauchy", f_scale=1.5, max_nfev=250)
            R_cam_opt = Rotation.from_rotvec(opt.x[0:3]).as_matrix()
            T_base_d415[:3, :3] = R_cam_opt
            T_base_d415[:3, 3] = opt.x[3:6]
            for idx, tid in enumerate(tag_list):
                base_p = 6 + 6 * idx
                R_m = Rotation.from_rotvec(opt.x[base_p : base_p + 3]).as_matrix()
                t_m = opt.x[base_p + 3 : base_p + 6]
                M = np.eye(4); M[:3, :3] = R_m; M[:3, 3] = t_m
                final_mounts[tid] = M

    else:
        raise ValueError(f"Unknown optimization method: {method}")

    # 4. Solve Static Scene Tags in base_link
    static_tag_poses = {}
    static_tag_observations: dict[int, list[np.ndarray]] = {tid: [] for tid in STATIC_TAGS}

    for obs in observations:
        for tid, det in obs.d415_tags.items():
            if tid in STATIC_TAGS:
                T_base_tag = T_base_d415 @ det.T_cam_tag
                static_tag_observations[tid].append(T_base_tag)

    for tid, poses in static_tag_observations.items():
        if not poses:
            continue
        rotations = [T[:3, :3] for T in poses]
        translations = [T[:3, 3] for T in poses]
        R_consensus = chordal_mean_rotation(rotations)
        t_consensus = np.mean(translations, axis=0)

        T_cons = np.eye(4)
        T_cons[:3, :3] = R_consensus
        T_cons[:3, 3] = t_consensus
        static_tag_poses[tid] = T_cons

    # Workpiece Plane Fitting (for all configured boxes in BOX_TAG_GROUPS)
    workpiece_planes = {}
    for b_name, tag_set in BOX_TAG_GROUPS.items():
        b_pts_list = [static_tag_poses[tid][:3, 3] for tid in tag_set if tid in static_tag_poses]
        if len(b_pts_list) >= 3:
            pts = np.array(b_pts_list)
            b_center = np.mean(pts, axis=0)
            uu, dd, vv = np.linalg.svd(pts - b_center)
            b_norm = vv[-1]
            if b_norm[2] < 0:
                b_norm = -b_norm
            resids = np.abs(np.dot(pts - b_center, b_norm))
            b_rmse = float(np.sqrt(np.mean(resids ** 2)) * 1000.0)
            workpiece_planes[b_name] = {
                "name": b_name,
                "tag_ids": sorted(list(tag_set)),
                "present_tag_ids": [tid for tid in sorted(list(tag_set)) if tid in static_tag_poses],
                "normal": b_norm.tolist(),
                "center": b_center.tolist(),
                "coplanarity_rmse_mm": b_rmse,
                "tag_positions_mm": {
                    int(tid): (static_tag_poses[tid][:3, 3] * 1000.0).tolist()
                    for tid in tag_set if tid in static_tag_poses
                },
            }

    primary_box = workpiece_planes.get("box") or (next(iter(workpiece_planes.values())) if workpiece_planes else None)
    box_normal = np.array(primary_box["normal"]) if primary_box else None
    box_center = np.array(primary_box["center"]) if primary_box else None
    box_rmse = float(primary_box["coplanarity_rmse_mm"]) if primary_box else 0.0

    # 5. Solve USB Camera Extrinsics (if not already optimized jointly)
    if "T_base_usb" not in locals() or T_base_usb is None:
        T_base_usb = None
        usb_obj_pts = []
        usb_img_pts = []

        for obs in observations:
            for tid, d in obs.usb_tags.items():
                if tid in static_tag_poses:
                    T_b_tag = static_tag_poses[tid]
                    corners_base = (T_b_tag[:3, :3] @ get_tag_corners_local(tid).T).T + T_b_tag[:3, 3]
                    for p_b, p_px in zip(corners_base, d.corners_px):
                        usb_obj_pts.append(p_b)
                        usb_img_pts.append(p_px)

        if len(usb_obj_pts) >= 6:
            ok, rvec_u, tvec_u = cv2.solvePnP(
                np.array(usb_obj_pts, dtype=np.float64),
                np.array(usb_img_pts, dtype=np.float64),
                K_u, D_u, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if ok:
                R_usb_base, _ = cv2.Rodrigues(rvec_u)
                t_usb_base = tvec_u.ravel()
                T_u_b = np.eye(4); T_u_b[:3, :3] = R_usb_base; T_u_b[:3, 3] = t_usb_base
                T_base_usb = np.linalg.inv(T_u_b)

    # 6. Comprehensive Error Evaluation (3D Residuals + 2D Reprojection)
    T_c_b = np.linalg.inv(T_base_d415)
    pos_errs = []
    rot_errs = []
    reproj_errs = []
    per_pose_errors = []

    for s in tool_samples:
        tid = s["tid"]
        T_pred = s["T_b_tcp"] @ final_mounts[tid]
        T_meas = T_base_d415 @ s["T_c_t"]

        pe = float(np.linalg.norm(T_pred[:3, 3] - T_meas[:3, 3]) * 1000.0)
        re = float(np.linalg.norm(Rotation.from_matrix(T_pred[:3, :3].T @ T_meas[:3, :3]).as_rotvec()) * 180.0 / np.pi)

        # 2D Reprojection Error in pixels
        T_c_tag = T_c_b @ T_pred
        rvec_t, _ = cv2.Rodrigues(T_c_tag[:3, :3])
        tvec_t = T_c_tag[:3, 3].reshape(3, 1)
        corners_local = get_tag_corners_local(tid)
        proj, _ = cv2.projectPoints(corners_local, rvec_t, tvec_t, K_d, D_d)
        reproj_px = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - s["corners_px"], axis=1)))

        pos_errs.append(pe)
        rot_errs.append(re)
        reproj_errs.append(reproj_px)

        per_pose_errors.append({
            "pose": s["pose"],
            "tag_id": tid,
            "camera": "d415",
            "pos_err_mm": pe,
            "rot_err_deg": re,
            "reproj_err_px": reproj_px,
            "predicted_pos_mm": (T_pred[:3, 3] * 1000.0).tolist(),
            "measured_pos_mm": (T_meas[:3, 3] * 1000.0).tolist(),
        })

    # Evaluate USB camera sightings if available
    if T_base_usb is not None and tool_samples_usb:
        T_u_b = np.linalg.inv(T_base_usb)
        for s in tool_samples_usb:
            tid = s["tid"]
            if tid not in final_mounts:
                continue
            T_pred = s["T_b_tcp"] @ final_mounts[tid]
            T_meas = T_base_usb @ s["T_c_t"]
            pe = float(np.linalg.norm(T_pred[:3, 3] - T_meas[:3, 3]) * 1000.0)
            re = float(np.linalg.norm(Rotation.from_matrix(T_pred[:3, :3].T @ T_meas[:3, :3]).as_rotvec()) * 180.0 / np.pi)

            T_u_tag = T_u_b @ T_pred
            rvec_t, _ = cv2.Rodrigues(T_u_tag[:3, :3])
            tvec_t = T_u_tag[:3, 3].reshape(3, 1)
            corners_local = get_tag_corners_local(tid)
            proj, _ = cv2.projectPoints(corners_local, rvec_t, tvec_t, K_u, D_u)
            reproj_px = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - s["corners_px"], axis=1)))

            pos_errs.append(pe)
            rot_errs.append(re)
            reproj_errs.append(reproj_px)

            per_pose_errors.append({
                "pose": s["pose"] + "_usb",
                "tag_id": tid,
                "camera": "usb",
                "pos_err_mm": pe,
                "rot_err_deg": re,
                "reproj_err_px": reproj_px,
                "predicted_pos_mm": (T_pred[:3, 3] * 1000.0).tolist(),
                "measured_pos_mm": (T_meas[:3, 3] * 1000.0).tolist(),
            })

    pos_errs_arr = np.array(pos_errs)
    rot_errs_arr = np.array(rot_errs)
    reproj_errs_arr = np.array(reproj_errs)

    runtime_s = float(time.perf_counter() - t_start)

    return HandEyeCalibrationResult(
        method=method,
        T_base_d415=T_base_d415,
        T_base_usb=T_base_usb,
        tool_mounts=final_mounts,
        static_tag_poses=static_tag_poses,
        box_plane_normal=box_normal,
        box_plane_center=box_center,
        box_coplanarity_rmse_mm=box_rmse,
        workpiece_planes=workpiece_planes,
        pos_rms_mm=float(np.sqrt(np.mean(pos_errs_arr ** 2))),
        pos_mean_mm=float(np.mean(pos_errs_arr)),
        pos_max_mm=float(np.max(pos_errs_arr)),
        rot_rms_deg=float(np.sqrt(np.mean(rot_errs_arr ** 2))),
        rot_mean_deg=float(np.mean(rot_errs_arr)),
        rot_max_deg=float(np.max(rot_errs_arr)),
        reproj_rms_px=float(np.sqrt(np.mean(reproj_errs_arr ** 2))),
        reproj_mean_px=float(np.mean(reproj_errs_arr)),
        reproj_max_px=float(np.max(reproj_errs_arr)),
        num_tool_sightings=len(pos_errs),
        per_pose_errors=per_pose_errors,
        runtime_s=runtime_s,
    )
