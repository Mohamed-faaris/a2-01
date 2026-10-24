#!/usr/bin/env python3
"""Visualization and annotation overlay tools for Hand-Eye-0 Demo."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from detector import PoseObservation
from solver import HandEyeCalibrationResult


def project_point(p_cam: np.ndarray, K: np.ndarray, D: np.ndarray) -> tuple[int, int] | None:
    """Projects a 3D point in camera frame to pixel integer coordinates."""
    if p_cam[2] <= 0.05:
        return None
    try:
        px, _ = cv2.projectPoints(p_cam.reshape(1, 3), np.zeros(3), np.zeros(3), K, D)
        uv = px.reshape(2)
        if not np.isfinite(uv).all() or abs(uv[0]) > 25000 or abs(uv[1]) > 25000:
            return None
        return (int(round(uv[0])), int(round(uv[1])))
    except Exception:
        return None


def draw_axes(img: np.ndarray, T_c_f: np.ndarray, K: np.ndarray, D: np.ndarray, length_m: float = 0.05):
    """Draws RGB coordinate axes (X red, Y green, Z blue) at pose T_c_f."""
    o_cam = T_c_f[:3, 3]
    R_cam = T_c_f[:3, :3]
    o_px = project_point(o_cam, K, D)
    if o_px is None:
        return

    cols = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X, Y, Z
    for i in range(3):
        axis_end_cam = o_cam + R_cam[:, i] * length_m
        e_px = project_point(axis_end_cam, K, D)
        if e_px is not None:
            cv2.arrowedLine(img, o_px, e_px, cols[i], 2, cv2.LINE_AA, tipLength=0.25)


def annotate_pose_frame(
    raw_img: np.ndarray,
    obs: PoseObservation,
    result: HandEyeCalibrationResult,
    camera_name: str,
    K: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """Annotates detection boxes, coordinate frames, TCP projection, and residual error vectors."""
    vis = raw_img.copy()
    h_img, w_img = vis.shape[:2]

    tags = obs.d415_tags if camera_name == "d415" else obs.usb_tags
    T_b_c = result.T_base_d415 if camera_name == "d415" else result.T_base_usb
    if T_b_c is None:
        return vis

    T_c_b = np.linalg.inv(T_b_c)

    # 1. Draw raw detected tags
    for tid, d in tags.items():
        pts = d.corners_px.astype(np.int32)
        if tid in (20, 21, 40):
            col = (255, 100, 255)  # Magenta for tool tags
            label = f"TOOL Tag {tid}"
        elif tid in (42, 43, 44, 45):
            col = (0, 165, 255)   # Orange for box tags
            label = f"BOX Tag {tid}"
        else:
            col = (0, 255, 120)   # Green for base tags
            label = f"BASE Tag {tid}"

        cv2.polylines(vis, [pts], isClosed=True, color=col, thickness=2, lineType=cv2.LINE_AA)
        c = tuple(d.center_px.astype(np.int32))
        cv2.circle(vis, c, 4, col, -1)
        draw_axes(vis, d.T_cam_tag, K, D, length_m=0.04)
        cv2.putText(vis, label, (pts[0][0], max(20, pts[0][1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)

    # 2. Draw TCP Forward Kinematics projection
    # T_c_tcp = T_c_b * T_b_tcp
    T_c_tcp = T_c_b @ obs.T_base_tcp
    p_tcp_px = project_point(T_c_tcp[:3, 3], K, D)
    if p_tcp_px is not None:
        cv2.circle(vis, p_tcp_px, 7, (0, 255, 255), -1)  # Yellow circle at TCP
        cv2.circle(vis, p_tcp_px, 10, (255, 255, 255), 1)
        draw_axes(vis, T_c_tcp, K, D, length_m=0.06)
        cv2.putText(vis, f"ROBOT TCP [{obs.T_base_tcp[0,3]:+.2f}, {obs.T_base_tcp[1,3]:+.2f}, {obs.T_base_tcp[2,3]:+.2f}]m",
                    (p_tcp_px[0] + 12, p_tcp_px[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    # 3. For tool tags: reproject predicted tag location and draw residual vector
    for tid, m_T in result.tool_mounts.items():
        if tid in tags:
            # Predicted tag in cam: T_c_b * T_b_tcp * T_tcp_tag
            T_c_pred = T_c_b @ obs.T_base_tcp @ m_T
            pred_px = project_point(T_c_pred[:3, 3], K, D)
            meas_px = tuple(tags[tid].center_px.astype(np.int32))

            if pred_px is not None:
                cv2.circle(vis, pred_px, 5, (0, 0, 255), 2)  # Red circle at predicted center
                # Line connecting measured and predicted
                cv2.line(vis, meas_px, pred_px, (0, 0, 255), 2, cv2.LINE_AA)
                # Compute 3D error
                err_mm = np.linalg.norm(T_c_pred[:3, 3] - tags[tid].T_cam_tag[:3, 3]) * 1000.0
                cv2.putText(vis, f"err: {err_mm:.1f}mm", (pred_px[0] + 8, pred_px[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 100, 255), 1, cv2.LINE_AA)

    # Header badge
    cv2.rectangle(vis, (12, 12), (520, 64), (25, 28, 35), -1)
    cv2.putText(vis, f"Pose: {obs.pose_name} | Camera: {camera_name.upper()}", (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2, cv2.LINE_AA)
    n_tool = sum(1 for tid in tags if tid in (20, 21, 40))
    n_static = len(tags) - n_tool
    cv2.putText(vis, f"Tags Detected: {len(tags)} (Tool: {n_tool}, Static: {n_static})", (20, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 220), 1, cv2.LINE_AA)

    return vis


def save_annotated_gallery(
    observations: list[PoseObservation],
    result: HandEyeCalibrationResult,
    K_d415: np.ndarray,
    D_d415: np.ndarray,
    K_usb: np.ndarray,
    D_usb: np.ndarray,
    output_dir: Path,
):
    """Generates and saves visual verification images for all poses."""
    output_dir.mkdir(parents=True, exist_ok=True)
    d415_dir = output_dir / "d415"
    usb_dir = output_dir / "usb"
    d415_dir.mkdir(parents=True, exist_ok=True)
    usb_dir.mkdir(parents=True, exist_ok=True)

    for obs in observations:
        if obs.d415_path and obs.d415_path.is_file():
            img_d = cv2.imread(str(obs.d415_path))
            if img_d is not None:
                vis_d = annotate_pose_frame(img_d, obs, result, "d415", K_d415, D_d415)
                cv2.imwrite(str(d415_dir / f"{obs.pose_name}.png"), vis_d)

        if obs.usb_path and obs.usb_path.is_file():
            img_u = cv2.imread(str(obs.usb_path))
            if img_u is not None:
                vis_u = annotate_pose_frame(img_u, obs, result, "usb", K_usb, D_usb)
                cv2.imwrite(str(usb_dir / f"{obs.pose_name}.png"), vis_u)
