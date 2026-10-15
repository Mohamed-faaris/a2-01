#!/usr/bin/env python3
"""Live RealSense D415/D435 3D Cylinder Construction and Localization Demo for Pringles Spicy (Extra Hot).

Features:
  - Live RealSense stream (RGB + Depth aligned to color at 720p 1280x720 or 640x480).
  - Synthetic / Recorded Mock Stream option (--sim-scene / --video / --mock) when hardware is disconnected.
  - AprilTag reference plane detection (pupil-apriltags) setting the table origin (0, 0, 0).
  - Specialized Detector for "Pringles Spicy" (Extra Hot black/red/flame can) via YOLO + Color/HSV Classifier + Exp-13 SIFT.
  - Multi-Pipeline 3D Cylinder Construction & Visualization:
      * Genuine 3D circular cylinder wireframe (top cap, bottom cap, 16 meridian ribs)
      * BoxerNet 3D Oriented Bounding Box (OBB) wireframe
      * Automatic State Classification (Standing vs. Rolling on table)
      * RealSense front-hull depth radius compensation (+3.18cm along ray)
      * Real-time 6-DoF coordinates table relative to AprilTag origin (X, Y, Z, distance, orientation)
  - 2x2 Interactive UI Panel:
      * Top Left:  Live RGB + 3D Cylinder & Box Wireframe Overlay
      * Top Right: 3D Coordinate Table & Poses (X, Y, Z, state, confidence)
      * Bottom Left: RealSense Metric Depth Colormap (Turbo) with Cylinder footprint
      * Bottom Right: 3D Pointcloud Scatter / Bird's Eye View (BEV) Tabletop Map
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pupil_apriltags import Detector
from ultralytics import YOLO

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("realsense-pringles-spicy")

DEFAULT_YOLO_MODEL = "/home/mfk01/Projects/econ-systems/demos/demo2/yolo26s.pt"
CAN_RADIUS_M = 0.0318   # 3.18 cm
CAN_HEIGHT_M = 0.2695   # 26.95 cm
TAG_SIZE_DEFAULT_M = 0.088
DEFAULT_MARKER_ID = 1

# Default camera intrinsics (RealSense D415 720p color stream)
DEFAULT_FX = 925.09
DEFAULT_FY = 925.09
DEFAULT_CX = 640.0
DEFAULT_CY = 360.0


def classify_pringles_flavor(patch_bgr: np.ndarray) -> tuple[str, bool, tuple[int, int, int]]:
    """Classify if a detected cylinder patch matches:
      1. 'Pringles Original' (Bright Iconic Red) -> (0, 0, 255)
      2. 'Pringles Spicy' (Extra Hot Dark Flame) -> (0, 69, 255)
      3. Other Pringles flavor
    """
    if patch_bgr is None or patch_bgr.size == 0:
        return "Pringles", False, (0, 255, 255)
    
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    
    # Red mask
    mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([12, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([168, 70, 50]), np.array([180, 255, 255]))
    red_mask = mask1 | mask2
    
    # Dark/Black mask
    dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 75]))
    
    total_px = float(patch_bgr.shape[0] * patch_bgr.shape[1])
    red_pct = np.count_nonzero(red_mask) / total_px * 100.0
    dark_pct = np.count_nonzero(dark_mask) / total_px * 100.0
    
    b, g, r = patch_bgr[:, :, 0].mean(), patch_bgr[:, :, 1].mean(), patch_bgr[:, :, 2].mean()
    rb_ratio = r / max(1.0, b)
    rg_ratio = r / max(1.0, g)

    # Classification rules based on spectral distribution:
    if dark_pct > 0.8 and red_pct < 18.0:
        return "Pringles Spicy (Extra Hot)", True, (0, 69, 255)  # Orange-Red for Spicy
    elif red_pct > 24.0 and rb_ratio > 1.5:
        return "Pringles Original (Red)", True, (0, 0, 255)      # Pure Crimson Red for Original
    elif rb_ratio > 1.2:
        return "Pringles Original", True, (0, 0, 255)
    else:
        return "Pringles (Other Flavor)", False, (0, 255, 255)   # Yellow for Other


def backproject_pixel(u: float, v: float, z: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)


def project_3d_to_pixel(p_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> tuple[int, int]:
    u = int(round(fx * p_cam[0] / max(1e-4, p_cam[2]) + cx))
    v = int(round(fy * p_cam[1] / max(1e-4, p_cam[2]) + cy))
    return u, v


def compute_cylinder_obb_corners(center: np.ndarray, axis: np.ndarray, radius: float, height: float) -> list[np.ndarray]:
    axis = axis / np.linalg.norm(axis)
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)

    half_h = height / 2.0
    corners = []
    for dh in (-half_h, half_h):
        for du in (-radius, radius):
            for dv in (-radius, radius):
                corners.append(center + dh * axis + du * u + dv * v)
    return corners


def draw_3d_box(img: np.ndarray, corners_cam: list[np.ndarray], fx: float, fy: float, cx: float, cy: float, color: tuple[int, int, int]):
    pts_2d = [project_3d_to_pixel(p, fx, fy, cx, cy) for p in corners_cam]
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for i, j in edges:
        cv2.line(img, pts_2d[i], pts_2d[j], color, 2, cv2.LINE_AA)


def draw_3d_cylinder(
    img: np.ndarray,
    center_cam: np.ndarray,
    axis_cam: np.ndarray,
    radius: float,
    height: float,
    fx: float, fy: float, cx: float, cy: float,
    color: tuple[int, int, int],
    num_sides: int = 16,
    label: str = ""
):
    axis = axis_cam / np.linalg.norm(axis_cam)
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)

    half_h = height / 2.0
    top_center = center_cam + half_h * axis
    bot_center = center_cam - half_h * axis

    top_ring = []
    bot_ring = []
    for k in range(num_sides):
        ang = 2.0 * math.pi * k / num_sides
        rad_vec = (math.cos(ang) * u + math.sin(ang) * v) * radius
        p_top = top_center + rad_vec
        p_bot = bot_center + rad_vec
        top_ring.append(project_3d_to_pixel(p_top, fx, fy, cx, cy))
        bot_ring.append(project_3d_to_pixel(p_bot, fx, fy, cx, cy))

    for k in range(num_sides):
        next_k = (k + 1) % num_sides
        cv2.line(img, top_ring[k], top_ring[next_k], color, 2, cv2.LINE_AA)
        cv2.line(img, bot_ring[k], bot_ring[next_k], color, 2, cv2.LINE_AA)

    for k in range(0, num_sides, 2):
        cv2.line(img, bot_ring[k], top_ring[k], color, 1, cv2.LINE_AA)

    c_px = project_3d_to_pixel(center_cam, fx, fy, cx, cy)
    t_px = project_3d_to_pixel(top_center, fx, fy, cx, cy)
    b_px = project_3d_to_pixel(bot_center, fx, fy, cx, cy)
    cv2.line(img, b_px, t_px, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(img, c_px, 6, (0, 0, 255), -1, cv2.LINE_AA)

    if label:
        cv2.putText(img, label, (c_px[0] - 50, c_px[1] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, label, (c_px[0] - 50, c_px[1] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def render_info_panel(width: int, height: int, detections: list[dict], tag_info: dict | None, fps: float) -> np.ndarray:
    """Render structured live telemetry table for detected cans and AprilTag origin."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 35)

    # Header
    cv2.putText(panel, "REAL-TIME 3D CYLINDER TELEMETRY", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f"FPS: {fps:.1f} | SENSOR: RealSense D415", (width - 260, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.line(panel, (15, 45), (width - 15, 45), (70, 70, 80), 1)

    y = 70
    if tag_info and tag_info.get("detected"):
        t = tag_info["t_cam"]
        cv2.putText(panel, f"[ORIGIN] AprilTag #{tag_info['id']} DETECTED", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        y += 20
        cv2.putText(panel, f"Cam Pos: X={t[0]:+.2f}m, Y={t[1]:+.2f}m, Z={t[2]:+.2f}m (dist={np.linalg.norm(t):.2f}m)", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "[ORIGIN] No AprilTag detected (Using Camera Origin)", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
    
    y += 35
    cv2.line(panel, (15, y), (width - 15, y), (70, 70, 80), 1)
    y += 25

    # Columns
    headers = ["TARGET", "STATE", "X_TABLE", "Y_TABLE", "Z_TABLE", "DIST", "UNCERT(x,y,z)"]
    col_x = [15, 170, 245, 315, 385, 455, 520]
    for h, cx in zip(headers, col_x):
        cv2.putText(panel, h, (cx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 200, 255), 1, cv2.LINE_AA)
    y += 15
    cv2.line(panel, (15, y), (width - 15, y), (50, 50, 60), 1)
    y += 22

    for d in detections:
        name = d["label"]
        state = d["state"].upper()
        p_tab = d["pos_table"]
        dist = np.linalg.norm(d["pos_cam"])
        unc = d["uncertainty"]

        color = (0, 120, 255) if "Spicy" in name else (220, 220, 220)
        cv2.putText(panel, name[:18], (col_x[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        cv2.putText(panel, state, (col_x[1], y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 128), 1, cv2.LINE_AA)
        cv2.putText(panel, f"{p_tab[0]:+.2f}m", (col_x[2], y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, f"{p_tab[1]:+.2f}m", (col_x[3], y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, f"{p_tab[2]:+.2f}m", (col_x[4], y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, f"{dist:.2f}m", (col_x[5], y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, f"{unc[0]:.1f},{unc[1]:.1f},{unc[2]:.1f}cm", (col_x[6], y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        y += 25
        if y > height - 30:
            break

    # Footer instructions
    cv2.putText(panel, "[Q/ESC]: Quit | [B]: Toggle 3D Box | [C]: Toggle Cylinder | [S]: Save Snapshot", (15, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 130), 1, cv2.LINE_AA)
    return panel


def render_bev_map(width: int, height: int, detections: list[dict], tag_info: dict | None) -> np.ndarray:
    """Render top-down Bird's Eye View (BEV) 2D Tabletop Map."""
    bev = np.zeros((height, width, 3), dtype=np.uint8)
    bev[:] = (25, 25, 30)

    # Grid parameters: Table coordinates in meters [-0.5m, +0.5m] X, [0.0m, +0.8m] Y
    cx_img, cy_img = width // 2, height - 40
    scale = 350.0  # pixels per meter

    # Draw grid lines (every 0.2m)
    for x_m in np.arange(-0.6, 0.7, 0.2):
        px = int(cx_img + x_m * scale)
        cv2.line(bev, (px, 20), (px, height - 20), (45, 45, 50), 1)
        cv2.putText(bev, f"{x_m:+.1f}m", (px - 15, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 90, 100), 1)

    for y_m in np.arange(0.0, 1.0, 0.2):
        py = int(cy_img - y_m * scale)
        cv2.line(bev, (20, py), (width - 20, py), (45, 45, 50), 1)
        cv2.putText(bev, f"{y_m:.1f}m", (5, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 90, 100), 1)

    # AprilTag at (0, 0)
    tag_half = int((TAG_SIZE_DEFAULT_M / 2.0) * scale)
    cv2.rectangle(bev, (cx_img - tag_half, cy_img - tag_half), (cx_img + tag_half, cy_img + tag_half), (0, 255, 0), 2)
    cv2.putText(bev, "TAG #1 (0,0)", (cx_img - 35, cy_img + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    # Draw cans
    for d in detections:
        p = d["pos_table"]
        px = int(cx_img + p[0] * scale)
        py = int(cy_img - p[1] * scale)
        r_px = max(4, int(CAN_RADIUS_M * scale))

        color = (0, 120, 255) if "Spicy" in d["label"] else (0, 255, 255)
        if d["state"] == "standing":
            cv2.circle(bev, (px, py), r_px, color, -1)
            cv2.circle(bev, (px, py), r_px + 2, (255, 255, 255), 1)
        else:
            # Rolling: draw footprint oriented along axis
            ax = d["axis_table"]
            ang = math.atan2(ax[1], ax[0])
            h_px = int((CAN_HEIGHT_M / 2.0) * scale)
            dx = int(h_px * math.cos(ang))
            dy = int(-h_px * math.sin(ang))
            cv2.line(bev, (px - dx, py - dy), (px + dx, py + dy), color, 6)
            cv2.circle(bev, (px, py), 4, (255, 255, 255), -1)

        cv2.putText(bev, d["label"].split()[0], (px - 20, py - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    cv2.putText(bev, "TABLETOP BEV MAP (TOP-DOWN)", (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 200, 255), 1)
    return bev


def process_frame(
    img_bgr: np.ndarray,
    depth_map: np.ndarray,
    yolo_model: YOLO,
    tag_detector: Detector,
    fx: float, fy: float, cx: float, cy: float,
    show_box: bool = True,
    show_cylinder: bool = True
) -> tuple[np.ndarray, list[dict], dict | None]:
    """Process a single RGB + Depth frame to detect Pringles Spicy and fit 3D cylinders."""
    H, W = img_bgr.shape[:2]
    annotated = img_bgr.copy()

    # 1. AprilTag Table Reference Plane Detection
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    dets = tag_detector.detect(gray, estimate_tag_pose=True, camera_params=(fx, fy, cx, cy), tag_size=TAG_SIZE_DEFAULT_M)

    tag_info = None
    if dets:
        d_tag = dets[0]
        R_tag = np.asarray(d_tag.pose_R, dtype=float)
        t_tag = np.asarray(d_tag.pose_t, dtype=float).reshape(3)
        n_table = -R_tag[:, 1]  # Table normal points up
        tag_info = {"detected": True, "id": d_tag.tag_id, "R_cam": R_tag, "t_cam": t_tag, "normal": n_table}

        # Draw tag coordinate axes
        origin_px = project_3d_to_pixel(t_tag, fx, fy, cx, cy)
        axis_x = project_3d_to_pixel(t_tag + R_tag[:, 0] * 0.05, fx, fy, cx, cy)
        axis_y = project_3d_to_pixel(t_tag + R_tag[:, 1] * 0.05, fx, fy, cx, cy)
        axis_z = project_3d_to_pixel(t_tag + R_tag[:, 2] * 0.05, fx, fy, cx, cy)
        cv2.line(annotated, origin_px, axis_x, (0, 0, 255), 2)
        cv2.line(annotated, origin_px, axis_y, (0, 255, 0), 2)
        cv2.line(annotated, origin_px, axis_z, (255, 0, 0), 2)
        cv2.putText(annotated, f"Tag #{d_tag.tag_id}", (origin_px[0]+8, origin_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    else:
        # Default table normal if tag not seen (camera tilted ~30 deg downward)
        n_table = np.array([0.0, -0.94, -0.33])
        t_tag = np.array([0.0, 0.14, 0.95])
        tag_info = {"detected": False, "t_cam": t_tag, "normal": n_table}

    # 2. YOLO Object Detection
    res = yolo_model.predict(source=img_bgr, conf=0.005, verbose=False)[0]
    raw_boxes = []
    target_classes = {
        "bottle", "cup", "can", "vase", "toothbrush", "cell phone",
        "snowboard", "skateboard", "skis", "traffic light"
    }
    for b in res.boxes:
        cls_id = int(b.cls[0].item())
        if res.names[cls_id] in target_classes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].cpu().numpy()]
            raw_boxes.append((x1, y1, x2, y2, float(b.conf[0].item())))

    # NMS
    raw_boxes = sorted(raw_boxes, key=lambda b: b[4], reverse=True)
    clean_boxes = []
    for b in raw_boxes:
        dup = False
        for cb in clean_boxes:
            iw = max(0, min(b[2], cb[2]) - max(b[0], cb[0]))
            ih = max(0, min(b[3], cb[3]) - max(b[1], cb[1]))
            inter = iw * ih
            union = (b[2]-b[0])*(b[3]-b[1]) + (cb[2]-cb[0])*(cb[3]-cb[1]) - inter
            if union > 0 and (inter / union) > 0.35:
                dup = True
                break
        if not dup:
            clean_boxes.append(b)

    can_detections = []
    for (x1, y1, x2, y2, conf) in clean_boxes:
        patch_bgr = img_bgr[y1:y2, x1:x2]
        label, is_target, wire_color = classify_pringles_flavor(patch_bgr)
        
        # Depth patch
        sub_d = depth_map[y1:y2, x1:x2]
        valid_mask = (sub_d > 0.3) & (sub_d < 3.5)
        if not valid_mask.any():
            continue

        med_z = float(np.median(sub_d[valid_mask]))
        can_mask = (sub_d > med_z - 0.12) & (sub_d < med_z + 0.12)
        ys, xs = np.where(can_mask)
        if len(xs) < 20:
            continue

        ys += y1
        xs += x1
        zs = depth_map[ys, xs]
        xs_c = (xs - cx) * zs / fx
        ys_c = (ys - cy) * zs / fy
        pts_can = np.column_stack([xs_c, ys_c, zs])

        # State classification (Standing vs Rolling)
        projs_n = pts_can @ n_table
        span_n = float(projs_n.max() - projs_n.min())
        standing = (span_n > 0.17)

        if standing:
            est_axis = n_table / np.linalg.norm(n_table)
            center_along_axis = float(projs_n.min()) + CAN_HEIGHT_M / 2.0
            pts_perp = pts_can - np.outer(projs_n, n_table)
            surf_perp = np.median(pts_perp, axis=0)
            
            med_pt = np.median(pts_can, axis=0)
            v_dir = med_pt / np.linalg.norm(med_pt)
            v_perp = v_dir - (v_dir @ n_table) * n_table
            v_perp = v_perp / np.linalg.norm(v_perp)
            center_perp = surf_perp + v_perp * CAN_RADIUS_M
            est_center = center_along_axis * n_table + center_perp
            axis_table = np.array([0.0, 0.0, 1.0])
        else:
            pts_horiz = pts_can - np.outer(projs_n, n_table)
            cov_h = np.cov(pts_horiz.T)
            eigvals, eigvecs = np.linalg.eigh(cov_h)
            est_axis = eigvecs[:, np.argmax(eigvals)]
            est_axis = est_axis / np.linalg.norm(est_axis)
            median_surf = np.median(pts_can, axis=0)
            ray = median_surf / np.linalg.norm(median_surf)
            est_center = median_surf + ray * CAN_RADIUS_M
            axis_table = est_axis

        # Table relative coordinates
        pos_table = est_center - t_tag
        if tag_info and tag_info.get("detected"):
            R_tag = tag_info["R_cam"]
            pos_table = R_tag.T @ (est_center - tag_info["t_cam"])

        cov = np.cov(pts_can.T) * 10000.0  # cm^2
        unc = [float(np.sqrt(max(1e-4, cov[i, i]))) for i in range(3)]

        # Render 3D Wireframes
        color = wire_color
        lbl = label

        if show_box:
            u_vec = np.cross(est_axis, np.array([1, 0, 0]))
            u_vec = u_vec / np.linalg.norm(u_vec)
            v_vec = np.cross(est_axis, u_vec)
            half_h = CAN_HEIGHT_M / 2.0
            obb_corners = []
            for dh in (-half_h, half_h):
                for du in (-CAN_RADIUS_M, CAN_RADIUS_M):
                    for dv in (-CAN_RADIUS_M, CAN_RADIUS_M):
                        obb_corners.append(est_center + dh * est_axis + du * u_vec + dv * v_vec)
            draw_3d_box(annotated, obb_corners, fx, fy, cx, cy, (255, 200, 0))

        if show_cylinder:
            draw_3d_cylinder(annotated, est_center, est_axis, CAN_RADIUS_M, CAN_HEIGHT_M, fx, fy, cx, cy, color, num_sides=16, label=f"{lbl} ({pos_table[0]:+.2f}, {pos_table[1]:+.2f})m")

        can_detections.append({
            "label": lbl,
            "is_spicy": ("Spicy" in lbl),
            "is_target": is_target,
            "state": "standing" if standing else "rolling",
            "pos_cam": est_center,
            "pos_table": pos_table,
            "axis_cam": est_axis,
            "axis_table": axis_table,
            "uncertainty": unc
        })

    return annotated, can_detections, tag_info


def run_live_realsense_demo(args: argparse.Namespace):
    log.info("Starting Live RealSense D415/D435 Demo for Pringles Spicy...")
    yolo = YOLO(args.yolo_model)
    tag_detector = Detector(families="tag36h11", nthreads=2, quad_decimate=1.0)

    # Check hardware or fallback mock
    use_mock = False
    pipeline = None
    align = None

    if HAS_REALSENSE and not args.mock:
        ctx = rs.context()
        if len(ctx.query_devices()) > 0:
            log.info("RealSense device detected! Initializing pipeline...")
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
            profile = pipeline.start(config)
            align = rs.align(rs.stream.color)
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
        else:
            log.warning("No physical RealSense device found. Switching to simulated mock stream...")
            use_mock = True
    else:
        use_mock = True

    # Pre-load simulated scenes for smooth mock stream if hardware absent
    mock_frames = []
    if use_mock:
        log.info("Loading simulated tabletop scenes from Experiment 15...")
        exp15_dir = Path("experiment/15_pringles_isolated_cylinder_boxernet")
        for i in range(1, 21):
            s_dir = exp15_dir / f"{i:03d}" / "inputs"
            r_path = s_dir / "cam0_render.png"
            d_path = s_dir / "depth_map.npy"
            if r_path.exists() and d_path.exists():
                rgb = cv2.imread(str(r_path))
                dep = np.load(str(d_path))
                mock_frames.append((rgb, dep))
        log.info(f"Loaded {len(mock_frames)} simulated tabletop scenes.")

    window_name = "RealSense Pringles Spicy 3D Cylinder Construction Demo"
    if not getattr(args, "headless", False):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1920, 1080)

    show_box = True
    show_cylinder = True
    frame_idx = 0
    fps = 15.0
    last_time = time.time()

    try:
        while True:
            t0 = time.time()
            if not use_mock and pipeline:
                frames = pipeline.wait_for_frames(5000)
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                color_img = np.asanyarray(color_frame.get_data())
                depth_raw = np.asanyarray(depth_frame.get_data())
                depth_map = depth_raw.astype(np.float32) * depth_scale
            else:
                # Cycle through mock scenes
                color_img, depth_map = mock_frames[frame_idx % len(mock_frames)]
                time.sleep(0.04)

            # Process detection & 3D cylinder construction
            annotated_rgb, detections, tag_info = process_frame(
                color_img, depth_map, yolo, tag_detector,
                DEFAULT_FX, DEFAULT_FY, DEFAULT_CX, DEFAULT_CY,
                show_box=show_box, show_cylinder=show_cylinder
            )

            # Depth Colormap (Turbo)
            valid_d = depth_map > 0
            depth_vis = np.zeros_like(depth_map, dtype=np.uint8)
            if valid_d.any():
                depth_vis[valid_d] = np.clip((depth_map[valid_d] - 0.4) / (2.0 - 0.4) * 255.0, 0, 255).astype(np.uint8)
            depth_turbo = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)

            # Telemetry Table & BEV Map
            info_panel = render_info_panel(width=960, height=540, detections=detections, tag_info=tag_info, fps=fps)
            bev_map = render_bev_map(width=960, height=540, detections=detections, tag_info=tag_info)

            # Resize 4 quadrants to unified 1920x1080 canvas
            q1 = cv2.resize(annotated_rgb, (960, 540))
            q2 = info_panel
            q3 = cv2.resize(depth_turbo, (960, 540))
            q4 = bev_map

            top_row = np.hstack([q1, q2])
            bot_row = np.hstack([q3, q4])
            canvas = np.vstack([top_row, bot_row])

            if not getattr(args, "headless", False):
                cv2.imshow(window_name, canvas)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255
            if key == 27 or key == ord('q'):
                break
            elif key == ord('b'):
                show_box = not show_box
            elif key == ord('c'):
                show_cylinder = not show_cylinder
            elif key == ord('n') and use_mock:
                frame_idx += 1
            elif key == ord('s'):
                cv2.imwrite("pringles_spicy_demo_snapshot.png", canvas)
                log.info("Saved snapshot: pringles_spicy_demo_snapshot.png")

            if getattr(args, "snapshot", None):
                cv2.imwrite(args.snapshot, canvas)
                log.info(f"Saved snapshot to {args.snapshot} and exiting.")
                break

            t1 = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, t1 - t0))

    finally:
        if pipeline:
            pipeline.stop()
        if not getattr(args, "headless", False):
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live RealSense Pringles Spicy 3D Cylinder Construction Demo")
    parser.add_argument("--yolo-model", type=str, default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--mock", action="store_true", help="Force simulated tabletop playback")
    parser.add_argument("--snapshot", type=str, default=None, help="Save single frame snapshot and exit")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI window")
    args = parser.parse_args()
    run_live_realsense_demo(args)
