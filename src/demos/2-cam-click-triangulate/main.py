#!/usr/bin/env python3
"""Two-Camera Interactive Manual Click Point Triangulator with Automatic Best-Tag Selection.

Features:
  - Connects to 2 cameras with 10 FPS zero-latency background buffer-draining grabbers.
  - Detects all AprilTags visible in both cameras.
  - Automatically selects the BEST reference tag (lowest pose error, highest decision margin, minimum grazing tilt).
    (Or allows manual override via CLI --tag-id).
  - Interactive Mouse Clicking:
      - Click any point on Camera 1 (Top-Left) to place marker P1.
      - Click the corresponding point on Camera 2 (Bottom-Left) to place marker P2.
      - Left-click to place/update point.
      - Right-click or press 'c' to clear current clicked points.
  - Triangulation & Real-Time Error Analysis:
      - Computes 3D coordinates (X, Y, Z, Euclidean distance) relative to the Best AprilTag (corner or center).
      - Computes ray skew optical difference (ray_diff_cm).
      - Computes ray convergence angle (deg).
      - Computes estimated depth uncertainty bound (est_err_cm).
      - Displays epipolar line / crosshairs and real-time dashboard readout.
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from pupil_apriltags import Detector

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("2cam-click-triangulator")

try:
    from common.markers import DEFAULT_MARKER_SIZES_M, DEFAULT_TAG_FAMILY, DEFAULT_TAG_SIZE_M
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from common.markers import DEFAULT_MARKER_SIZES_M, DEFAULT_TAG_FAMILY, DEFAULT_TAG_SIZE_M


@dataclass
class CameraCalibration:
    name: str
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]
    device_node: str | None = None
    card_name: str | None = None
    stream_type: str | None = None

    @classmethod
    def from_npz(cls, npz_path: str | Path, expected_stream_type: str = "RGB / Color") -> CameraCalibration:
        data = np.load(npz_path, allow_pickle=True)
        K = np.asarray(data["camera_matrix"], dtype=np.float64)
        D = np.asarray(data["dist_coeffs"], dtype=np.float64)
        size = tuple(int(v) for v in data["image_size"])
        name = str(data.get("camera_name", Path(npz_path).stem))
        node = str(data.get("device_node")) if "device_node" in data else None
        card = str(data.get("card_name")) if "card_name" in data else None
        stype = str(data.get("stream_type")) if "stream_type" in data else None
        calib = cls(name=name, camera_matrix=K, dist_coeffs=D, image_size=size,
                    device_node=node, card_name=card, stream_type=stype)
        cls._check_stream_type(calib, expected_stream_type)
        return calib

    @classmethod
    def fallback_fov(cls, name: str, width: int = 1280, height: int = 720, hfov_deg: float = 60.0) -> CameraCalibration:
        fx = (width / 2.0) / np.tan(np.radians(hfov_deg / 2.0))
        K = np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        D = np.zeros(5, dtype=np.float64)
        return cls(name=name, camera_matrix=K, dist_coeffs=D, image_size=(width, height), stream_type="RGB / Color (Fallback FOV)")

    @staticmethod
    def _check_stream_type(calib: CameraCalibration, expected_stream_type: str | None = None) -> None:
        res_str = f" @ {calib.image_size[0]}x{calib.image_size[1]}" if calib.image_size else ""
        if calib.stream_type:
            log.info(f"Loaded calibration '{calib.name}' stream type: '{calib.stream_type}'{res_str} (Card: {calib.card_name or 'Unknown'})")
            is_color_calib = any(c in calib.stream_type.lower() for c in ("color", "rgb", "bgr", "mjpeg", "yuyv"))
            is_expected_color = any(c in (expected_stream_type or "").lower() for c in ("color", "rgb", "bgr"))
            if is_expected_color and not is_color_calib:
                log.warning(f"Stream type mismatch for {calib.name}! Expected color stream, but got '{calib.stream_type}'.")
            else:
                log.info(f"Calibration {calib.name} stream type verified: '{calib.stream_type}'.")
        else:
            log.warning(f"Calibration {calib.name} does not specify 'stream_type'{res_str}.")


class NonQueuingCameraReader:
    """Spawns background grab thread that constantly drains V4L2 buffers,
    ensuring .get_latest_frame() always retrieves the zero-latency real-time frame."""

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.lock = threading.Lock()
        self.latest_frame: np.ndarray | None = None
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest_frame = frame

    def get_latest_frame(self) -> np.ndarray | None:
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)
        self.cap.release()


def open_camera_source(source_arg: str | int | None, calib: CameraCalibration | None) -> cv2.VideoCapture:
    node = source_arg
    if node is None:
        if calib and calib.device_node:
            node = calib.device_node
        else:
            node = 0

    m = re.search(r"video(\d+)", str(node))
    if m:
        idx = int(m.group(1))
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    else:
        try:
            idx = int(node)
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        except ValueError:
            cap = cv2.VideoCapture(str(node))

    if not cap.isOpened():
        log.warning(f"Could not open source {node}; attempting fallback index 0")
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if calib and calib.image_size:
        w, h = calib.image_size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_label = calib.name if calib else str(node)
    log.info(f"Camera '{cam_label}' stream opened at resolution: {actual_w}x{actual_h}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def parse_marker_sizes(entries: Sequence[str] | None, default_size: float = 0.0885) -> dict[int, float]:
    sizes = dict(DEFAULT_MARKER_SIZES_M)
    if not entries:
        return sizes
    for entry in entries:
        if "=" in entry:
            k, v = entry.split("=", 1)
            try:
                sizes[int(k.strip())] = float(v.strip())
            except ValueError:
                log.warning(f"Invalid marker size entry: {entry}")
        else:
            try:
                default_size = float(entry.strip())
            except ValueError:
                pass
    return sizes


def detect_all_tags(gray: np.ndarray, K: np.ndarray, detector: Detector, marker_sizes: dict[int, float], default_size: float) -> list[dict]:
    intrinsics = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    dets = detector.detect(gray, estimate_tag_pose=True, camera_params=intrinsics, tag_size=default_size)
    tag_results = []
    for d in dets:
        size = marker_sizes.get(d.tag_id, default_size)
        if abs(size - default_size) > 1e-4:
            d_spec = detector.detect(gray, estimate_tag_pose=True, camera_params=intrinsics, tag_size=size)
            for ds in d_spec:
                if ds.tag_id == d.tag_id:
                    d = ds
                    break
        R = np.asarray(d.pose_R, dtype=np.float64)
        t = np.asarray(d.pose_t, dtype=np.float64).reshape(3)
        pose_err = float(d.pose_err) if hasattr(d, "pose_err") and d.pose_err is not None else 0.0
        # Compute grazing tilt angle from tag normal
        tag_normal_cam = R @ np.array([0.0, 0.0, 1.0])
        view_ray = t / max(1e-6, np.linalg.norm(t))
        cos_tilt = abs(float(np.dot(tag_normal_cam, view_ray)))
        tilt_deg = math.degrees(math.acos(np.clip(cos_tilt, 0.0, 1.0)))

        tag_results.append({
            "id": d.tag_id,
            "R": R,
            "t": t,
            "corners": np.asarray(d.corners, dtype=np.float64),
            "size": size,
            "center_px": [float(d.center[0]), float(d.center[1])],
            "pose_err": pose_err,
            "decision_margin": float(getattr(d, "decision_margin", 0.0)),
            "tilt_deg": tilt_deg,
        })
    return tag_results


def select_best_common_tag(tags0: list[dict], tags1: list[dict], override_id: int | None = None) -> tuple[dict, dict] | tuple[None, None]:
    """Finds AprilTags visible in both cameras and ranks them to pick the BEST reference tag."""
    t0_map = {t["id"]: t for t in tags0}
    t1_map = {t["id"]: t for t in tags1}
    common_ids = set(t0_map.keys()) & set(t1_map.keys())

    if not common_ids:
        return None, None

    if override_id is not None and override_id in common_ids:
        return t0_map[override_id], t1_map[override_id]

    candidates = []
    for tid in common_ids:
        t0, t1 = t0_map[tid], t1_map[tid]
        # Quality score: lower pose error is better, lower tilt angle is better, higher decision margin is better
        # Score penalty = (pose_err0 + pose_err1)*100 + (tilt0 + tilt1) - (margin0 + margin1)*0.5
        score = (t0["pose_err"] + t1["pose_err"]) * 100.0 + (t0["tilt_deg"] + t1["tilt_deg"]) - (t0["decision_margin"] + t1["decision_margin"]) * 0.2
        candidates.append((score, t0, t1))

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[0][2]


def triangulate_points(pt0_px: tuple[float, float], pt1_px: tuple[float, float], P0: np.ndarray, P1: np.ndarray,
                       K0: np.ndarray, K1: np.ndarray,
                       R0_w: np.ndarray, t0_w: np.ndarray,
                       R1_w: np.ndarray, t1_w: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
    pts0 = np.array(pt0_px, dtype=np.float64).reshape(2, 1)
    pts1 = np.array(pt1_px, dtype=np.float64).reshape(2, 1)
    homo = cv2.triangulatePoints(P0, P1, pts0, pts1).reshape(4)
    if abs(homo[3]) < 1e-9:
        return np.zeros(3), float("inf"), float("inf"), 0.0, float("inf")
    pt3d = homo[:3] / homo[3]
    pt3d_h = np.append(pt3d, 1.0)
    p0_p = P0 @ pt3d_h
    p1_p = P1 @ pt3d_h
    e0 = float(np.linalg.norm(p0_p[:2] / p0_p[2] - np.array(pt0_px)))
    e1 = float(np.linalg.norm(p1_p[:2] / p1_p[2] - np.array(pt1_px)))
    reproj_err_px = (e0 + e1) / 2.0

    c0 = -R0_w.T @ t0_w
    c1 = -R1_w.T @ t1_w

    v0_cam = np.linalg.inv(K0) @ np.array([pt0_px[0], pt0_px[1], 1.0])
    v1_cam = np.linalg.inv(K1) @ np.array([pt1_px[0], pt1_px[1], 1.0])
    d0 = R0_w.T @ v0_cam
    d0 = d0 / max(1e-9, np.linalg.norm(d0))
    d1 = R1_w.T @ v1_cam
    d1 = d1 / max(1e-9, np.linalg.norm(d1))

    dot_val = np.clip(float(np.dot(d0, d1)), -1.0, 1.0)
    ray_angle_deg = math.degrees(math.acos(dot_val))

    cross_d = np.cross(d0, d1)
    cross_norm = np.linalg.norm(cross_d)
    if cross_norm < 1e-6:
        ray_diff_cm = float(np.linalg.norm(np.cross(c1 - c0, d0))) * 100.0
    else:
        ray_diff_cm = float(abs(np.dot(c1 - c0, cross_d)) / cross_norm) * 100.0

    dist_to_obj = float(np.linalg.norm(pt3d - (c0 + c1) / 2.0))
    sin_theta = max(1e-3, math.sin(math.radians(ray_angle_deg)))
    f_avg = (float(K0[0, 0]) + float(K1[0, 0])) / 2.0
    geom_err_m = (dist_to_obj / sin_theta) * (max(1.0, reproj_err_px) / f_avg)
    est_depth_err_cm = (geom_err_m * 100.0) + (ray_diff_cm * 0.5)

    return pt3d, reproj_err_px, ray_diff_cm, ray_angle_deg, est_depth_err_cm


def rotation_matrix_to_euler_deg(R: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    if sy >= 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def project_point_3d(pt_w: np.ndarray, R_c_w: np.ndarray, t_c_w: np.ndarray, K: np.ndarray) -> tuple[int, int] | None:
    """Project a 3D world/tag coordinate point into 2D camera pixels."""
    p_cam = R_c_w @ pt_w + t_c_w
    if p_cam[2] <= 0.05:
        return None
    u = (K[0, 0] * p_cam[0] / p_cam[2]) + K[0, 2]
    v = (K[1, 1] * p_cam[1] / p_cam[2]) + K[1, 2]
    return int(round(u)), int(round(v))


def draw_axis_measurement_lines(img: np.ndarray, pt_rel: np.ndarray, corner_offset: np.ndarray,
                                R_c_w: np.ndarray, t_c_w: np.ndarray, K: np.ndarray):
    """Draw orthogonal X, Y, and Z measurement lines connecting the tag origin to the clicked point."""
    # World points along axes:
    # Origin O = [0, 0, 0] (relative to tag corner/center)
    p_O = corner_offset
    p_X = corner_offset + np.array([pt_rel[0], 0.0, 0.0])
    p_XY = corner_offset + np.array([pt_rel[0], pt_rel[1], 0.0])
    p_XYZ = corner_offset + pt_rel

    px_O = project_point_3d(p_O, R_c_w, t_c_w, K)
    px_X = project_point_3d(p_X, R_c_w, t_c_w, K)
    px_XY = project_point_3d(p_XY, R_c_w, t_c_w, K)
    px_XYZ = project_point_3d(p_XYZ, R_c_w, t_c_w, K)

    # 1. Direct Euclidean Line (Hypotenuse): Origin -> Point
    if px_O and px_XYZ:
        dist_cm = float(np.linalg.norm(pt_rel)) * 100.0
        cv2.line(img, px_O, px_XYZ, (0, 255, 255), 2, cv2.LINE_AA)
        mid_x = (px_O[0] + px_XYZ[0]) // 2
        mid_y = (px_O[1] + px_XYZ[1]) // 2
        cv2.putText(img, f"d={dist_cm:.1f}cm", (mid_x + 8, mid_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2, cv2.LINE_AA)

    # 2. X-axis line (Red): Origin -> (X, 0, 0)
    if px_O and px_X:
        cv2.line(img, px_O, px_X, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(img, px_X, 4, (0, 0, 255), -1)
        mx, my = (px_O[0] + px_X[0]) // 2, (px_O[1] + px_X[1]) // 2
        cv2.putText(img, f"X:{pt_rel[0]*100:+.1f}cm", (mx + 6, my - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)

    # 3. Y-axis line (Green): (X, 0, 0) -> (X, Y, 0)
    if px_X and px_XY:
        cv2.line(img, px_X, px_XY, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(img, px_XY, 4, (0, 255, 0), -1)
        mx, my = (px_X[0] + px_XY[0]) // 2, (px_X[1] + px_XY[1]) // 2
        cv2.putText(img, f"Y:{pt_rel[1]*100:+.1f}cm", (mx + 6, my - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)

    # 4. Z-axis line (Blue/Cyan): (X, Y, 0) -> (X, Y, Z)
    if px_XY and px_XYZ:
        cv2.line(img, px_XY, px_XYZ, (255, 200, 0), 2, cv2.LINE_AA)
        cv2.circle(img, px_XYZ, 4, (255, 200, 0), -1)
        mx, my = (px_XY[0] + px_XYZ[0]) // 2, (px_XY[1] + px_XYZ[1]) // 2
        cv2.putText(img, f"Z:{pt_rel[2]*100:+.1f}cm", (mx + 6, my - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 0), 1, cv2.LINE_AA)


class ClickTracker:
    def __init__(self, panel_w: int, panel_h: int, orig_w0: int, orig_h0: int, orig_w1: int, orig_h1: int):
        self.panel_w = panel_w
        self.panel_h = panel_h
        self.orig_w0 = orig_w0
        self.orig_h0 = orig_h0
        self.orig_w1 = orig_w1
        self.orig_h1 = orig_h1

        self.pt0: tuple[float, float] | None = None
        self.pt1: tuple[float, float] | None = None
        self.history: list[dict] = []

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check Top-Left: Camera 0 view
            if 0 <= x < self.panel_w and 0 <= y < self.panel_h:
                orig_x = (x / float(self.panel_w)) * self.orig_w0
                orig_y = (y / float(self.panel_h)) * self.orig_h0
                self.pt0 = (orig_x, orig_y)
                log.info(f"Clicked Camera 1 point: ({orig_x:.1f}, {orig_y:.1f})")

            # Check Bottom-Left: Camera 1 view
            elif 0 <= x < self.panel_w and self.panel_h <= y < self.panel_h * 2:
                local_y = y - self.panel_h
                orig_x = (x / float(self.panel_w)) * self.orig_w1
                orig_y = (local_y / float(self.panel_h)) * self.orig_h1
                self.pt1 = (orig_x, orig_y)
                log.info(f"Clicked Camera 2 point: ({orig_x:.1f}, {orig_y:.1f})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clear_current()

    def clear_current(self):
        self.pt0 = None
        self.pt1 = None
        log.info("Cleared clicked points.")


def render_info_panel(width: int, height: int, best_tag_info: dict | None, click_result: dict | None, fps: float) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (24, 24, 28)

    # Header
    cv2.rectangle(panel, (0, 0), (width, 42), (40, 40, 48), -1)
    cv2.putText(panel, f"MANUAL CLICK 3D TRIANGULATOR ({fps:.1f} FPS)", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)

    y = 68
    # Section: Best AprilTag Origin
    cv2.putText(panel, "SELECTED REFERENCE APRILTAG", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 200, 50), 2)
    y += 10
    cv2.line(panel, (16, y), (width - 16, y), (60, 60, 70), 1)
    y += 24

    if best_tag_info:
        tid = best_tag_info["id"]
        t0 = best_tag_info["t0"]
        t1 = best_tag_info["t1"]
        d0_cm = float(np.linalg.norm(t0["t"])) * 100.0
        d1_cm = float(np.linalg.norm(t1["t"])) * 100.0
        fit0 = t0["pose_err"]
        tilt0 = t0["tilt_deg"]
        txt1 = f"Tag #{tid:<2} (Auto Best) | Size: {t0['size']*100:.1f}cm | Origin: {'Corner' if best_tag_info['corner'] else 'Center'}"
        txt2 = f"Cam1 Dist: {d0_cm:5.1f}cm (fit:{fit0:.2f}, tilt:{tilt0:.0f}°) | Cam2 Dist: {d1_cm:5.1f}cm"
        cv2.putText(panel, txt1, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 240, 255), 1, cv2.LINE_AA)
        y += 22
        cv2.putText(panel, txt2, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 220, 240), 1, cv2.LINE_AA)
        y += 26
    else:
        cv2.putText(panel, "NO SHARED APRILTAG: Looking for AprilTag visible in both cameras...", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
        y += 32

    # Section: Instructions
    cv2.putText(panel, "INTERACTIVE CLICK INSTRUCTIONS", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 255, 120), 2)
    y += 10
    cv2.line(panel, (16, y), (width - 16, y), (60, 60, 70), 1)
    y += 22
    cv2.putText(panel, "1. Left-click a feature on Camera 1 (Top-Left view).", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
    y += 20
    cv2.putText(panel, "2. Left-click the matching feature on Camera 2 (Bottom-Left).", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
    y += 20
    cv2.putText(panel, "3. Right-click or press 'C' to clear points.", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
    y += 26

    # Section: 3D Triangulation Result
    cv2.putText(panel, "CLICKED POINT 3D MEASUREMENT", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 2)
    y += 10
    cv2.line(panel, (16, y), (width - 16, y), (60, 60, 70), 1)
    y += 24

    if click_result and "xyz_cm" in click_result:
        x, y_pos, z = click_result["xyz_cm"]
        dist = click_result["dist_cm"]
        skew = click_result["ray_diff_cm"]
        ang = click_result["ray_angle_deg"]
        err = click_result["est_err_cm"]
        reproj = click_result["reproj_px"]

        box_color = (0, 255, 0) if err < 2.5 else ((0, 200, 255) if err < 5.0 else (0, 80, 255))

        res_line1 = f"3D Position: X={x:+6.1f} cm,  Y={y_pos:+6.1f} cm,  Z={z:+6.1f} cm"
        res_line2 = f"Euclidean Distance to Tag: {dist:5.1f} cm (Uncertainty: ±{err:4.1f} cm)"
        res_line3 = f"Ray Skew Distance: {skew:4.2f} cm | Convergence Angle: {ang:4.1f}° | Reproj: {reproj:.1f}px"

        cv2.putText(panel, res_line1, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 2)
        y += 24
        cv2.putText(panel, res_line2, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, box_color, 2)
        y += 24
        cv2.putText(panel, res_line3, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 220, 240), 1)
    else:
        st0 = "Set (Ready)" if (click_result and click_result.get("pt0_set")) else "Waiting for Click"
        st1 = "Set (Ready)" if (click_result and click_result.get("pt1_set")) else "Waiting for Click"
        cv2.putText(panel, f"Cam 1 Point: {st0}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 220, 255), 1)
        y += 22
        cv2.putText(panel, f"Cam 2 Point: {st1}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 220, 255), 1)

    return panel


def render_history_panel(width: int, height: int, history: list[dict], camera_poses: list[dict]) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (20, 20, 24)

    # Header
    cv2.rectangle(panel, (0, 0), (width, 42), (36, 36, 44), -1)
    cv2.putText(panel, "CAMERAS & MEASUREMENT LOG", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 200, 255), 2, cv2.LINE_AA)

    y = 66
    # Camera stereo baseline status
    if camera_poses:
        for cp in camera_poses:
            x, yp, z = cp["xyz_cm"]
            r, p, yw = cp["rpy_deg"]
            col = (255, 120, 255)
            txt = f"{cp['name']:<5} | Pos: [{x:+5.1f}, {yp:+5.1f}, {z:+5.1f}]cm | R:{r:+.0f}° P:{p:+.0f}° Y:{yw:+.0f}° | {cp.get('base','')}"
            cv2.putText(panel, txt, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
            y += 20
    else:
        cv2.putText(panel, "Stereo camera poses: Calibrating relative to Best Tag...", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 130), 1)
        y += 20

    y += 10
    cv2.line(panel, (16, y), (width - 16, y), (60, 60, 70), 1)
    y += 20

    # History Table Header
    cv2.putText(panel, f"{'#':<3} | {'EUCLID':<8} | {'X(cm)':<6} {'Y(cm)':<6} {'Z(cm)':<6} | {'SKEW':<7} {'ANGLE':<7} {'ERR'}", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 170), 1)
    y += 18

    if not history:
        cv2.putText(panel, "Click points on both cameras to record measurements (Press Space to Save).", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 130), 1)
    else:
        for idx, h in enumerate(history[-8:]):
            x, yp, z = h["xyz_cm"]
            dist = h["dist_cm"]
            skew = h["ray_diff_cm"]
            ang = h["ray_angle_deg"]
            err = h["est_err_cm"]
            col = (100, 255, 100) if err < 2.5 else ((50, 220, 255) if err < 5.0 else (50, 80, 255))
            txt = f"{idx+1:<3} | {dist:5.1f}cm | {x:+6.1f} {yp:+6.1f} {z:+6.1f} | {skew:4.1f}cm  {ang:4.0f}°  ±{err:4.1f}cm"
            cv2.putText(panel, txt, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)
            y += 20

    return panel


def main() -> int:
    parser = argparse.ArgumentParser(
        description="2-Camera Interactive Manual Point Triangulation with Automatic Best-Tag Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--cam0", default=None, help="Camera 0 device node or index (default: from calib0 or 0)")
    parser.add_argument("--cam1", default=None, help="Camera 1 device node or index (default: from calib1 or 1)")
    parser.add_argument("--calib0", type=Path, default=None, help="Camera 0 calibration NPZ")
    parser.add_argument("--calib1", type=Path, default=None, help="Camera 1 calibration NPZ")
    parser.add_argument("--tag-id", type=int, default=None, help="Manual reference Tag ID override (default: auto best tag)")
    parser.add_argument("--tag-size", type=float, default=0.0885, help="Physical edge length of default AprilTag in meters")
    parser.add_argument("--marker-size", action="append", help="Override marker size for specific ID, format: ID=METRES (e.g. 0=0.124)")
    parser.add_argument("--corner-origin", action="store_true", default=True, help="Report relative positions from physical marker corner rather than center")
    parser.add_argument("--fps", type=float, default=10.0, help="Target FPS limiter rate")
    parser.add_argument("--tag-family", default="tag36h11", help="AprilTag dictionary family")
    parser.add_argument("--undistort", action="store_true", default=True, help="Undistort camera frames using calibration dist_coeffs")
    args = parser.parse_args()

    calib0 = CameraCalibration.from_npz(args.calib0) if args.calib0 and args.calib0.exists() else CameraCalibration.fallback_fov("cam0")
    calib1 = CameraCalibration.from_npz(args.calib1) if args.calib1 and args.calib1.exists() else CameraCalibration.fallback_fov("cam1")

    marker_sizes = parse_marker_sizes(args.marker_size, default_size=args.tag_size)
    detector = Detector(families=args.tag_family, nthreads=2, quad_decimate=1.0)

    log.info(f"Target FPS Limiter: {args.fps:.1f} FPS (buffer dropping enabled)")

    cap0_raw = open_camera_source(args.cam0, calib0)
    cap1_raw = open_camera_source(args.cam1, calib1)
    reader0 = NonQueuingCameraReader(cap0_raw)
    reader1 = NonQueuingCameraReader(cap1_raw)

    panel_w, panel_h = 640, 360
    tracker = ClickTracker(panel_w, panel_h, calib0.image_size[0], calib0.image_size[1], calib1.image_size[0], calib1.image_size[1])

    window_name = "2-Camera Interactive Manual Point Triangulator (Click Camera 1 & Camera 2)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, tracker.on_mouse)

    frame_interval = 1.0 / max(0.1, args.fps)
    last_process_time = 0.0
    fps_smooth = args.fps

    log.info("Starting 2-Camera Interactive Point Triangulation loop... Press 'q' or ESC to exit.")

    try:
        while True:
            now = time.time()
            elapsed = now - last_process_time
            if elapsed < frame_interval:
                time.sleep(max(0.001, frame_interval - elapsed))
                continue

            last_process_time = time.time()

            frame0 = reader0.get_latest_frame()
            frame1 = reader1.get_latest_frame()

            if frame0 is None or frame1 is None:
                time.sleep(0.01)
                continue

            if args.undistort and np.any(calib0.dist_coeffs):
                frame0 = cv2.undistort(frame0, calib0.camera_matrix, calib0.dist_coeffs)
            if args.undistort and np.any(calib1.dist_coeffs):
                frame1 = cv2.undistort(frame1, calib1.camera_matrix, calib1.dist_coeffs)

            if frame0.ndim == 2:
                gray0 = frame0
                frame0 = cv2.cvtColor(frame0, cv2.COLOR_GRAY2BGR)
            elif frame0.shape[2] == 4:
                gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGRA2GRAY)
                frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGRA2BGR)
            else:
                gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

            if frame1.ndim == 2:
                gray1 = frame1
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
            elif frame1.shape[2] == 4:
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGRA2GRAY)
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGRA2BGR)
            else:
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

            tags0 = detect_all_tags(gray0, calib0.camera_matrix, detector, marker_sizes, args.tag_size)
            tags1 = detect_all_tags(gray1, calib1.camera_matrix, detector, marker_sizes, args.tag_size)

            best_tag0, best_tag1 = select_best_common_tag(tags0, tags1, override_id=args.tag_id)

            annotated0 = frame0.copy()
            annotated1 = frame1.copy()

            # Draw AprilTags on Camera 1
            for t in tags0:
                corners = t["corners"].astype(int)
                is_best = (best_tag0 is not None and t["id"] == best_tag0["id"])
                col = (0, 255, 255) if is_best else (255, 180, 0)
                cv2.polylines(annotated0, [corners], True, col, 2)
                cv2.circle(annotated0, (int(corners[0, 0]), int(corners[0, 1])), 5, (0, 0, 255), -1)
                cv2.putText(annotated0, f"Tag #{t['id']}{' [BEST]' if is_best else ''}",
                            (corners[0, 0], max(16, corners[0, 1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

            # Draw AprilTags on Camera 2
            for t in tags1:
                corners = t["corners"].astype(int)
                is_best = (best_tag1 is not None and t["id"] == best_tag1["id"])
                col = (0, 255, 255) if is_best else (255, 180, 0)
                cv2.polylines(annotated1, [corners], True, col, 2)
                cv2.circle(annotated1, (int(corners[0, 0]), int(corners[0, 1])), 5, (0, 0, 255), -1)
                cv2.putText(annotated1, f"Tag #{t['id']}{' [BEST]' if is_best else ''}",
                            (corners[0, 0], max(16, corners[0, 1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

            camera_poses = []
            click_result = None
            best_tag_info = None

            if best_tag0 and best_tag1:
                ref_size = best_tag0["size"]
                corner_offset = np.array([-ref_size / 2.0, -ref_size / 2.0, 0.0]) if args.corner_origin else np.zeros(3)
                best_tag_info = {"id": best_tag0["id"], "t0": best_tag0, "t1": best_tag1, "corner": args.corner_origin}

                R_c0_t, t_c0_t = best_tag0["R"], best_tag0["t"]
                R_c1_t, t_c1_t = best_tag1["R"], best_tag1["t"]

                P0 = calib0.camera_matrix @ np.hstack((R_c0_t, t_c0_t.reshape(3, 1)))
                P1 = calib1.camera_matrix @ np.hstack((R_c1_t, t_c1_t.reshape(3, 1)))

                p_c0 = -R_c0_t.T @ t_c0_t - corner_offset
                p_c1 = -R_c1_t.T @ t_c1_t - corner_offset
                baseline_m = float(np.linalg.norm(p_c1 - p_c0))

                camera_poses.append({
                    "name": "Cam0",
                    "xyz_cm": [p_c0[0] * 100.0, p_c0[1] * 100.0, p_c0[2] * 100.0],
                    "rpy_deg": rotation_matrix_to_euler_deg(R_c0_t.T),
                    "base": f"Base: {baseline_m*100:4.1f}cm",
                })
                camera_poses.append({
                    "name": "Cam1",
                    "xyz_cm": [p_c1[0] * 100.0, p_c1[1] * 100.0, p_c1[2] * 100.0],
                    "rpy_deg": rotation_matrix_to_euler_deg(R_c1_t.T),
                    "base": f"Base: {baseline_m*100:4.1f}cm",
                })

                # If both points are clicked, perform 3D triangulation
                if tracker.pt0 is not None and tracker.pt1 is not None:
                    pt_tag, reproj_px, ray_diff_cm, ray_angle_deg, est_depth_err_cm = triangulate_points(
                        tracker.pt0, tracker.pt1,
                        P0, P1, calib0.camera_matrix, calib1.camera_matrix,
                        R_c0_t, t_c0_t, R_c1_t, t_c1_t
                    )
                    pt_rel = pt_tag - corner_offset
                    dist_cm = float(np.linalg.norm(pt_rel)) * 100.0

                    click_result = {
                        "xyz_cm": [pt_rel[0] * 100.0, pt_rel[1] * 100.0, pt_rel[2] * 100.0],
                        "dist_cm": dist_cm,
                        "ray_diff_cm": ray_diff_cm,
                        "ray_angle_deg": ray_angle_deg,
                        "est_err_cm": est_depth_err_cm,
                        "reproj_px": reproj_px,
                        "pt0_set": True,
                        "pt1_set": True,
                    }
                else:
                    click_result = {
                        "pt0_set": tracker.pt0 is not None,
                        "pt1_set": tracker.pt1 is not None,
                    }

            # Draw clicked crosshairs on Camera 1
            if tracker.pt0 is not None:
                cx0, cy0 = int(tracker.pt0[0]), int(tracker.pt0[1])
                cv2.drawMarker(annotated0, (cx0, cy0), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.circle(annotated0, (cx0, cy0), 6, (0, 255, 255), 2)
                cv2.putText(annotated0, f"P1 ({cx0},{cy0})", (cx0 + 10, cy0 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Draw clicked crosshairs on Camera 2
            if tracker.pt1 is not None:
                cx1, cy1 = int(tracker.pt1[0]), int(tracker.pt1[1])
                cv2.drawMarker(annotated1, (cx1, cy1), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.circle(annotated1, (cx1, cy1), 6, (0, 255, 255), 2)
                cv2.putText(annotated1, f"P2 ({cx1},{cy1})", (cx1 + 10, cy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Draw 3D measurement overlay if triangulated
            if click_result and "xyz_cm" in click_result and best_tag0 and best_tag1:
                x, y_pos, z = click_result["xyz_cm"]
                dist = click_result["dist_cm"]
                err = click_result["est_err_cm"]
                skew = click_result["ray_diff_cm"]
                ang = click_result["ray_angle_deg"]
                pt_rel_m = np.array([x / 100.0, y_pos / 100.0, z / 100.0])

                # Draw 3D X, Y, Z orthogonal distance lines and Euclidean line from Tag Origin
                draw_axis_measurement_lines(annotated0, pt_rel_m, corner_offset,
                                            best_tag0["R"], best_tag0["t"], calib0.camera_matrix)
                draw_axis_measurement_lines(annotated1, pt_rel_m, corner_offset,
                                            best_tag1["R"], best_tag1["t"], calib1.camera_matrix)

                box_col = (0, 255, 0) if err < 2.5 else ((0, 200, 255) if err < 5.0 else (0, 80, 255))
                lbl_top = f"3D Point: [{x:+5.1f}, {y_pos:+5.1f}, {z:+5.1f}] cm (d={dist:.1f}cm ±{err:.1f}cm)"
                lbl_bot = f"skew={skew:.1f}cm ang={ang:.0f}°"

                if tracker.pt0 is not None:
                    cx0, cy0 = int(tracker.pt0[0]), int(tracker.pt0[1])
                    cv2.putText(annotated0, lbl_top, (max(16, cx0 - 120), max(24, cy0 - 24)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_col, 2)
                    cv2.putText(annotated0, lbl_bot, (max(16, cx0 - 120), max(40, cy0 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                if tracker.pt1 is not None:
                    cx1, cy1 = int(tracker.pt1[0]), int(tracker.pt1[1])
                    cv2.putText(annotated1, lbl_top, (max(16, cx1 - 120), max(24, cy1 - 24)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_col, 2)
                    cv2.putText(annotated1, lbl_bot, (max(16, cx1 - 120), max(40, cy1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # Titles on video feeds
            instant_fps = 1.0 / max(1e-4, time.time() - now)
            fps_smooth = 0.9 * fps_smooth + 0.1 * min(args.fps, instant_fps)

            cv2.putText(annotated0, f"Camera 1 ({calib0.name}) - Click P1", (16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated1, f"Camera 2 ({calib1.name}) - Click P2", (16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 2x2 Dashboard assembly
            cam1_view = cv2.resize(annotated0, (panel_w, panel_h))
            cam2_view = cv2.resize(annotated1, (panel_w, panel_h))

            info_panel = render_info_panel(panel_w, panel_h, best_tag_info, click_result, fps_smooth)
            history_panel = render_history_panel(panel_w, panel_h, tracker.history, camera_poses)

            top_row = np.hstack((cam1_view, info_panel))
            bot_row = np.hstack((cam2_view, history_panel))
            dashboard = np.vstack((top_row, bot_row))

            cv2.imshow(window_name, dashboard)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c'):
                tracker.clear_current()
            elif key == 32:  # Space bar to save current measurement to log
                if click_result and "xyz_cm" in click_result:
                    tracker.history.append(click_result)
                    log.info(f"Saved measurement #{len(tracker.history)}: {click_result['xyz_cm']}")

    finally:
        reader0.stop()
        reader1.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
