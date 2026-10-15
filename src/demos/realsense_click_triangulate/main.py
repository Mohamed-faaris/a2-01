#!/usr/bin/env python3
"""Intel RealSense D415 Interactive Manual Click Point Triangulator & Depth Measurer with Best-Tag Origin.

Features:
  - Connects to Intel RealSense D415 (Color BGR8 + Depth Z16 aligned to color).
  - 10 FPS zero-latency frame dropping limiter.
  - Automatically selects the BEST reference AprilTag visible (lowest fit error, lowest grazing tilt).
    (Or allows manual tag override via --tag-id).
  - Interactive Mouse Clicking:
      - Left-click any pixel on the Color video view to sample its depth and 3D position.
      - Samples robust median depth from RealSense Z16 depth frame.
      - Backprojects (u, v, depth) to 3D camera frame.
      - Transforms 3D coordinates relative to the Best AprilTag Origin (Corner or Center).
      - Draws orthogonal 3D X, Y, and Z measurement lines directly on the live color stream.
      - Draws Euclidean distance hypotenuse line (Origin -> Clicked Point) with live label.
      - Right-click or press 'c' to clear current clicked point.
      - Press Spacebar to record current measurement into the history table.
  - 2x2 Unified Dashboard Layout:
      - Top-Left:     Color Stream with Clicked Crosshair & 3D Axis Measurement Lines
      - Top-Right:    RealSense Depth Colorized Stream
      - Bottom-Left:  Real-time Measurement Info Panel (X, Y, Z, Euclid dist, depth health)
      - Bottom-Right: Camera Pose & Measurement History Log Table
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
from typing import Sequence

import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("realsense-click-triangulator")

DEFAULT_CALIB_DIR = "calibrations/calibarotor0/outputs/rl720"

DEFAULT_TAG_FAMILY = "tag36h11"

# Default AprilTag physical marker sizes in meters
# Mappings:
#   Tag 0: 134 mm (0.1340 m)
#   Tag 1: 124 mm (0.1240 m)
#   Tags 2-5: 88.5 mm (0.0885 m)
#   Tags 6-13: 57.5 mm (0.0575 m)
#   Tags 20-31: 41 mm (0.0410 m)
#   Tags 40-45: 66 mm (0.0660 m)
DEFAULT_MARKER_SIZES_M: dict[int, float] = {
    # 134 mm
    0: 0.1340,
    # 124 mm
    1: 0.1240,
    # 88.5 mm
    2: 0.0885,
    3: 0.0885,
    4: 0.0885,
    5: 0.0885,
    # 57.5 mm
    6: 0.0575,
    7: 0.0575,
    8: 0.0575,
    9: 0.0575,
    10: 0.0575,
    11: 0.0575,
    12: 0.0575,
    13: 0.0575,
    # 41 mm (IDs 20 - 31)
    **{i: 0.0410 for i in range(20, 32)},
    # 66 mm (IDs 40 - 45)
    **{i: 0.0660 for i in range(40, 46)},
}

# Marker metadata including family and physical size description
DEFAULT_MARKER_METADATA: dict[int, dict[str, str | float]] = {
    tag_id: {
        "family": DEFAULT_TAG_FAMILY,
        "size_m": size_m,
        "size_mm": round(size_m * 1000.0, 1),
    }
    for tag_id, size_m in DEFAULT_MARKER_SIZES_M.items()
}


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
    def load(cls, path_str: str, expected_stream_type: str = "RGB / Color") -> CameraCalibration:
        p = Path(path_str)
        if not p.is_absolute():
            script_dir = Path(__file__).resolve().parent
            candidates = [
                Path.cwd() / p,
                script_dir / p,
                Path("/home/mfk01/.local/share/econ-systems/calibrations/calibarotor0/output/rl720"),
            ]
            for cand in candidates:
                if cand.exists():
                    p = cand
                    break

        if p.is_dir():
            npz_cand = p / "camera_calibration.npz"
            json_cand = p / "camera_calibration.json"
            if npz_cand.exists():
                p = npz_cand
            elif json_cand.exists():
                p = json_cand

        if p.suffix == ".npz":
            data = np.load(p, allow_pickle=True)
            K = np.asarray(data["camera_matrix"], dtype=np.float64)
            D = np.asarray(data["dist_coeffs"], dtype=np.float64)
            size = tuple(int(v) for v in data["image_size"])
            name = str(data.get("camera_name", p.stem))
            node = str(data.get("device_node")) if "device_node" in data else None
            card = str(data.get("card_name")) if "card_name" in data else None
            stype = str(data.get("stream_type")) if "stream_type" in data else None
            calib = cls(name=name, camera_matrix=K, dist_coeffs=D, image_size=size,
                        device_node=node, card_name=card, stream_type=stype)
            cls._check_stream_type(calib, expected_stream_type)
            return calib

        if p.suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            mat_data = data["camera_matrix"]
            if "matrix_3x3" in mat_data:
                K = np.array(mat_data["matrix_3x3"], dtype=np.float64)
            else:
                fx = float(mat_data["fx"])
                fy = float(mat_data["fy"])
                cx = float(mat_data["cx"])
                cy = float(mat_data["cy"])
                K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
            D = np.array(data["distortion_coefficients"], dtype=np.float64)
            info = data.get("calibration_info", {})
            w = int(info.get("image_width", 1280))
            h = int(info.get("image_height", 720))
            dev = data.get("device_info", {})
            cfg = data.get("config", {})
            stype = dev.get("stream_type") or cfg.get("stream_type")
            calib = cls(
                name=data.get("camera_name", "rl720"),
                camera_matrix=K,
                dist_coeffs=D,
                image_size=(w, h),
                device_node=dev.get("device_node") or cfg.get("device_node"),
                card_name=dev.get("card_name") or cfg.get("camera_card"),
                stream_type=stype,
            )
            cls._check_stream_type(calib, expected_stream_type)
            return calib

        raise ValueError(f"Unsupported calibration format: {p}")

    @staticmethod
    def _check_stream_type(calib: CameraCalibration, expected_stream_type: str | None = None) -> None:
        res_str = f" @ {calib.image_size[0]}x{calib.image_size[1]}" if calib.image_size else ""
        if calib.stream_type:
            log.info(f"Calibration stream type: '{calib.stream_type}'{res_str} (Camera: {calib.name}, Card: {calib.card_name or 'Unknown'})")
            if expected_stream_type and expected_stream_type.lower() not in calib.stream_type.lower():
                log.warning(f"Stream type mismatch! Expected '{expected_stream_type}', got '{calib.stream_type}'.")
            else:
                log.info(f"Stream type verified matching expected '{expected_stream_type}'.")
        else:
            log.warning(f"Calibration does not specify 'stream_type'{res_str}.")


class NonQueuingRealSenseStream:
    """Background thread continuously polling RealSense frames,
    dropping queued frames to ensure real-time latest frame delivery."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.align = rs.align(rs.stream.color)
        self.lock = threading.Lock()
        self.latest_color = None
        self.latest_depth = None
        self.latest_depth_frame = None
        self.running = False
        self.thread = None

    def start(self):
        log.info(f"Starting Intel RealSense stream at {self.width}x{self.height} @ {self.fps} FPS...")
        try:
            self.profile = self.pipeline.start(self.config)
        except RuntimeError as e:
            if "busy" in str(e).lower() or "errno=16" in str(e).lower():
                log.error("RealSense device node is currently locked by another process!")
                log.error("Please close any active camera viewer or process using the RealSense camera.")
            raise

        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def get_color_intrinsics(self) -> tuple[np.ndarray, np.ndarray]:
        color_prof = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_prof.get_intrinsics()
        K = np.array([
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        D = np.array(intr.coeffs, dtype=np.float64)
        return K, D

    def _worker(self):
        consecutive_timeouts = 0
        while self.running:
            try:
                frameset = self.pipeline.wait_for_frames(timeout_ms=1000)
                aligned = self.align.process(frameset)
                df = aligned.get_depth_frame()
                cf = aligned.get_color_frame()
                if df and cf:
                    c_data = np.asanyarray(cf.get_data())
                    d_data = np.asanyarray(df.get_data())
                    with self.lock:
                        self.latest_color = c_data
                        self.latest_depth = d_data
                        self.latest_depth_frame = df
                    consecutive_timeouts = 0
            except Exception:
                consecutive_timeouts += 1
                if consecutive_timeouts == 5:
                    log.warning("RealSense frame timeout (USB buffer delay). Still waiting for frames...")

    def get_latest_frames(self):
        with self.lock:
            if self.latest_color is None or self.latest_depth_frame is None:
                return None, None, None
            return self.latest_color.copy(), self.latest_depth.copy(), self.latest_depth_frame

    def wait_until_ready(self, timeout_sec: float = 6.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            with self.lock:
                if self.latest_color is not None and self.latest_depth_frame is not None:
                    return True
            time.sleep(0.05)
        return False

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        try:
            self.pipeline.stop()
        except Exception:
            pass


def get_robust_median_depth(depth_frame: rs.depth_frame, cx: float, cy: float, window: int = 6) -> float:
    w, h = depth_frame.get_width(), depth_frame.get_height()
    x_min, x_max = max(0, int(cx - window)), min(w, int(cx + window + 1))
    y_min, y_max = max(0, int(cy - window)), min(h, int(cy + window + 1))
    depths = []
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            d = depth_frame.get_distance(x, y)
            if 0.05 < d < 5.0:
                depths.append(d)
    return float(np.median(depths)) if depths else 0.0


def backproject_to_3d(u: float, v: float, depth_m: float, K: np.ndarray) -> np.ndarray:
    x = (u - K[0, 2]) * depth_m / K[0, 0]
    y = (v - K[1, 2]) * depth_m / K[1, 1]
    return np.array([x, y, depth_m], dtype=np.float64)


def project_point_3d(pt_w: np.ndarray, R_c_w: np.ndarray, t_c_w: np.ndarray, K: np.ndarray) -> tuple[int, int] | None:
    p_cam = R_c_w @ pt_w + t_c_w
    if p_cam[2] <= 0.05:
        return None
    u = (K[0, 0] * p_cam[0] / p_cam[2]) + K[0, 2]
    v = (K[1, 1] * p_cam[1] / p_cam[2]) + K[1, 2]
    return int(round(u)), int(round(v))


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


def select_best_tag(tags: list[dict], override_id: int | None = None) -> dict | None:
    if not tags:
        return None
    if override_id is not None:
        for t in tags:
            if t["id"] == override_id:
                return t

    # Score: lower pose error and lower tilt is best, higher decision margin is best
    candidates = []
    for t in tags:
        score = (t["pose_err"] * 100.0) + t["tilt_deg"] - (t["decision_margin"] * 0.2)
        candidates.append((score, t))
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def draw_axis_measurement_lines(img: np.ndarray, pt_rel: np.ndarray, corner_offset: np.ndarray,
                                R_c_w: np.ndarray, t_c_w: np.ndarray, K: np.ndarray):
    """Draw orthogonal X, Y, and Z measurement lines connecting the tag origin to the clicked point."""
    p_O = corner_offset
    p_X = corner_offset + np.array([pt_rel[0], 0.0, 0.0])
    p_XY = corner_offset + np.array([pt_rel[0], pt_rel[1], 0.0])
    p_XYZ = corner_offset + pt_rel

    px_O = project_point_3d(p_O, R_c_w, t_c_w, K)
    px_X = project_point_3d(p_X, R_c_w, t_c_w, K)
    px_XY = project_point_3d(p_XY, R_c_w, t_c_w, K)
    px_XYZ = project_point_3d(p_XYZ, R_c_w, t_c_w, K)

    # 1. Direct Euclidean Line (Yellow): Origin -> Point
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

    # 4. Z-axis line (Cyan): (X, Y, 0) -> (X, Y, Z)
    if px_XY and px_XYZ:
        cv2.line(img, px_XY, px_XYZ, (255, 200, 0), 2, cv2.LINE_AA)
        cv2.circle(img, px_XYZ, 4, (255, 200, 0), -1)
        mx, my = (px_XY[0] + px_XYZ[0]) // 2, (px_XY[1] + px_XYZ[1]) // 2
        cv2.putText(img, f"Z:{pt_rel[2]*100:+.1f}cm", (mx + 6, my - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 0), 1, cv2.LINE_AA)


class RealSenseClickTracker:
    def __init__(self, panel_w: int, panel_h: int, orig_w: int, orig_h: int):
        self.panel_w = panel_w
        self.panel_h = panel_h
        self.orig_w = orig_w
        self.orig_h = orig_h

        self.clicked_pt: tuple[float, float] | None = None
        self.history: list[dict] = []

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check Top-Left: Color view
            if 0 <= x < self.panel_w and 0 <= y < self.panel_h:
                orig_x = (x / float(self.panel_w)) * self.orig_w
                orig_y = (y / float(self.panel_h)) * self.orig_h
                self.clicked_pt = (orig_x, orig_y)
                log.info(f"Clicked RealSense Color point: ({orig_x:.1f}, {orig_y:.1f})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clear_current()

    def clear_current(self):
        self.clicked_pt = None
        log.info("Cleared clicked point.")


def render_info_panel(width: int, height: int, best_tag: dict | None, click_res: dict | None, fps: float) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (24, 24, 28)

    # Header
    cv2.rectangle(panel, (0, 0), (width, 42), (40, 40, 48), -1)
    cv2.putText(panel, f"REALSENSE 3D POINT MEASURER ({fps:.1f} FPS)", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)

    y = 68
    # Section: Best AprilTag Origin
    cv2.putText(panel, "REFERENCE APRILTAG ORIGIN", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 200, 50), 2)
    y += 10
    cv2.line(panel, (16, y), (width - 16, y), (60, 60, 70), 1)
    y += 24

    if best_tag:
        tid = best_tag["id"]
        d_cm = float(np.linalg.norm(best_tag["t"])) * 100.0
        fit = best_tag["pose_err"]
        tilt = best_tag["tilt_deg"]
        txt1 = f"Tag #{tid:<2} (Auto Best) | Size: {best_tag['size']*100:.1f}cm | Cam Dist: {d_cm:5.1f}cm"
        txt2 = f"PnP Fit Error: {fit:.2f} | Grazing Tilt Angle: {tilt:.0f}°"
        cv2.putText(panel, txt1, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 240, 255), 1, cv2.LINE_AA)
        y += 22
        cv2.putText(panel, txt2, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 220, 240), 1, cv2.LINE_AA)
        y += 26
    else:
        cv2.putText(panel, "NO REFERENCE TAG: Searching for visible AprilTags in view...", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
        y += 32

    # Section: Instructions
    cv2.putText(panel, "INTERACTIVE CONTROLS", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 255, 120), 2)
    y += 10
    cv2.line(panel, (16, y), (width - 16, y), (60, 60, 70), 1)
    y += 22
    cv2.putText(panel, "1. Left-click anywhere on Color feed (Top-Left) to measure 3D.", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
    y += 20
    cv2.putText(panel, "2. Right-click or press 'C' to clear points.", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
    y += 20
    cv2.putText(panel, "3. Press Spacebar to record measurement into history table.", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
    y += 26

    # Section: Measurement Output
    cv2.putText(panel, "CLICKED POINT 3D MEASUREMENT", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 2)
    y += 10
    cv2.line(panel, (16, y), (width - 16, y), (60, 60, 70), 1)
    y += 24

    if click_res and "xyz_cm" in click_res:
        x, y_pos, z = click_res["xyz_cm"]
        dist = click_res["dist_cm"]
        raw_depth_cm = click_res["raw_depth_m"] * 100.0

        res_line1 = f"3D Position: X={x:+6.1f} cm,  Y={y_pos:+6.1f} cm,  Z={z:+6.1f} cm"
        res_line2 = f"Euclidean Distance to Tag: {dist:5.1f} cm"
        res_line3 = f"RealSense Sensor Depth: {raw_depth_cm:5.1f} cm (Window Median)"

        cv2.putText(panel, res_line1, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 2)
        y += 24
        cv2.putText(panel, res_line2, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 100), 2)
        y += 24
        cv2.putText(panel, res_line3, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 220, 240), 1)
    else:
        st = "Waiting for Click" if not click_res or not click_res.get("clicked") else "Sampling Depth..."
        cv2.putText(panel, f"Status: {st}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 170), 1)

    return panel


def render_history_panel(width: int, height: int, history: list[dict], camera_pose: dict | None) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (20, 20, 24)

    # Header
    cv2.rectangle(panel, (0, 0), (width, 42), (36, 36, 44), -1)
    cv2.putText(panel, "CAMERA POSE & MEASUREMENT LOG", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 200, 255), 2, cv2.LINE_AA)

    y = 66
    if camera_pose:
        x, yp, z = camera_pose["xyz_cm"]
        r, p, yw = camera_pose["rpy_deg"]
        txt = f"Camera Pos: [{x:+5.1f}, {yp:+5.1f}, {z:+5.1f}]cm | R:{r:+.0f}° P:{p:+.0f}° Y:{yw:+.0f}°"
        cv2.putText(panel, txt, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 120, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "Camera pose: Searching for Reference Tag...", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 130), 1)
    y += 22

    cv2.line(panel, (16, y), (width - 16, y), (60, 60, 70), 1)
    y += 20

    # History Table Header
    cv2.putText(panel, f"{'#':<3} | {'EUCLID':<8} | {'X(cm)':<6} {'Y(cm)':<6} {'Z(cm)':<6} | {'SENSOR DEPTH':<14}", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 170), 1)
    y += 18

    if not history:
        cv2.putText(panel, "Click points on Color view to measure (Press Spacebar to Save).", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 130), 1)
    else:
        for idx, h in enumerate(history[-8:]):
            x, yp, z = h["xyz_cm"]
            dist = h["dist_cm"]
            raw_depth_cm = h["raw_depth_m"] * 100.0
            txt = f"{idx+1:<3} | {dist:5.1f}cm | {x:+6.1f} {yp:+6.1f} {z:+6.1f} | {raw_depth_cm:5.1f}cm"
            cv2.putText(panel, txt, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 255, 100), 1, cv2.LINE_AA)
            y += 20

    return panel


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Intel RealSense Interactive Point Measurer & Triangulator with Best-Tag Origin",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--calib", type=str, default=DEFAULT_CALIB_DIR,
                        help="Path to calibration directory or npz/json")
    parser.add_argument("--factory-calib", action="store_true", default=False,
                        help="Use RealSense factory color camera intrinsics from device EEPROM")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Target FPS limiter rate")
    parser.add_argument("--tag-id", type=int, default=None,
                        help="Manual reference Tag ID override (default: auto best tag)")
    parser.add_argument("--tag-size", type=float, default=0.0885,
                        help="Physical edge length of default AprilTag in meters")
    parser.add_argument("--marker-size", action="append",
                        help="Override marker size for specific ID, format: ID=METRES (e.g. 0=0.124)")
    parser.add_argument("--corner-origin", action="store_true", default=True,
                        help="Report relative positions from physical marker corner rather than center")
    parser.add_argument("--tag-family", default=DEFAULT_TAG_FAMILY,
                        help=f"AprilTag dictionary family (default: {DEFAULT_TAG_FAMILY})")
    parser.add_argument("--bottle-radius", type=float, default=0.0,
                        help="Radius in meters to compensate along optical ray (default: 0.0)")
    args = parser.parse_args()

    calib = CameraCalibration.load(args.calib)
    K = calib.camera_matrix
    D = calib.dist_coeffs

    marker_sizes = parse_marker_sizes(args.marker_size, default_size=args.tag_size)
    detector = Detector(families=args.tag_family, nthreads=2, quad_decimate=1.0)
    colorizer = rs.colorizer()

    streamer = NonQueuingRealSenseStream(width=1280, height=720, fps=30)
    streamer.start()

    if not streamer.wait_until_ready(timeout_sec=5.0):
        log.warning("No frames received from RealSense within 5s.")
    else:
        log.info("RealSense video and depth streams active and verified.")

    if args.factory_calib:
        K, D = streamer.get_color_intrinsics()
        log.info(f"Using RealSense Factory Color Intrinsics: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
    else:
        log.info(f"Loaded Custom Calibration: {calib.name} (fx={K[0,0]:.2f}, fy={K[1,1]:.2f})")

    panel_w, panel_h = 640, 360
    tracker = RealSenseClickTracker(panel_w, panel_h, 1280, 720)

    window_name = "RealSense Interactive Point Measurer (Click Color feed to measure)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, tracker.on_mouse)

    frame_interval = 1.0 / max(0.1, args.fps)
    last_process_time = 0.0
    fps_smooth = args.fps

    log.info("Starting RealSense Interactive Point Triangulation loop... Press 'q' or ESC to exit.")

    try:
        while True:
            now = time.time()
            elapsed = now - last_process_time
            if elapsed < frame_interval:
                time.sleep(max(0.001, frame_interval - elapsed))
                continue

            last_process_time = time.time()

            color_img, depth_img, depth_frame = streamer.get_latest_frames()
            if color_img is None or depth_frame is None:
                time.sleep(0.01)
                continue

            annotated_color = color_img.copy()
            gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

            # Detect all AprilTags
            tags = detect_all_tags(gray, K, detector, marker_sizes, args.tag_size)
            best_tag = select_best_tag(tags, override_id=args.tag_id)

            # Draw AprilTags
            for t in tags:
                corners = t["corners"].astype(int)
                is_best = (best_tag is not None and t["id"] == best_tag["id"])
                col = (0, 255, 255) if is_best else (255, 180, 0)
                cv2.polylines(annotated_color, [corners], True, col, 2)
                cv2.circle(annotated_color, (int(corners[0, 0]), int(corners[0, 1])), 5, (0, 0, 255), -1)
                cv2.putText(annotated_color, f"Tag #{t['id']}{' [BEST]' if is_best else ''}",
                            (corners[0, 0], max(16, corners[0, 1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

            camera_pose = None
            click_res = None
            corner_offset = np.zeros(3)

            if best_tag:
                ref_size = best_tag["size"]
                corner_offset = np.array([-ref_size / 2.0, -ref_size / 2.0, 0.0]) if args.corner_origin else np.zeros(3)
                R_tag = best_tag["R"]
                t_tag = best_tag["t"]

                # Camera position in tag frame: C = -R^T * t
                p_cam_tag = -R_tag.T @ t_tag - corner_offset
                rpy_cam = rotation_matrix_to_euler_deg(R_tag.T)
                camera_pose = {
                    "xyz_cm": [p_cam_tag[0] * 100.0, p_cam_tag[1] * 100.0, p_cam_tag[2] * 100.0],
                    "rpy_deg": rpy_cam,
                }

                # Sample clicked point if clicked
                if tracker.clicked_pt is not None:
                    u_click, v_click = tracker.clicked_pt
                    measured_depth = get_robust_median_depth(depth_frame, u_click, v_click, window=6)

                    if measured_depth > 0:
                        p_cam_surface = backproject_to_3d(u_click, v_click, measured_depth, K)
                        ray_norm = np.linalg.norm(p_cam_surface)
                        if ray_norm > 1e-6 and args.bottle_radius > 0:
                            p_cam = p_cam_surface + (p_cam_surface / ray_norm) * args.bottle_radius
                        else:
                            p_cam = p_cam_surface

                        # Express in reference tag coordinates
                        p_in_tag = R_tag.T @ (p_cam - t_tag)
                        pt_rel = p_in_tag - corner_offset
                        dist_cm = float(np.linalg.norm(pt_rel)) * 100.0

                        click_res = {
                            "xyz_cm": [pt_rel[0] * 100.0, pt_rel[1] * 100.0, pt_rel[2] * 100.0],
                            "dist_cm": dist_cm,
                            "raw_depth_m": measured_depth,
                            "clicked": True,
                        }

                        # Draw orthogonal 3D X, Y, Z lines and distance line from Tag origin
                        draw_axis_measurement_lines(annotated_color, pt_rel, corner_offset,
                                                    R_tag, t_tag, K)
                    else:
                        click_res = {"clicked": True, "raw_depth_m": 0.0}

            # Draw crosshair at clicked position
            if tracker.clicked_pt is not None:
                cx, cy = int(tracker.clicked_pt[0]), int(tracker.clicked_pt[1])
                cv2.drawMarker(annotated_color, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.circle(annotated_color, (cx, cy), 6, (0, 255, 255), 2)
                cv2.putText(annotated_color, f"Click ({cx},{cy})", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                if click_res and "xyz_cm" in click_res:
                    x, y_pos, z = click_res["xyz_cm"]
                    dist = click_res["dist_cm"]
                    lbl_top = f"3D Point: [{x:+5.1f}, {y_pos:+5.1f}, {z:+5.1f}] cm (d={dist:.1f}cm)"
                    cv2.putText(annotated_color, lbl_top, (max(16, cx - 120), max(24, cy - 24)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)

            # Titles on Color feed
            instant_fps = 1.0 / max(1e-4, time.time() - now)
            fps_smooth = 0.9 * fps_smooth + 0.1 * min(args.fps, instant_fps)

            cv2.putText(annotated_color, f"RealSense Color (Click to Measure) | {fps_smooth:.1f} FPS", (16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Colorized depth stream
            depth_color_frame = colorizer.colorize(depth_frame)
            depth_color_data = np.asanyarray(depth_color_frame.get_data())

            # 2x2 Dashboard assembly
            cam1_view = cv2.resize(annotated_color, (panel_w, panel_h))
            depth_view = cv2.resize(depth_color_data, (panel_w, panel_h))

            info_panel = render_info_panel(panel_w, panel_h, best_tag, click_res, fps_smooth)
            history_panel = render_history_panel(panel_w, panel_h, tracker.history, camera_pose)

            top_row = np.hstack((cam1_view, depth_view))
            bot_row = np.hstack((info_panel, history_panel))
            dashboard = np.vstack((top_row, bot_row))

            cv2.imshow(window_name, dashboard)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c'):
                tracker.clear_current()
            elif key == 32:  # Spacebar to save measurement
                if click_res and "xyz_cm" in click_res:
                    tracker.history.append(click_res)
                    log.info(f"Saved measurement #{len(tracker.history)}: {click_res['xyz_cm']}")

    finally:
        streamer.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
