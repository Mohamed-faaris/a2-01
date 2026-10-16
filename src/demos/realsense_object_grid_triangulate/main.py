#!/usr/bin/env python3
"""
RealSense Object Surface Grid Triangulator & Measurer Demo
==========================================================
Detects cans, bottles, cups, cartons/boxes, and common tabletop objects using YOLO26s,
constructs a dense/configurable 2D-to-3D sampling grid across each object's surface,
evaluates 3D metric coordinates and geometry relative to AprilTag reference frames,
and supports interactive point clicking and measurement inspection.

Features:
  - YOLO26s object detection for cans, bottles, cups, boxes, cartons, cartons/books/vases.
  - Generates uniform or interior-inset NxM surface grids of 3D points on detected objects.
  - Projects each grid point into real-world tabletop metric coordinates (X, Y, Z in cm)
    relative to the calibrated AprilTag origin (using calibrations/markers.yaml).
  - Calculates object 3D dimensions (width, height, depth spread), centroid, and surface normal.
  - RealSense D415/D435 live streaming + interactive mouse clicking for point inspection.
  - Fallback simulated mock playback if camera is disconnected or in test environments.
  - 4-Quadrant 1080p / 720p Dashboard:
      * Top-Left: Color RGB with Object BBoxes, Class Labels, and 3D Surface Grids.
      * Top-Right: Object Surface Grid Telemetry HUD (Selected object metrics & grid table).
      * Bottom-Left: RealSense Metric Depth Colormap (Turbo).
      * Bottom-Right: Top-down Bird's-Eye View (BEV) Tabletop Map showing objects & tag origin.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import logging
import math
from pathlib import Path
import sys
import threading
import time
from typing import Sequence

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False

from pupil_apriltags import Detector
from ultralytics import YOLO

# Centralized AprilTag marker registry
try:
    from common.markers import (
        DEFAULT_MARKER_METADATA,
        DEFAULT_MARKER_SIZES_M,
        DEFAULT_TAG_FAMILY,
        DEFAULT_TAG_SIZE_M,
        get_marker_size,
        get_marker_sizes,
        parse_marker_size_overrides,
    )
except ImportError:
    repo_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(repo_root))
    from common.markers import (
        DEFAULT_MARKER_METADATA,
        DEFAULT_MARKER_SIZES_M,
        DEFAULT_TAG_FAMILY,
        DEFAULT_TAG_SIZE_M,
        get_marker_size,
        get_marker_sizes,
        parse_marker_size_overrides,
    )

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("realsense-object-grid")

DEFAULT_YOLO_MODEL = "/home/mfk01/Projects/econ-systems/demos/demo2/yolo26s.pt"
DEFAULT_CALIB_DIR = "calibrations/calibarotor0/outputs/rl720"

# Target object classes: cans, bottles, cups, cartons, boxes, etc.
TARGET_OBJECT_CLASSES = {
    "bottle", "cup", "can", "vase", "bowl", "box", "carton",
    "book", "cell phone", "toothbrush", "snowboard", "skateboard"
}

# Friendly display naming for common tabletop objects
DISPLAY_NAME_MAP = {
    "snowboard": "can/cylinder",
    "skateboard": "can/cylinder",
    "cell phone": "carton/box",
    "book": "carton/box",
    "toothbrush": "cylinder",
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
            return cls(name=name, camera_matrix=K, dist_coeffs=D, image_size=size,
                       device_node=node, card_name=card, stream_type=stype)

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
            return cls(
                name=data.get("camera_name", "rl720"),
                camera_matrix=K,
                dist_coeffs=D,
                image_size=(w, h),
                device_node=dev.get("device_node") or cfg.get("device_node"),
                card_name=dev.get("card_name") or cfg.get("camera_card"),
                stream_type=stype,
            )

        # Fallback default intrinsics for RealSense D415 720p
        log.warning(f"Could not load custom calibration from {path_str}. Using RealSense D415 default intrinsics.")
        K_def = np.array([[925.09, 0.0, 640.0], [0.0, 925.09, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        D_def = np.zeros(5, dtype=np.float64)
        return cls(name="realsense_d415_default", camera_matrix=K_def, dist_coeffs=D_def, image_size=(1280, 720))


class NonQueuingRealSenseStream:
    """Threaded non-queuing stream reader for RealSense RGB + Depth frames."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.align = rs.align(rs.stream.color)
        self.lock = threading.Lock()
        self.latest_color = None
        self.latest_depth_data = None
        self.latest_depth_frame = None
        self.depth_scale = 0.001
        self.running = False
        self.thread = None
        self.profile = None

    def start(self):
        self.profile = self.pipeline.start(self.config)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
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
                        self.latest_depth_data = d_data.astype(np.float32) * self.depth_scale
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
            return self.latest_color.copy(), self.latest_depth_data.copy(), self.latest_depth_frame

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


def get_robust_median_depth(depth_map: np.ndarray, cx: float, cy: float, window: int = 6) -> float:
    """Extract median depth around a pixel coordinate (cx, cy) within a square window, with adaptive fallback for thin/reflective surfaces."""
    h, w = depth_map.shape[:2]
    for win in [window, window * 2, window * 3]:
        x_min, x_max = max(0, int(cx - win)), min(w, int(cx + win + 1))
        y_min, y_max = max(0, int(cy - win)), min(h, int(cy + win + 1))
        sub = depth_map[y_min:y_max, x_min:x_max]
        valid = sub[(sub > 0.05) & (sub < 5.0)]
        if len(valid) > 0:
            return float(np.median(valid))
    return 0.0


def backproject_pixel(u: float, v: float, depth_m: float, K: np.ndarray) -> np.ndarray:
    """Backproject a 2D image pixel (u, v) and metric depth Z to 3D camera coordinate (X, Y, Z)."""
    x = (u - K[0, 2]) * depth_m / K[0, 0]
    y = (v - K[1, 2]) * depth_m / K[1, 1]
    return np.array([x, y, depth_m], dtype=np.float64)


def project_3d_to_pixel(pt_3d: np.ndarray, K: np.ndarray) -> tuple[int, int]:
    """Project a 3D camera point (X, Y, Z) into 2D image pixel coordinates (u, v)."""
    if pt_3d[2] <= 1e-4:
        return 0, 0
    u = (K[0, 0] * pt_3d[0] / pt_3d[2]) + K[0, 2]
    v = (K[1, 1] * pt_3d[1] / pt_3d[2]) + K[1, 2]
    return int(round(u)), int(round(v))


def get_corner_offset_in_tag(corner_idx: int, tag_size: float) -> tuple[np.ndarray, str]:
    """
    Get 3D offset (dx, dy, dz) from AprilTag center to the chosen origin.
    corner_idx:
      -1: Center origin (0, 0, 0)
       0: Top-Left Corner (-s/2, -s/2, 0)
       1: Top-Right Corner (+s/2, -s/2, 0)
       2: Bottom-Right Corner (+s/2, +s/2, 0)
       3: Bottom-Left Corner (-s/2, +s/2, 0)
    """
    half = tag_size / 2.0
    if corner_idx == 0:
        return np.array([-half, -half, 0.0], dtype=np.float64), "Top-Left Corner"
    elif corner_idx == 1:
        return np.array([+half, -half, 0.0], dtype=np.float64), "Top-Right Corner"
    elif corner_idx == 2:
        return np.array([+half, +half, 0.0], dtype=np.float64), "Bottom-Right Corner"
    elif corner_idx == 3:
        return np.array([-half, +half, 0.0], dtype=np.float64), "Bottom-Left Corner"
    else:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64), "Tag Center"


def draw_tag_with_corner_origin(
    img: np.ndarray,
    tag: dict,
    K: np.ndarray,
    active_corner_idx: int = 0
):
    """
    Draw AprilTag polygon, highlight all 4 corners, and clearly mark the chosen Origin corner
    with prominent 3D coordinate axes and clear text labels.
    """
    corners = tag["corners"].astype(int)
    R_tag = tag["R"]
    t_tag = tag["t"]
    tag_size = tag["size"]
    tid = tag["id"]

    # 1. Draw outer AprilTag boundary
    cv2.polylines(img, [corners], True, (0, 255, 0), 2, cv2.LINE_AA)

    # 2. Draw subtle dots on all 4 corners with numbers
    for i, pt in enumerate(corners):
        px, py = int(pt[0]), int(pt[1])
        cv2.circle(img, (px, py), 4, (180, 220, 180), -1, cv2.LINE_AA)
        cv2.putText(img, f"c{i}", (px - 14, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 220, 180), 1)

    # 3. Determine active origin point in 3D camera space
    corner_offset, corner_name = get_corner_offset_in_tag(active_corner_idx, tag_size)
    # p_cam_origin = t_tag + R_tag @ corner_offset
    p_cam_origin = t_tag + R_tag @ corner_offset

    origin_px = project_3d_to_pixel(p_cam_origin, K)

    # 4. Draw prominent Origin Marker: concentric rings + crosshair
    cv2.circle(img, origin_px, 9, (0, 255, 255), 2, cv2.LINE_AA)       # Yellow outer ring
    cv2.circle(img, origin_px, 4, (0, 0, 255), -1, cv2.LINE_AA)       # Red inner dot
    cv2.drawMarker(img, origin_px, (0, 255, 255), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)

    # 5. Draw 3D Coordinate Axes anchored at this exact Corner Origin
    axis_len = 0.08  # 8 cm axes
    ax_x = project_3d_to_pixel(p_cam_origin + R_tag[:, 0] * axis_len, K)
    ax_y = project_3d_to_pixel(p_cam_origin + R_tag[:, 1] * axis_len, K)
    ax_z = project_3d_to_pixel(p_cam_origin + R_tag[:, 2] * axis_len, K)

    # X-axis: Red, Y-axis: Green, Z-axis: Blue
    cv2.line(img, origin_px, ax_x, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(img, "+X", (ax_x[0] + 4, ax_x[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.line(img, origin_px, ax_y, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(img, "+Y", (ax_y[0] + 4, ax_y[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    cv2.line(img, origin_px, ax_z, (255, 100, 0), 3, cv2.LINE_AA)
    cv2.putText(img, "+Z", (ax_z[0] + 4, ax_z[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 2)

    # 6. Clear label with corner origin title
    lbl_text = f"ORIGIN (0,0,0) [Tag #{tid} {corner_name}]"
    (tw, th), _ = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
    lx = max(10, origin_px[0] - tw // 2)
    ly = max(24, origin_px[1] - 16)
    cv2.rectangle(img, (lx - 4, ly - th - 4), (lx + tw + 4, ly + 4), (20, 20, 20), -1)
    cv2.rectangle(img, (lx - 4, ly - th - 4), (lx + tw + 4, ly + 4), (0, 255, 255), 1)
    cv2.putText(img, lbl_text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2, cv2.LINE_AA)


def draw_axis_measurement_from_corner(
    img: np.ndarray,
    p_cam_origin: np.ndarray,
    p_cam_click: np.ndarray,
    pt_table_rel_m: np.ndarray,
    R_tag: np.ndarray,
    K: np.ndarray
):
    """
    Draw direct 3D Euclidean distance line and orthogonal 3D X, Y, Z lines connecting
    the exact physical corner origin to the clicked surface point.
    """
    # 3D points in Camera Frame
    # Origin: p_cam_origin
    # X step: p_cam_origin + R_tag @ [dx, 0, 0]
    # XY step: p_cam_origin + R_tag @ [dx, dy, 0]
    # XYZ (clicked target): p_cam_click
    p_X = p_cam_origin + R_tag @ np.array([pt_table_rel_m[0], 0.0, 0.0])
    p_XY = p_cam_origin + R_tag @ np.array([pt_table_rel_m[0], pt_table_rel_m[1], 0.0])
    p_XYZ = p_cam_click

    px_O = project_3d_to_pixel(p_cam_origin, K)
    px_X = project_3d_to_pixel(p_X, K)
    px_XY = project_3d_to_pixel(p_XY, K)
    px_XYZ = project_3d_to_pixel(p_XYZ, K)

    dist_cm = float(np.linalg.norm(pt_table_rel_m)) * 100.0

    # 1. Direct Euclidean Line (Yellow): Corner Origin -> Clicked Point
    cv2.line(img, px_O, px_XYZ, (0, 255, 255), 2, cv2.LINE_AA)
    mid_x = (px_O[0] + px_XYZ[0]) // 2
    mid_y = (px_O[1] + px_XYZ[1]) // 2
    cv2.putText(img, f"d={dist_cm:.1f}cm", (mid_x + 8, mid_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2, cv2.LINE_AA)

    # 2. X-axis projection line (Red): Origin -> (X, 0, 0)
    cv2.line(img, px_O, px_X, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(img, px_X, 4, (0, 0, 255), -1, cv2.LINE_AA)
    mx_x, my_x = (px_O[0] + px_X[0]) // 2, (px_O[1] + px_X[1]) // 2
    cv2.putText(img, f"X:{pt_table_rel_m[0]*100:+.1f}cm", (mx_x + 6, my_x - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1, cv2.LINE_AA)

    # 3. Y-axis projection line (Green): (X, 0, 0) -> (X, Y, 0)
    cv2.line(img, px_X, px_XY, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(img, px_XY, 4, (0, 255, 0), -1, cv2.LINE_AA)
    mx_y, my_y = (px_X[0] + px_XY[0]) // 2, (px_X[1] + px_XY[1]) // 2
    cv2.putText(img, f"Y:{pt_table_rel_m[1]*100:+.1f}cm", (mx_y + 6, my_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)

    # 4. Z-axis elevation line (Blue / Cyan): (X, Y, 0) -> (X, Y, Z)
    cv2.line(img, px_XY, px_XYZ, (255, 200, 0), 2, cv2.LINE_AA)
    cv2.circle(img, px_XYZ, 4, (255, 200, 0), -1, cv2.LINE_AA)
    mx_z, my_z = (px_XY[0] + px_XYZ[0]) // 2, (px_XY[1] + px_XYZ[1]) // 2
    cv2.putText(img, f"Z:{pt_table_rel_m[2]*100:+.1f}cm", (mx_z + 6, my_z - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 200, 0), 1, cv2.LINE_AA)


def generate_object_surface_grid(
    bbox_xyxy: list[int | float],
    depth_map: np.ndarray,
    K: np.ndarray,
    R_tag: np.ndarray | None,
    t_tag: np.ndarray | None,
    corner_offset: np.ndarray,
    grid_rows: int = 5,
    grid_cols: int = 5,
    margin_ratio: float = 0.15
) -> dict:
    """Generate a uniform NxM 2D-to-3D sampling grid across the object's visible bounding box."""
    x1, y1, x2, y2 = bbox_xyxy
    w = max(10, x2 - x1)
    h = max(10, y2 - y1)

    # Inset grid slightly from edges to prevent sampling background pixels
    x_start = x1 + w * margin_ratio
    x_end = x2 - w * margin_ratio
    y_start = y1 + h * margin_ratio
    y_end = y2 - h * margin_ratio

    xs = np.linspace(x_start, x_end, grid_cols)
    ys = np.linspace(y_start, y_end, grid_rows)

    grid_points = []
    cam_pts = []
    table_pts = []

    for r_idx, py in enumerate(ys):
        row_pts = []
        for c_idx, px in enumerate(xs):
            d_m = get_robust_median_depth(depth_map, px, py, window=4)
            pt_info = {
                "row": r_idx,
                "col": c_idx,
                "px": (int(round(px)), int(round(py))),
                "depth_m": d_m,
                "valid": False,
                "cam_xyz_cm": None,
                "table_xyz_cm": None,
            }
            if d_m > 0.05:
                p_cam = backproject_pixel(px, py, d_m, K)
                cam_pts.append(p_cam)
                pt_info["valid"] = True
                pt_info["cam_xyz_cm"] = (p_cam * 100.0).tolist()

                if R_tag is not None and t_tag is not None:
                    # Express in tag reference frame: p_tag = R^T * (p_cam - t) - corner_offset
                    p_tag = R_tag.T @ (p_cam - t_tag) - corner_offset
                    table_pts.append(p_tag)
                    pt_info["table_xyz_cm"] = (p_tag * 100.0).tolist()

            row_pts.append(pt_info)
        grid_points.append(row_pts)

    # Compute object surface metrics if valid points exist
    centroid_cam = np.mean(cam_pts, axis=0) if cam_pts else None
    centroid_table = np.mean(table_pts, axis=0) if table_pts else None

    # Compute spatial spread (dimensions)
    if len(cam_pts) >= 4:
        c_arr = np.array(cam_pts)
        dim_cam = (c_arr.max(axis=0) - c_arr.min(axis=0)) * 100.0
    else:
        dim_cam = np.zeros(3)

    return {
        "grid": grid_points,
        "valid_count": len(cam_pts),
        "total_count": grid_rows * grid_cols,
        "centroid_cam_cm": (centroid_cam * 100.0).tolist() if centroid_cam is not None else None,
        "centroid_table_cm": (centroid_table * 100.0).tolist() if centroid_table is not None else None,
        "spread_cam_cm": dim_cam.tolist(),
    }


def draw_object_grid(
    img_bgr: np.ndarray,
    grid_data: dict,
    color_scheme: tuple[int, int, int] = (0, 255, 255),
    draw_mesh: bool = True
):
    """Draw interactive surface grid mesh and nodes onto the annotated image."""
    grid = grid_data["grid"]
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Draw mesh grid lines connecting adjacent grid nodes
    if draw_mesh:
        for r in range(rows):
            for c in range(cols):
                curr = grid[r][c]
                if not curr["valid"]:
                    continue
                # Right neighbor
                if c + 1 < cols and grid[r][c + 1]["valid"]:
                    cv2.line(img_bgr, curr["px"], grid[r][c + 1]["px"], (color_scheme[0]//2, color_scheme[1]//2, color_scheme[2]//2), 1, cv2.LINE_AA)
                # Bottom neighbor
                if r + 1 < rows and grid[r + 1][c]["valid"]:
                    cv2.line(img_bgr, curr["px"], grid[r + 1][c]["px"], (color_scheme[0]//2, color_scheme[1]//2, color_scheme[2]//2), 1, cv2.LINE_AA)

    # Draw nodes
    for r in range(rows):
        for c in range(cols):
            pt = grid[r][c]
            if pt["valid"]:
                cv2.circle(img_bgr, pt["px"], 3, color_scheme, -1, cv2.LINE_AA)
            else:
                cv2.circle(img_bgr, pt["px"], 2, (80, 80, 80), -1, cv2.LINE_AA)


class ClickInspectionTracker:
    def __init__(self, display_w: int, display_h: int, orig_w: int = 1280, orig_h: int = 720):
        self.display_w = display_w
        self.display_h = display_h
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.clicked_pt: tuple[float, float] | None = None
        self.selected_object_idx: int | None = None
        self.history: list[dict] = []

    def handle_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x < self.display_w and y < self.display_h:
                orig_x = (x / float(self.display_w)) * self.orig_w
                orig_y = (y / float(self.display_h)) * self.orig_h
                self.clicked_pt = (orig_x, orig_y)

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clicked_pt = None
            self.selected_object_idx = None


def render_telemetry_hud(
    width: int,
    height: int,
    objects: list[dict],
    best_tag: dict | None,
    corner_name: str,
    selected_idx: int | None,
    clicked_pt_info: dict | None,
    fps: float,
    grid_size: int
) -> np.ndarray:
    """Render high-density telemetry table showing detected objects and 3D surface grid points."""
    hud = np.zeros((height, width, 3), dtype=np.uint8)
    hud[:] = (20, 20, 24)

    # Title header
    cv2.rectangle(hud, (0, 0), (width, 38), (35, 35, 45), -1)
    cv2.putText(hud, f"OBJECT SURFACE 3D GRID TELEMETRY ({fps:.1f} FPS)", (14, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)

    y = 60
    # Tag status banner
    if best_tag:
        tid = best_tag["id"]
        t_dist = float(np.linalg.norm(best_tag["t"])) * 100.0
        cv2.putText(hud, f"[ORIGIN (0,0,0)] Tag #{tid} [{corner_name.upper()}] | Dist: {t_dist:.1f} cm",
                    (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(hud, "[REF ORIGIN] No AprilTag detected - displaying camera optical coordinates",
                    (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 150, 255), 1, cv2.LINE_AA)

    y += 24
    cv2.line(hud, (14, y), (width - 14, y), (50, 50, 60), 1)
    y += 18

    # Objects table header
    hdr = f"{'OBJ':<4} {'CLASS':<12} {'CONF':<6} {'TABLE_XYZ (cm)':<22} {'GRID PTS':<10}"
    cv2.putText(hud, hdr, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 200), 1, cv2.LINE_AA)
    y += 16
    cv2.line(hud, (14, y), (width - 14, y), (40, 40, 50), 1)
    y += 18

    # List up to 4 objects
    for idx, obj in enumerate(objects[:4]):
        is_sel = (selected_idx == idx)
        prefix = f"#{idx+1:<2}"
        cls_name = obj["class"][:11]
        conf_str = f"{obj['conf']:.2f}"
        c_table = obj["grid_data"]["centroid_table_cm"]
        if c_table:
            pos_str = f"[{c_table[0]:+5.1f}, {c_table[1]:+5.1f}, {c_table[2]:+5.1f}]"
        else:
            c_cam = obj["grid_data"]["centroid_cam_cm"]
            pos_str = f"cam:[{c_cam[0]:+5.1f}, {c_cam[2]:+5.1f}]" if c_cam else "N/A"
        
        valid_cnt = f"{obj['grid_data']['valid_count']}/{obj['grid_data']['total_count']}"
        row_str = f"{prefix:<4} {cls_name:<12} {conf_str:<6} {pos_str:<22} {valid_cnt:<10}"

        color = (0, 255, 255) if is_sel else (220, 230, 240)
        cv2.putText(hud, row_str, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)
        y += 20

    y = max(y + 10, 200)
    cv2.line(hud, (14, y), (width - 14, y), (50, 50, 60), 1)
    y += 20

    # Inspection details of selected object or clicked point
    cv2.putText(hud, "POINT / OBJECT INSPECTION", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 200, 50), 1, cv2.LINE_AA)
    y += 20

    if clicked_pt_info and clicked_pt_info.get("clicked"):
        pt = clicked_pt_info
        cv2.putText(hud, f"Clicked Pixel: ({pt['u']}, {pt['v']}) | Raw Depth: {pt['depth_m']:.3f} m",
                    (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 240, 255), 1, cv2.LINE_AA)
        y += 20
        if pt.get("table_xyz_cm"):
            tx, ty, tz = pt["table_xyz_cm"]
            dist = pt["dist_cm"]
            cv2.putText(hud, f"Origin Relative: X={tx:+6.1f} cm, Y={ty:+6.1f} cm, Z={tz:+6.1f} cm (Dist={dist:.1f}cm)",
                        (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
            y += 20
    else:
        cv2.putText(hud, "Click anywhere on Color Feed to inspect specific 3D surface point.",
                    (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 150), 1, cv2.LINE_AA)
        y += 20

    # Controls instructions at bottom
    y = height - 40
    cv2.line(hud, (14, y), (width - 14, y), (50, 50, 60), 1)
    y += 18
    controls = f"[O]: Cycle Origin Corner | [G]: Grid {grid_size}x{grid_size} | [M]: Mesh | [C]: Clear | [Q]: Quit"
    cv2.putText(hud, controls, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 200, 160), 1, cv2.LINE_AA)

    return hud


def render_bev_tabletop_map(
    width: int,
    height: int,
    objects: list[dict],
    best_tag: dict | None,
    corner_name: str = "Top-Left Corner",
    clicked_pt_info: dict | None = None
) -> np.ndarray:
    """Render 2D top-down Bird's-Eye View (BEV) radar map of tabletop objects and clicked point."""
    bev = np.zeros((height, width, 3), dtype=np.uint8)
    bev[:] = (18, 18, 22)

    cx = width // 2
    cy = int(height * 0.85)
    scale = 350.0  # pixels per meter

    # Draw grid lines (every 20 cm)
    for dist_m in [0.2, 0.4, 0.6, 0.8, 1.0]:
        py = int(cy - dist_m * scale)
        if 0 <= py < height:
            cv2.line(bev, (20, py), (width - 20, py), (35, 35, 45), 1)
            cv2.putText(bev, f"{dist_m:.1f}m", (width - 60, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 70, 80), 1)

    for offset_m in [-0.4, -0.2, 0.0, 0.2, 0.4]:
        px = int(cx + offset_m * scale)
        if 0 <= px < width:
            cv2.line(bev, (px, 20), (px, height - 20), (35, 35, 45), 1)
            cv2.putText(bev, f"{offset_m:+.1f}m", (px - 15, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 70, 80), 1)

    # Origin (AprilTag Corner)
    cv2.circle(bev, (cx, cy), 7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(bev, (cx, cy), 3, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.drawMarker(bev, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
    cv2.putText(bev, f"ORIGIN (0,0) [{corner_name}]", (cx + 10, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)

    # Draw objects
    for idx, obj in enumerate(objects):
        c_table = obj["grid_data"]["centroid_table_cm"]
        if not c_table:
            continue
        # In tag coordinate system: X is right, Y is forward
        tx_m = c_table[0] / 100.0
        ty_m = c_table[1] / 100.0

        px = int(cx + tx_m * scale)
        py = int(cy - ty_m * scale)

        if 0 <= px < width and 0 <= py < height:
            col = obj["color"]
            cv2.circle(bev, (px, py), 8, col, -1, cv2.LINE_AA)
            cv2.circle(bev, (px, py), 10, (255, 255, 255), 1, cv2.LINE_AA)
            lbl = f"#{idx+1} {obj['class']}"
            cv2.putText(bev, lbl, (px + 12, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

    # Draw Clicked Point on Coordinate Map
    if clicked_pt_info and clicked_pt_info.get("clicked") and clicked_pt_info.get("table_xyz_cm"):
        c_xyz = clicked_pt_info["table_xyz_cm"]
        click_x_m = c_xyz[0] / 100.0
        click_y_m = c_xyz[1] / 100.0
        click_px = int(cx + click_x_m * scale)
        click_py = int(cy - click_y_m * scale)

        # Clamp marker coordinates so click is always visibly marked on the map canvas
        clamped_px = int(np.clip(click_px, 16, width - 16))
        clamped_py = int(np.clip(click_py, 16, height - 16))

        # 1. Line connecting Origin (cx, cy) to Clicked Position
        cv2.line(bev, (cx, cy), (clamped_px, clamped_py), (0, 255, 255), 2, cv2.LINE_AA)

        # 2. Prominent Click Target Marker (Red & Yellow Target with Pulsing Rings)
        cv2.circle(bev, (clamped_px, clamped_py), 9, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(bev, (clamped_px, clamped_py), 14, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(bev, (clamped_px, clamped_py), (255, 255, 255), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)

        # 3. Coordinate and Distance Label
        dist_cm = clicked_pt_info.get("dist_cm", math.hypot(c_xyz[0], c_xyz[1]))
        click_lbl = f"CLICK [{c_xyz[0]:+5.1f}, {c_xyz[1]:+5.1f}] cm (d={dist_cm:.1f}cm)"
        (tw, th), _ = cv2.getTextSize(click_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        lx = min(width - tw - 12, max(12, clamped_px + 16))
        ly = max(24, min(height - 10, clamped_py - 6))
        cv2.rectangle(bev, (lx - 3, ly - th - 3), (lx + tw + 3, ly + 3), (10, 10, 10), -1)
        cv2.rectangle(bev, (lx - 3, ly - th - 3), (lx + tw + 3, ly + 3), (0, 255, 255), 1)
        cv2.putText(bev, click_lbl, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

    # Title
    cv2.putText(bev, "TOP-DOWN TABLETOP BEV MAP", (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 240), 1)
    return bev


def run_demo(args: argparse.Namespace) -> int:
    log.info("Starting RealSense Object Surface Grid Triangulator & Measurer Demo...")

    # Load calibration & YOLO
    calib = CameraCalibration.load(args.calib)
    K = calib.camera_matrix
    D = calib.dist_coeffs

    log.info(f"Loading YOLO model: {args.yolo_model}")
    yolo = YOLO(args.yolo_model)

    marker_sizes = parse_marker_size_overrides(args.marker_size)
    tag_detector = Detector(families=args.tag_family, nthreads=2, quad_decimate=1.0)
    colorizer = rs.colorizer()

    # Hardware connection check or simulated mock fallback
    use_mock = False
    streamer = None

    if HAS_REALSENSE and not args.mock:
        ctx = rs.context()
        if len(ctx.query_devices()) > 0:
            log.info("RealSense device detected! Initializing live RGB-D stream...")
            streamer = NonQueuingRealSenseStream(width=1280, height=720, fps=30)
            streamer.start()
            if not streamer.wait_until_ready(timeout_sec=5.0):
                log.warning("No live frames from RealSense. Falling back to simulated mock scenes...")
                use_mock = True
            else:
                if args.factory_calib:
                    K, D = streamer.get_color_intrinsics()
                    log.info(f"Using RealSense Factory Intrinsics: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}")
        else:
            log.warning("No physical RealSense device found. Using simulated tabletop scenes...")
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

    panel_w, panel_h = 640, 360
    tracker = ClickInspectionTracker(panel_w, panel_h, 1280, 720)

    window_name = "RealSense Object Surface Grid Triangulator (YOLO26s + AprilTag)"
    if not getattr(args, "headless", False):
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, tracker.handle_mouse)

    grid_sizes = [3, 5, 7, 9]
    grid_size_idx = 1  # default 5x5
    draw_mesh = True
    frame_idx = 0
    fps_smooth = 15.0
    active_corner_idx = getattr(args, "corner_id", 0)  # 0: Top-Left, 1: Top-Right, 2: Bottom-Right, 3: Bottom-Left, -1: Center

    # Color palette for detected object types
    class_palette = {
        "can": (0, 200, 255),       # Orange-Yellow
        "bottle": (255, 180, 0),    # Cyan-Blue
        "cup": (0, 255, 120),       # Green
        "box": (255, 100, 200),     # Magenta
        "carton": (150, 100, 255),  # Violet
        "book": (200, 220, 50),     # Yellow-Green
    }

    try:
        while True:
            t0 = time.time()
            if not use_mock and streamer:
                color_img, depth_map, df = streamer.get_latest_frames()
                if color_img is None or depth_map is None:
                    time.sleep(0.01)
                    continue
            else:
                if not mock_frames:
                    log.error("No mock frames available.")
                    break
                curr_mock = mock_frames[frame_idx % len(mock_frames)]
                color_img = curr_mock[0].copy()
                depth_map = curr_mock[1].copy()
                time.sleep(0.03)

            annotated_color = color_img.copy()
            gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

            # 1. Detect AprilTags for Tabletop Reference
            dets = tag_detector.detect(
                gray, estimate_tag_pose=True,
                camera_params=(float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])),
                tag_size=args.tag_size
            )

            best_tag = None
            corner_name = "Top-Left Corner"
            corner_offset = np.zeros(3)

            if dets:
                d0 = dets[0]
                t_size = marker_sizes.get(d0.tag_id, args.tag_size)
                R_tag = np.asarray(d0.pose_R, dtype=np.float64)
                t_tag = np.asarray(d0.pose_t, dtype=np.float64).reshape(3)
                best_tag = {
                    "id": d0.tag_id,
                    "R": R_tag,
                    "t": t_tag,
                    "size": t_size,
                    "corners": np.asarray(d0.corners, dtype=np.float64),
                }

                # Calculate selected corner origin offset and name
                corner_offset, corner_name = get_corner_offset_in_tag(active_corner_idx, t_size)

                # Draw clearly marked Corner Origin with coordinate axes
                draw_tag_with_corner_origin(annotated_color, best_tag, K, active_corner_idx=active_corner_idx)

            R_tag_ref = best_tag["R"] if best_tag else None
            t_tag_ref = best_tag["t"] if best_tag else None

            # 2. YOLO Object Detection
            yolo_res = yolo.predict(source=color_img, conf=args.conf, verbose=False)[0]
            detected_objects = []

            for box in yolo_res.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = yolo_res.names[cls_id].lower()
                if cls_name in TARGET_OBJECT_CLASSES:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                    conf = float(box.conf[0].item())
                    display_cls = DISPLAY_NAME_MAP.get(cls_name, cls_name)

                    # Construct 3D Surface Grid
                    curr_grid_dim = grid_sizes[grid_size_idx]
                    grid_data = generate_object_surface_grid(
                        [x1, y1, x2, y2], depth_map, K,
                        R_tag_ref, t_tag_ref, corner_offset,
                        grid_rows=curr_grid_dim, grid_cols=curr_grid_dim
                    )

                    col = class_palette.get(cls_name, (0, 255, 255))
                    detected_objects.append({
                        "class": display_cls,
                        "raw_class": cls_name,
                        "conf": conf,
                        "bbox": [x1, y1, x2, y2],
                        "grid_data": grid_data,
                        "color": col,
                    })

            # 3. Draw Object Bounding Boxes and 3D Surface Grids
            for idx, obj in enumerate(detected_objects):
                bx1, by1, bx2, by2 = obj["bbox"]
                col = obj["color"]

                # Draw 2D bounding box
                cv2.rectangle(annotated_color, (bx1, by1), (bx2, by2), col, 2)
                lbl = f"#{idx+1} {obj['class'].upper()} ({obj['conf']:.2f})"
                cv2.putText(annotated_color, lbl, (bx1, max(18, by1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 2, cv2.LINE_AA)

                # Overlay 3D surface point grid
                draw_object_grid(annotated_color, obj["grid_data"], color_scheme=col, draw_mesh=draw_mesh)

            # 4. Handle Interactive Clicked Inspection
            click_info = None
            if tracker.clicked_pt is not None:
                cx_px, cy_px = int(tracker.clicked_pt[0]), int(tracker.clicked_pt[1])
                cv2.drawMarker(annotated_color, (cx_px, cy_px), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.circle(annotated_color, (cx_px, cy_px), 5, (0, 255, 255), 2)

                d_click = get_robust_median_depth(depth_map, cx_px, cy_px, window=4)
                if d_click > 0.05:
                    p_cam = backproject_pixel(cx_px, cy_px, d_click, K)
                    p_table = None
                    if R_tag_ref is not None and t_tag_ref is not None:
                        # Vector in tag frame relative to the chosen corner origin (in meters)
                        pt_table_rel_m = R_tag_ref.T @ (p_cam - t_tag_ref) - corner_offset
                        p_table = pt_table_rel_m * 100.0  # in cm

                        # Active origin in 3D camera space
                        p_cam_origin = t_tag_ref + R_tag_ref @ corner_offset

                        # Draw direct distance line and orthogonal 3D X, Y, Z lines from Corner Origin!
                        draw_axis_measurement_from_corner(
                            annotated_color, p_cam_origin, p_cam, pt_table_rel_m, R_tag_ref, K
                        )

                    click_info = {
                        "clicked": True,
                        "u": cx_px,
                        "v": cy_px,
                        "depth_m": d_click,
                        "cam_xyz_cm": (p_cam * 100.0).tolist(),
                        "table_xyz_cm": p_table.tolist() if p_table is not None else None,
                        "dist_cm": float(np.linalg.norm(p_table)) if p_table is not None else float(np.linalg.norm(p_cam)*100.0),
                    }
                    lbl_click = f"Point: [{p_cam[0]*100:+.1f}, {p_cam[1]*100:+.1f}, {p_cam[2]*100:+.1f}]cm"
                    cv2.putText(annotated_color, lbl_click, (cx_px + 10, cy_px - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            # 5. Assemble 4-Quadrant 1280x720 / 1920x1080 Unified Dashboard
            t1 = time.time()
            instant_fps = 1.0 / max(1e-4, t1 - t0)
            fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps

            # Q1: Color View with Grids
            q1 = cv2.resize(annotated_color, (panel_w, panel_h))
            cv2.putText(q1, f"RealSense Color + 3D Grids | {fps_smooth:.1f} FPS", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            # Q2: Telemetry HUD
            q2 = render_telemetry_hud(
                panel_w, panel_h, detected_objects, best_tag, corner_name,
                tracker.selected_object_idx, click_info, fps_smooth,
                grid_sizes[grid_size_idx]
            )

            # Q3: Metric Depth Colormap
            valid_d = depth_map > 0
            depth_vis = np.zeros_like(depth_map, dtype=np.uint8)
            if valid_d.any():
                depth_vis[valid_d] = np.clip((depth_map[valid_d] - 0.3) / (2.0 - 0.3) * 255.0, 0, 255).astype(np.uint8)
            depth_turbo = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
            q3 = cv2.resize(depth_turbo, (panel_w, panel_h))
            cv2.putText(q3, "RealSense Aligned Depth Colormap", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Q4: Tabletop BEV Radar Map
            q4 = render_bev_tabletop_map(panel_w, panel_h, detected_objects, best_tag, corner_name=corner_name, clicked_pt_info=click_info)

            top_row = np.hstack([q1, q2])
            bot_row = np.hstack([q3, q4])
            dashboard = np.vstack([top_row, bot_row])

            if not getattr(args, "headless", False):
                cv2.imshow(window_name, dashboard)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255

            if getattr(args, "snapshot", None):
                cv2.imwrite(args.snapshot, dashboard)
                log.info(f"Saved demo snapshot to {args.snapshot} and exiting.")
                break

            if key == ord('q') or key == 27:
                break
            elif key == ord('o'):
                # Cycle between corners: 0 (TL) -> 1 (TR) -> 2 (BR) -> 3 (BL) -> -1 (Center)
                corner_cycle = [0, 1, 2, 3, -1]
                curr_pos = corner_cycle.index(active_corner_idx) if active_corner_idx in corner_cycle else 0
                active_corner_idx = corner_cycle[(curr_pos + 1) % len(corner_cycle)]
                _, next_c_name = get_corner_offset_in_tag(active_corner_idx, 0.1)
                log.info(f"Origin switched to: {next_c_name}")
            elif key == ord('g'):
                grid_size_idx = (grid_size_idx + 1) % len(grid_sizes)
                log.info(f"Switched grid resolution to {grid_sizes[grid_size_idx]}x{grid_sizes[grid_size_idx]}")
            elif key == ord('m'):
                draw_mesh = not draw_mesh
            elif key == ord('c'):
                tracker.clicked_pt = None
            elif key == ord('n') and use_mock:
                frame_idx += 1
            elif key == ord('s'):
                snap_name = f"realsense_grid_snapshot_{int(time.time())}.png"
                cv2.imwrite(snap_name, dashboard)
                log.info(f"Saved snapshot: {snap_name}")

    finally:
        if streamer:
            streamer.stop()
        if not getattr(args, "headless", False):
            cv2.destroyAllWindows()

    return 0


def main():
    parser = argparse.ArgumentParser(description="RealSense Object Surface Grid Triangulator & Measurer Demo")
    parser.add_argument("--yolo-model", type=str, default=DEFAULT_YOLO_MODEL,
                        help="Path to YOLO weights (default: yolo26s.pt)")
    parser.add_argument("--conf", type=float, default=0.10,
                        help="YOLO detection confidence threshold")
    parser.add_argument("--calib", default=DEFAULT_CALIB_DIR,
                        help="Camera calibration directory or file")
    parser.add_argument("--factory-calib", action="store_true", default=False,
                        help="Use RealSense EEPROM factory intrinsics")
    parser.add_argument("--tag-family", default=DEFAULT_TAG_FAMILY,
                        help=f"AprilTag family (default: {DEFAULT_TAG_FAMILY})")
    parser.add_argument("--tag-size", type=float, default=DEFAULT_TAG_SIZE_M,
                        help=f"Default AprilTag size in meters (default: {DEFAULT_TAG_SIZE_M})")
    parser.add_argument("--marker-size", action="append",
                        help="Override marker size for specific ID, format: ID=METRES")
    parser.add_argument("--corner-id", type=int, default=0, choices=[-1, 0, 1, 2, 3],
                        help="Chosen origin corner ID: 0=Top-Left, 1=Top-Right, 2=Bottom-Right, 3=Bottom-Left, -1=Center (default: 0)")
    parser.add_argument("--corner-origin", action="store_true", default=True,
                        help="Report measurements relative to physical marker corner")
    parser.add_argument("--mock", action="store_true",
                        help="Force simulated tabletop playback")
    parser.add_argument("--snapshot", type=str, default=None,
                        help="Save single frame snapshot to file and exit")
    parser.add_argument("--headless", action="store_true",
                        help="Run without opening GUI window")
    args = parser.parse_args()
    return run_demo(args)


if __name__ == "__main__":
    sys.exit(main())
