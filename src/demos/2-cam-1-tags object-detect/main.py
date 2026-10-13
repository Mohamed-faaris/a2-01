#!/usr/bin/env python3
"""Two-Camera, One-AprilTag Relative Object Detection & Triangulation Pipeline.

Architecture:
  - Real-time frame dropping and FPS limiter (default: 10 FPS, non-queuing latest-frame grabber).
  - Connects to 2 cameras (webcams, video files, synthetic streams, or static frames).
  - Automatically loads camera calibrations from custom-calibrator NPZ archives.
  - Automatically undistorts frames using OpenCV cv2.undistort.
  - Detects AprilTags using pupil-apriltags to register camera poses relative to a reference tag.
  - Performs object detection with YOLO (classes configurable via CLI or code defaults).
  - Triangulates detected objects in 3D using calibrated projection matrices.
  - Expresses 3D object locations relative to the reference AprilTag (center or measured corner).
  - 2x2 Unified UI Layout:
      Top:    Camera 1 Video Stream    | Table of all found objects & AprilTags
      Bottom: Camera 2 Video Stream    | 6D Poses & 3D Coordinates
"""

from __future__ import annotations

import argparse
import logging
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from pupil_apriltags import Detector
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("2cam-1tag-object-detect")

# Default AprilTag physical marker sizes in meters
DEFAULT_MARKER_SIZES_M: dict[int, float] = {
    0: 0.1240,
    1: 0.1240,
    2: 0.0885,
    3: 0.0885,
    4: 0.0885,
    5: 0.0885,
    6: 0.0575,
    7: 0.0575,
    8: 0.0575,
    9: 0.0575,
    10: 0.0575,
    11: 0.0575,
    12: 0.0575,
    13: 0.0575,
}

# Default YOLO classes to detect and triangulate
DEFAULT_YOLO_CLASSES: list[str] = [
    "bottle", "cup", "can", "vase", "bowl", "apple", "orange", "cell phone"
]

DEFAULT_YOLO_MODEL: str = "/home/mfk01/Projects/econ-systems/demos/demo2/yolo26s.pt"


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
        size = tuple(int(v) for v in data["image_size"])  # (w, h)
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
        if calib.stream_type:
            log.info(f"Loaded calibration '{calib.name}' stream type: '{calib.stream_type}' (Card: {calib.card_name or 'Unknown'})")
            # Consider color variants like "Color (MJPEG)", "RGB / Color", "YUYV" as color streams
            is_color_calib = any(c in calib.stream_type.lower() for c in ("color", "rgb", "bgr", "mjpeg", "yuyv"))
            is_expected_color = any(c in (expected_stream_type or "").lower() for c in ("color", "rgb", "bgr"))
            if is_expected_color and not is_color_calib:
                log.warning(
                    f"Stream type mismatch for {calib.name}! Expected color stream, but calibrator produced '{calib.stream_type}'."
                )
            else:
                log.info(f"Calibration {calib.name} stream type verified: '{calib.stream_type}'.")
        else:
            log.warning(f"Calibration {calib.name} does not specify 'stream_type'. Assuming default {expected_stream_type or 'RGB / Color'}.")


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

    try:
        idx = int(node)
        cap = cv2.VideoCapture(idx)
    except ValueError:
        cap = cv2.VideoCapture(str(node))

    if not cap.isOpened():
        log.warning(f"Could not open source {node}; attempting fallback index 0")
        cap = cv2.VideoCapture(0)

    if calib and calib.image_size:
        w, h = calib.image_size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    # Drop internal buffer size to 1 to reduce driver latency
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


def detect_all_tags(gray: np.ndarray, K: np.ndarray, detector: Detector, marker_sizes: dict[int, float], default_size: float):
    """Detect all visible AprilTags with 6D poses."""
    intrinsics = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    # Run tag pose estimation
    dets = detector.detect(gray, estimate_tag_pose=True, camera_params=intrinsics, tag_size=default_size)
    tag_results = []
    for d in dets:
        size = marker_sizes.get(d.tag_id, default_size)
        # Re-estimate if size differs from default
        if abs(size - default_size) > 1e-4:
            d_spec = detector.detect(gray, estimate_tag_pose=True, camera_params=intrinsics, tag_size=size)
            for ds in d_spec:
                if ds.tag_id == d.tag_id:
                    d = ds
                    break
        R = np.asarray(d.pose_R, dtype=np.float64)
        t = np.asarray(d.pose_t, dtype=np.float64).reshape(3)
        tag_results.append({
            "id": d.tag_id,
            "R": R,
            "t": t,
            "corners": np.asarray(d.corners, dtype=np.float64),
            "size": size,
            "center_px": [float(d.center[0]), float(d.center[1])],
        })
    return tag_results


def detect_yolo_objects(frame_bgr: np.ndarray, model: YOLO, target_classes: set[str], conf_thresh: float = 0.15) -> list[dict]:
    results = model.predict(source=frame_bgr, conf=conf_thresh, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        cls_name = results.names[cls_id].lower()
        if cls_name in target_classes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].cpu().numpy()]
            detections.append({
                "class": cls_name,
                "confidence": float(box.conf[0].item()),
                "bbox_xyxy": [x1, y1, x2, y2],
                "center_px": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                "width_px": x2 - x1,
                "height_px": y2 - y1,
            })
    return detections


def triangulate_points(pt0_px: tuple[float, float], pt1_px: tuple[float, float], P0: np.ndarray, P1: np.ndarray,
                       K0: np.ndarray, K1: np.ndarray,
                       R0_w: np.ndarray, t0_w: np.ndarray,
                       R1_w: np.ndarray, t1_w: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Triangulate 3D point and compute both reprojection error (px) and 3D ray distance / difference (cm)."""
    pts0 = np.array(pt0_px, dtype=np.float64).reshape(2, 1)
    pts1 = np.array(pt1_px, dtype=np.float64).reshape(2, 1)
    homo = cv2.triangulatePoints(P0, P1, pts0, pts1).reshape(4)
    if abs(homo[3]) < 1e-9:
        return np.zeros(3), float("inf"), float("inf")
    pt3d = homo[:3] / homo[3]
    pt3d_h = np.append(pt3d, 1.0)
    p0_p = P0 @ pt3d_h
    p1_p = P1 @ pt3d_h
    e0 = float(np.linalg.norm(p0_p[:2] / p0_p[2] - np.array(pt0_px)))
    e1 = float(np.linalg.norm(p1_p[:2] / p1_p[2] - np.array(pt1_px)))
    reproj_err_px = (e0 + e1) / 2.0

    # Ray difference: shortest 3D distance between the two viewing optical rays
    # Optical centers in world (Tag) frame: C = -R^T * t
    c0 = -R0_w.T @ t0_w
    c1 = -R1_w.T @ t1_w

    # Ray directions in world frame
    v0_cam = np.linalg.inv(K0) @ np.array([pt0_px[0], pt0_px[1], 1.0])
    v1_cam = np.linalg.inv(K1) @ np.array([pt1_px[0], pt1_px[1], 1.0])
    d0 = R0_w.T @ v0_cam
    d0 = d0 / max(1e-9, np.linalg.norm(d0))
    d1 = R1_w.T @ v1_cam
    d1 = d1 / max(1e-9, np.linalg.norm(d1))

    # Shortest distance between skew lines L0: c0 + s*d0 and L1: c1 + t*d1
    cross_d = np.cross(d0, d1)
    cross_norm = np.linalg.norm(cross_d)
    if cross_norm < 1e-6:
        # Near-parallel rays
        ray_diff_cm = float(np.linalg.norm(np.cross(c1 - c0, d0))) * 100.0
    else:
        # Distance = |(c1 - c0) . (d0 x d1)| / ||d0 x d1||
        ray_diff_cm = float(abs(np.dot(c1 - c0, cross_d)) / cross_norm) * 100.0

    return pt3d, reproj_err_px, ray_diff_cm


def associate_stereo_detections(dets0: list[dict], dets1: list[dict]) -> list[tuple[dict, dict]]:
    matched = []
    used_1 = set()
    for d0 in dets0:
        candidates = [
            (i1, d1) for i1, d1 in enumerate(dets1)
            if i1 not in used_1 and d1["class"] == d0["class"]
        ]
        if candidates:
            best_i1, best_d1 = min(candidates, key=lambda pair: abs(pair[1]["center_px"][0] - d0["center_px"][0]))
            matched.append((d0, best_d1))
            used_1.add(best_i1)
    return matched


def rotation_matrix_to_euler_deg(R: np.ndarray) -> tuple[float, float, float]:
    """Convert 3x3 rotation matrix to Euler angles (Roll, Pitch, Yaw) in degrees."""
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def render_table_panel(width: int, height: int, all_tags: list[dict], objects_found: list[dict], fps: float) -> np.ndarray:
    """Render top-right panel: Table of all found objects and tags."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (24, 24, 28)

    # Header
    cv2.rectangle(panel, (0, 0), (width, 42), (40, 40, 48), -1)
    cv2.putText(panel, f"FOUND OBJECTS & TAGS (Real-time: {fps:.1f} FPS)", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    y = 68
    # Section 1: AprilTags
    cv2.putText(panel, f"APRILTAGS DETECTED ({len(all_tags)})", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 50), 2, cv2.LINE_AA)
    y += 10
    cv2.line(panel, (16, y), (width - 16, y), (60, 60, 70), 1)
    y += 24

    if not all_tags:
        cv2.putText(panel, "No AprilTags in field of view", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 130), 1)
        y += 28
    else:
        for t in all_tags:
            tag_id = t["id"]
            pos = t["t"]
            dist_m = float(np.linalg.norm(pos))
            txt = f"Tag #{tag_id:<2} | Dist: {dist_m*100:5.1f}cm | Cam Pos: [{pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}]m"
            cv2.putText(panel, txt, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 240, 255), 1, cv2.LINE_AA)
            y += 24

    y += 16
    # Section 2: Detected Objects
    cv2.putText(panel, f"OBJECTS DETECTED ({len(objects_found)})", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 120), 2, cv2.LINE_AA)
    y += 10
    cv2.line(panel, (16, y), (width - 16, y), (60, 60, 70), 1)
    y += 24

    if not objects_found:
        cv2.putText(panel, "No target objects detected", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 130), 1)
    else:
        # Table Header
        cv2.putText(panel, f"{'ID':<5} {'CLASS':<9} {'CONF':<6} {'EUCLID DIST':<14} {'STATUS'}", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 160), 1)
        y += 20
        for obj in objects_found[:10]:
            oid = obj.get("id", "O01")
            cname = obj.get("class", "unknown")
            conf = obj.get("confidence", 0.0)
            dist_val = obj.get("euclidean_dist_cm")
            dist_str = f"{dist_val:5.1f} cm" if dist_val is not None else "--"
            status = "3D TRACK" if obj.get("triangulated") else "2D Only"
            color = (120, 255, 120) if obj.get("triangulated") else (180, 180, 180)
            txt = f"{oid:<5} {cname:<9} {conf:4.2f}   {dist_str:<14} {status}"
            cv2.putText(panel, txt, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.39, color, 1, cv2.LINE_AA)
            y += 22

    return panel


def render_poses_panel(width: int, height: int, poses: list[dict], ref_tag_info: dict | None) -> np.ndarray:
    """Render bottom-right panel: 6D Poses & 3D Points Table with ID, Euclidean Distance, and Ray Difference."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (20, 20, 24)

    # Header
    cv2.rectangle(panel, (0, 0), (width, 42), (36, 36, 44), -1)
    cv2.putText(panel, "6D POSES & STEREO RAY DIFFERENCE", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)

    y = 66
    # Reference frame status
    if ref_tag_info:
        r_txt = f"Origin: Tag #{ref_tag_info['id']} ({'Corner' if ref_tag_info['corner'] else 'Center'}) | Stereo Baselines Calibrated"
        cv2.putText(panel, r_txt, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 220, 255), 1)
    else:
        cv2.putText(panel, "NO ORIGIN: Searching for AprilTag #1 in both cameras...", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 255), 1)
    y += 12
    cv2.line(panel, (16, y), (width - 16, y), (60, 60, 70), 1)
    y += 22

    # Table Header with ID, Target, Euclid Norm, X, Y, Z, Ray Diff (cm) or R, P, Y
    cv2.putText(panel, f"{'ID':<5} {'TARGET':<7} | {'EUCLID':<8} | {'X(cm)':<6} {'Y(cm)':<6} {'Z(cm)':<6} | {'RAY DIFF':<9} {'REPROJ'}", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 170), 1)
    y += 18

    if not poses:
        cv2.putText(panel, "Awaiting 3D stereo triangulations...", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 130), 1)
    else:
        for p in poses[:12]:
            pid = p.get("id", "O01")
            name = p["name"]
            x, y_pos, z = p["xyz_cm"]
            euclid_cm = math.sqrt(x * x + y_pos * y_pos + z * z)
            is_tag = p.get("is_tag", False)
            is_cam = p.get("is_cam", False)

            if is_cam:
                color = (255, 120, 255)
                extra = f"{p.get('baseline_str', '')}"
            elif is_tag:
                color = (255, 220, 100)
                r, pitch, yaw = p.get("rpy_deg", (0.0, 0.0, 0.0))
                extra = f"R:{r:+.0f}° P:{pitch:+.0f}° Y:{yaw:+.0f}°"
            else:
                color = (100, 255, 100)
                ray_diff = p.get("ray_diff_cm")
                reproj = p.get("reproj_px")
                ray_str = f"{ray_diff:5.2f}cm" if ray_diff is not None else "--"
                rep_str = f"{reproj:4.1f}px" if reproj is not None else "--"
                extra = f"{ray_str:<9} {rep_str}"

            row_str = f"{pid:<5} {name:<7} | {euclid_cm:5.1f}cm | {x:+6.1f} {y_pos:+6.1f} {z:+6.1f} | {extra}"
            cv2.putText(panel, row_str, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
            y += 21

    return panel


def main() -> int:
    parser = argparse.ArgumentParser(
        description="2-Camera 1-Tag Relative Object Detection & Triangulation (Real-Time 10 FPS Dropping Limiter)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Camera Sources
    parser.add_argument("--cam0", default=None, help="Camera 0 device node, video file, or image (default: /dev/video0 or 0)")
    parser.add_argument("--cam1", default=None, help="Camera 1 device node, video file, or image (default: /dev/video1 or 1)")
    # Calibration files (.npz)
    parser.add_argument("--calib0", type=Path, default=None, help="Camera 0 calibration NPZ")
    parser.add_argument("--calib1", type=Path, default=None, help="Camera 1 calibration NPZ")
    # Tag parameters
    parser.add_argument("--tag-id", type=int, default=1, help="Reference AprilTag ID defining world coordinate origin")
    parser.add_argument("--tag-size", type=float, default=0.0885, help="Physical edge length of default AprilTag in meters")
    parser.add_argument("--marker-size", action="append", help="Override marker size for specific ID, format: ID=METRES (e.g. 0=0.124)")
    parser.add_argument("--corner-origin", action="store_true", help="Report relative positions from measurable physical marker corner rather than center")
    # FPS Limiter (Real-time frame dropping)
    parser.add_argument("--fps", type=float, default=10.0, help="Target FPS limiter rate. Drops buffered frames to maintain zero-latency real-time processing")
    # YOLO parameters
    parser.add_argument("--yolo-weights", type=Path, default=Path(DEFAULT_YOLO_MODEL), help="Path to YOLO weights (.pt)")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_YOLO_CLASSES, help="YOLO classes to detect and triangulate")
    parser.add_argument("--conf", type=float, default=0.15, help="YOLO confidence detection threshold")
    # Processing options
    parser.add_argument("--tag-family", default="tag36h11", help="AprilTag dictionary family")
    parser.add_argument("--undistort", action="store_true", default=True, help="Undistort camera frames using calibration dist_coeffs")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI windows")
    args = parser.parse_args()

    # 1. Load Calibrations
    calib0 = CameraCalibration.from_npz(args.calib0) if args.calib0 and args.calib0.exists() else CameraCalibration.fallback_fov("cam0")
    calib1 = CameraCalibration.from_npz(args.calib1) if args.calib1 and args.calib1.exists() else CameraCalibration.fallback_fov("cam1")

    marker_sizes = parse_marker_sizes(args.marker_size, default_size=args.tag_size)
    ref_marker_size = marker_sizes.get(args.tag_id, args.tag_size)

    # 2. Initialize Detectors
    detector = Detector(families=args.tag_family, nthreads=2, quad_decimate=1.0)
    yolo_model = YOLO(str(args.yolo_weights))
    target_classes = {c.lower() for c in args.classes}

    log.info(f"Loaded YOLO model: {args.yolo_weights} (detecting classes: {sorted(target_classes)})")
    log.info(f"Reference Tag ID: {args.tag_id} (size: {ref_marker_size*100:.1f} cm)")
    log.info(f"Real-time FPS Limiter set to: {args.fps:.1f} FPS (buffer dropping enabled)")

    # 3. Open Video Sources via Non-Queuing Readers
    cap0_raw = open_camera_source(args.cam0, calib0)
    cap1_raw = open_camera_source(args.cam1, calib1)
    reader0 = NonQueuingCameraReader(cap0_raw)
    reader1 = NonQueuingCameraReader(cap1_raw)

    log.info("Starting relative stereo object tracking loop... Press 'q' or ESC to exit.")

    frame_interval = 1.0 / max(0.1, args.fps)
    last_process_time = 0.0

    fps_smooth = args.fps

    try:
        while True:
            now = time.time()
            elapsed = now - last_process_time
            if elapsed < frame_interval:
                time.sleep(max(0.001, frame_interval - elapsed))
                continue

            last_process_time = time.time()

            # Retrieve latest frame (any stale buffered frames were drained in background)
            frame0 = reader0.get_latest_frame()
            frame1 = reader1.get_latest_frame()

            if frame0 is None or frame1 is None:
                time.sleep(0.01)
                continue

            # Undistort if requested and calibrated
            if args.undistort and np.any(calib0.dist_coeffs):
                frame0 = cv2.undistort(frame0, calib0.camera_matrix, calib0.dist_coeffs)
            if args.undistort and np.any(calib1.dist_coeffs):
                frame1 = cv2.undistort(frame1, calib1.camera_matrix, calib1.dist_coeffs)

            gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

            # Detect all AprilTags in both cameras
            tags0 = detect_all_tags(gray0, calib0.camera_matrix, detector, marker_sizes, ref_marker_size)
            tags1 = detect_all_tags(gray1, calib1.camera_matrix, detector, marker_sizes, ref_marker_size)

            ref_tag0 = next((t for t in tags0 if t["id"] == args.tag_id), None)
            ref_tag1 = next((t for t in tags1 if t["id"] == args.tag_id), None)

            # Detect Objects with YOLO
            objs0 = detect_yolo_objects(frame0, yolo_model, target_classes, conf_thresh=args.conf)
            objs1 = detect_yolo_objects(frame1, yolo_model, target_classes, conf_thresh=args.conf)

            annotated0 = frame0.copy()
            annotated1 = frame1.copy()

            # Draw All Tags
            for t in tags0:
                corners = t["corners"].astype(int)
                is_ref = (t["id"] == args.tag_id)
                col = (0, 255, 255) if is_ref else (255, 180, 0)
                cv2.polylines(annotated0, [corners], True, col, 2)
                cv2.putText(annotated0, f"Tag #{t['id']}{' (Ref)' if is_ref else ''}",
                            (corners[0, 0], corners[0, 1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

            for t in tags1:
                corners = t["corners"].astype(int)
                is_ref = (t["id"] == args.tag_id)
                col = (0, 255, 255) if is_ref else (255, 180, 0)
                cv2.polylines(annotated1, [corners], True, col, 2)
                cv2.putText(annotated1, f"Tag #{t['id']}{' (Ref)' if is_ref else ''}",
                            (corners[0, 0], corners[0, 1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

            # Draw 2D bounding boxes for all detected objects on Camera 0
            for idx, o in enumerate(objs0):
                b0 = [int(v) for v in o["bbox_xyxy"]]
                cls_prefix = "b" if o["class"] == "bottle" else ("c" if o["class"] == "can" else ("A" if o["class"] == "apple" else o["class"][0].upper()))
                obj_id = f"{cls_prefix}{idx + 1:02d}"
                cv2.rectangle(annotated0, (b0[0], b0[1]), (b0[2], b0[3]), (0, 200, 255), 2)
                cx, cy = int(o["center_px"][0]), int(o["center_px"][1])
                cv2.circle(annotated0, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(annotated0, f"[{obj_id}] {o['class']} {o['confidence']:.2f}",
                            (b0[0], max(16, b0[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

            # Draw 2D bounding boxes for all detected objects on Camera 1
            for o in objs1:
                b1 = [int(v) for v in o["bbox_xyxy"]]
                cv2.rectangle(annotated1, (b1[0], b1[1]), (b1[2], b1[3]), (0, 200, 255), 2)
                cx, cy = int(o["center_px"][0]), int(o["center_px"][1])
                cv2.circle(annotated1, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(annotated1, f"{o['class']} {o['confidence']:.2f}",
                            (b1[0], max(16, b1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

            poses_list: list[dict] = []
            objects_list: list[dict] = []

            # Populate detected objects list for display with category-prefixed IDs
            for idx, o in enumerate(objs0):
                cls_prefix = "b" if o["class"] == "bottle" else ("c" if o["class"] == "can" else ("A" if o["class"] == "apple" else o["class"][0].upper()))
                obj_id = f"{cls_prefix}{idx + 1:02d}"
                objects_list.append({
                    "id": obj_id,
                    "class": o["class"],
                    "confidence": o["confidence"],
                    "triangulated": False,
                })

            # Check if Reference AprilTag is detected in both cameras
            if ref_tag0 and ref_tag1:
                R_c0_t, t_c0_t = ref_tag0["R"], ref_tag0["t"]
                R_c1_t, t_c1_t = ref_tag1["R"], ref_tag1["t"]

                P0 = calib0.camera_matrix @ np.hstack((R_c0_t, t_c0_t.reshape(3, 1)))
                P1 = calib1.camera_matrix @ np.hstack((R_c1_t, t_c1_t.reshape(3, 1)))

                corner_offset = np.array([-ref_marker_size / 2.0, -ref_marker_size / 2.0, 0.0]) if args.corner_origin else np.zeros(3)

                # 1. Compute Camera 1 and Camera 2 6D poses in the reference (world) frame
                # Camera position in tag frame: C = -R^T * t
                p_cam0_tag = -R_c0_t.T @ t_c0_t - corner_offset
                p_cam1_tag = -R_c1_t.T @ t_c1_t - corner_offset
                rpy_c0 = rotation_matrix_to_euler_deg(R_c0_t.T)
                rpy_c1 = rotation_matrix_to_euler_deg(R_c1_t.T)
                baseline_m = float(np.linalg.norm(p_cam1_tag - p_cam0_tag))

                poses_list.append({
                    "id": "C01",
                    "name": "Cam0",
                    "xyz_cm": [p_cam0_tag[0] * 100.0, p_cam0_tag[1] * 100.0, p_cam0_tag[2] * 100.0],
                    "rpy_deg": rpy_c0,
                    "is_cam": True,
                    "baseline_str": f"Base: {baseline_m*100:4.1f}cm",
                })
                poses_list.append({
                    "id": "C02",
                    "name": "Cam1",
                    "xyz_cm": [p_cam1_tag[0] * 100.0, p_cam1_tag[1] * 100.0, p_cam1_tag[2] * 100.0],
                    "rpy_deg": rpy_c1,
                    "is_cam": True,
                    "baseline_str": f"Base: {baseline_m*100:4.1f}cm",
                })

                # 2. Add all detected tags into 6D pose list relative to reference
                for t in tags0:
                    rpy = rotation_matrix_to_euler_deg(t["R"])
                    p_in_ref = R_c0_t.T @ (t["t"] - t_c0_t) - corner_offset
                    poses_list.append({
                        "id": f"T{t['id']:02d}",
                        "name": f"Tag #{t['id']}",
                        "xyz_cm": [p_in_ref[0] * 100.0, p_in_ref[1] * 100.0, p_in_ref[2] * 100.0],
                        "rpy_deg": rpy,
                        "is_tag": True,
                    })

                matched_pairs = associate_stereo_detections(objs0, objs1)

                for idx, (d0, d1) in enumerate(matched_pairs):
                    pt_tag, reproj_px, ray_diff_cm = triangulate_points(
                        tuple(d0["center_px"]), tuple(d1["center_px"]),
                        P0, P1, calib0.camera_matrix, calib1.camera_matrix,
                        R_c0_t, t_c0_t, R_c1_t, t_c1_t
                    )
                    pt_rel = pt_tag - corner_offset
                    dist_rel = float(np.linalg.norm(pt_rel))

                    obj_id = objects_list[idx]["id"] if idx < len(objects_list) else f"b{idx+1:02d}"

                    # Update status and Euclidean distance in object list
                    if idx < len(objects_list):
                        objects_list[idx]["triangulated"] = True
                        objects_list[idx]["euclidean_dist_cm"] = dist_rel * 100.0

                    poses_list.append({
                        "id": obj_id,
                        "name": d0["class"][:10],
                        "xyz_cm": [pt_rel[0] * 100.0, pt_rel[1] * 100.0, pt_rel[2] * 100.0],
                        "rpy_deg": (0.0, 0.0, 0.0),
                        "ray_diff_cm": ray_diff_cm,
                        "reproj_px": reproj_px,
                        "is_tag": False,
                    })

                    # Draw triangulated 3D overlay with Euclidean distance and ray difference
                    b0 = [int(v) for v in d0["bbox_xyxy"]]
                    cv2.rectangle(annotated0, (b0[0], b0[1]), (b0[2], b0[3]), (0, 255, 0), 2)
                    lbl0_top = f"[{obj_id}] {d0['class']} d={dist_rel*100:.1f}cm"
                    lbl0_bot = f"ray_diff={ray_diff_cm:.1f}cm err={reproj_px:.1f}px"
                    cv2.putText(annotated0, lbl0_top, (b0[0], max(16, b0[1] - 18)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                    cv2.putText(annotated0, lbl0_bot, (b0[0], max(32, b0[1] - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)

                    b1 = [int(v) for v in d1["bbox_xyxy"]]
                    cv2.rectangle(annotated1, (b1[0], b1[1]), (b1[2], b1[3]), (0, 255, 0), 2)
                    lbl1_top = f"[{obj_id}] {d1['class']} d={dist_rel*100:.1f}cm"
                    lbl1_bot = f"ray_diff={ray_diff_cm:.1f}cm err={reproj_px:.1f}px"
                    cv2.putText(annotated1, lbl1_top, (b1[0], max(16, b1[1] - 18)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                    cv2.putText(annotated1, lbl1_bot, (b1[0], max(32, b1[1] - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)
            else:
                msg = "NO ORIGIN: Searching for AprilTag #1 in both cameras..."
                cv2.putText(annotated0, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.putText(annotated1, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            # Smooth FPS calculation
            instant_fps = 1.0 / max(1e-4, time.time() - now)
            fps_smooth = 0.9 * fps_smooth + 0.1 * min(args.fps, instant_fps)

            cv2.putText(annotated0, f"Camera 1 ({calib0.name}) | Limiter: {args.fps:.0f} FPS", (16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated1, f"Camera 2 ({calib1.name}) | Real-time: {fps_smooth:.1f} FPS", (16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if not args.headless:
                target_w, target_h = 640, 360
                cam1_view = cv2.resize(annotated0, (target_w, target_h))
                cam2_view = cv2.resize(annotated1, (target_w, target_h))

                # Combine all detected tags
                all_found_tags = tags0 if tags0 else tags1

                table_panel = render_table_panel(target_w, target_h, all_found_tags, objects_list, fps_smooth)
                poses_panel = render_poses_panel(
                    target_w, target_h, poses_list,
                    {"id": args.tag_id, "corner": args.corner_origin} if (ref_tag0 and ref_tag1) else None
                )

                top_row = np.hstack((cam1_view, table_panel))
                bot_row = np.hstack((cam2_view, poses_panel))
                dashboard = np.vstack((top_row, bot_row))

                cv2.imshow("Stereo Object Tracking & 6D Poses Dashboard", dashboard)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

    finally:
        reader0.stop()
        reader1.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
