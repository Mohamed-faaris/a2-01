#!/usr/bin/env python3
"""Intel RealSense D415 Depth + RGB 720p 3D Localization Demo.

Features:
  - Real-time frame dropping and FPS limiter (default: 10 FPS, non-queuing latest-frame grabber).
  - Streams RGB + Depth from Intel RealSense at 720p (1280x720).
  - Aligns depth stream to color stream using rs.align(rs.stream.color).
  - Automatically loads camera intrinsics & distortion coefficients from:
      calibrations/calibarotor0/outputs/rl720/camera_calibration.npz (or .json)
  - Detects AprilTags (using pupil-apriltags) to establish the world/table reference frame.
  - Detects objects with YOLO (default: YOLO26s / YOLO11s).
  - Backprojects (u, v, depth) -> 3D Camera Coordinate System (X, Y, Z).
  - Expresses 3D points relative to the AprilTag plane and corner origin: (x_tag, y_tag, z_tag).
  - 2x2 Unified UI Layout:
      Top:    Camera 1 (Color Stream) | Table of all found objects & AprilTags
      Bottom: Camera 2 (Depth Colormap)| 6D Poses & 3D Coordinates
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
import pyrealsense2 as rs
from pupil_apriltags import Detector
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("realsense-depth-720p")

DEFAULT_CALIB_DIR = "calibrations/calibarotor0/outputs/rl720"
DEFAULT_YOLO_MODEL = "/home/mfk01/Projects/econ-systems/demos/demo2/yolo26s.pt"
FALLBACK_YOLO_MODEL = "yolo11n.pt"
DEFAULT_MARKER_SIZE = 0.124
TARGET_CLASSES = {"bottle", "cup", "can", "vase", "bowl", "apple", "orange", "cell phone"}


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
    def load(cls, calib_path_or_dir: str | Path, expected_stream_type: str = "RGB / Color") -> CameraCalibration:
        p = Path(calib_path_or_dir)
        if p.is_dir():
            npz_candidate = p / "camera_calibration.npz"
            json_candidate = p / "camera_calibration.json"
            if npz_candidate.exists():
                p = npz_candidate
            elif json_candidate.exists():
                p = json_candidate
            else:
                raise FileNotFoundError(f"No camera_calibration.npz or .json found in {p}")

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
        if calib.stream_type:
            log.info(f"Calibration stream type: '{calib.stream_type}' (Camera: {calib.name}, Card: {calib.card_name or 'Unknown'})")
            if expected_stream_type and expected_stream_type.lower() not in calib.stream_type.lower():
                log.warning(
                    f"Stream type mismatch! Expected '{expected_stream_type}', but calibrator produced '{calib.stream_type}'. "
                    f"Verify that calibration matches the active sensor stream."
                )
            else:
                log.info(f"Stream type verified matching expected '{expected_stream_type}'.")
        else:
            log.warning(f"Calibration file does not specify 'stream_type'. Assuming default {expected_stream_type or 'RGB / Color'}.")


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
        """Fetch factory intrinsics from active RealSense color stream profile."""
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
            except Exception as e:
                consecutive_timeouts += 1
                if consecutive_timeouts == 5:
                    log.warning("RealSense frame timeout (USB buffer delay). Still waiting for frames...")

    def get_latest_frames(self):
        with self.lock:
            if self.latest_color is None or self.latest_depth_frame is None:
                return None, None, None
            return self.latest_color.copy(), self.latest_depth.copy(), self.latest_depth_frame

    def wait_until_ready(self, timeout_sec: float = 6.0) -> bool:
        """Block until the first valid frames arrive from the sensor."""
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


def render_table_panel(width: int, height: int, all_tags: list[dict], objects_found: list[dict], fps: float) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (24, 24, 28)

    cv2.rectangle(panel, (0, 0), (width, 42), (40, 40, 48), -1)
    cv2.putText(panel, f"FOUND OBJECTS & TAGS (Real-time: {fps:.1f} FPS)", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    y = 68
    cv2.putText(panel, f"APRILTAGS DETECTED ({len(all_tags)})", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 50), 2, cv2.LINE_AA)
    y += 10
    cv2.line(panel, (16, y), (width - 16, y), (60, 60, 70), 1)
    y += 24

    if not all_tags:
        cv2.putText(panel, "No AprilTags detected", (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 130), 1)
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
    cv2.putText(panel, f"OBJECTS DETECTED ({len(objects_found)})", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 120), 2, cv2.LINE_AA)
    y += 10
    cv2.line(panel, (16, y), (width - 16, y), (60, 60, 70), 1)
    y += 24

    if not objects_found:
        cv2.putText(panel, "No target objects detected", (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 130), 1)
    else:
        cv2.putText(panel, f"{'ID':<5} {'CLASS':<9} {'CONF':<6} {'EUCLID DIST':<14} {'DEPTH(Z)':<10} {'STATUS'}", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 160), 1)
        y += 20
        for obj in objects_found[:10]:
            oid = obj.get("id", "b01")
            cname = obj["class"]
            conf = obj["confidence"]
            dist_val = obj.get("euclidean_dist_cm")
            dist_str = f"{dist_val:5.1f} cm" if dist_val is not None else "--"
            depth_str = f"{obj['depth_m']*100:.1f} cm" if obj["depth_m"] > 0 else "N/A"
            status = "3D TRACK" if obj.get("has_3d") else "2D Only"
            color = (120, 255, 120) if obj.get("has_3d") else (180, 180, 180)
            txt = f"{oid:<5} {cname:<9} {conf:4.2f}   {dist_str:<14} {depth_str:<10} {status}"
            cv2.putText(panel, txt, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.39, color, 1, cv2.LINE_AA)
            y += 22

    return panel


def render_poses_panel(width: int, height: int, poses: list[dict], ref_tag_id: int | None) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (20, 20, 24)

    cv2.rectangle(panel, (0, 0), (width, 42), (36, 36, 44), -1)
    cv2.putText(panel, "6D POSES & 3D RELATIVE COORDINATES", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)

    y = 68
    if ref_tag_id is not None:
        r_txt = f"Origin: AprilTag #{ref_tag_id} Corner | RealSense D415 Dense Depth"
        cv2.putText(panel, r_txt, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 220, 255), 1)
    else:
        cv2.putText(panel, "AprilTag not found - displaying 3D Camera Coordinates", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 255), 1)
    y += 14
    cv2.line(panel, (16, y), (width - 16, y), (60, 60, 70), 1)
    y += 24

    # Table Header with ID, Target, Euclidean Norm ||P||, and 6D values
    cv2.putText(panel, f"{'ID':<5} {'TARGET':<8} | {'EUCLID':<9} | {'X(cm)':<7} {'Y(cm)':<7} {'Z(cm)':<7} | {'R°':<5} {'P°':<5} {'Y°'}", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 170), 1)
    y += 20

    if not poses:
        cv2.putText(panel, "Awaiting valid depth / 3D points...", (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 130), 1)
    else:
        for p in poses[:12]:
            pid = p.get("id", "b01")
            name = p["name"]
            x, y_pos, z = p["xyz_cm"]
            euclid_cm = math.sqrt(x * x + y_pos * y_pos + z * z)
            r, pitch, yaw = p.get("rpy_deg", (0.0, 0.0, 0.0))
            is_tag = p.get("is_tag", False)
            is_cam = p.get("is_cam", False)
            if is_cam:
                color = (255, 120, 255)
            elif is_tag:
                color = (255, 220, 100)
            else:
                color = (100, 255, 100)
            row_str = f"{pid:<5} {name:<8} | {euclid_cm:6.1f} cm | {x:+6.1f} {y_pos:+6.1f} {z:+6.1f} | {r:+4.0f}  {pitch:+4.0f}  {yaw:+4.0f}"
            cv2.putText(panel, row_str, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, color, 1, cv2.LINE_AA)
            y += 22

    return panel


def run_realsense_demo(args: argparse.Namespace):
    calib = CameraCalibration.load(args.calib)
    K = calib.camera_matrix
    D = calib.dist_coeffs

    yolo_model_path = Path(args.yolo_model)
    if not yolo_model_path.exists():
        yolo_model_path = Path(FALLBACK_YOLO_MODEL)
    yolo_model = YOLO(str(yolo_model_path))

    detector = Detector(families=args.tag_family, nthreads=2, quad_decimate=1.0)
    colorizer = rs.colorizer()

    streamer = NonQueuingRealSenseStream(width=1280, height=720, fps=30)
    streamer.start()

    if not streamer.wait_until_ready(timeout_sec=5.0):
        log.warning("No frames received from RealSense within 5s. If USB is stuck, try unplugging and reconnecting.")
    else:
        log.info("RealSense video and depth streams active and verified.")

    if args.factory_calib:
        K, D = streamer.get_color_intrinsics()
        log.info(f"Using RealSense Factory Color Intrinsics: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
    else:
        log.info(f"Loaded Custom Calibration: {calib.name} from {args.calib}")
        log.info(f"Custom Intrinsics: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")

    log.info(f"Real-time FPS Limiter set to: {args.fps:.1f} FPS (buffer dropping enabled)")
    log.info(f"Object radius compensation: {args.bottle_radius*100:.1f} cm (diameter: {args.bottle_radius*200:.1f} cm)")

    frame_interval = 1.0 / max(0.1, args.fps)
    last_process_time = 0.0
    fps_smooth = args.fps

    try:
        frame_idx = 0
        while True:
            now = time.time()
            elapsed = now - last_process_time
            if elapsed < frame_interval:
                time.sleep(max(0.001, frame_interval - elapsed))
                continue

            last_process_time = time.time()

            color_img, depth_img, depth_frame = streamer.get_latest_frames()
            if color_img is None:
                time.sleep(0.01)
                continue

            frame_idx += 1
            if args.undistort and np.any(D):
                color_img = cv2.undistort(color_img, K, D)

            annotated_color = color_img.copy()
            gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

            # Detect AprilTags
            cparams = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
            raw_dets = detector.detect(gray, estimate_tag_pose=True, camera_params=cparams, tag_size=args.tag_size)

            all_tags = []
            ref_tag = None
            for d in raw_dets:
                R = np.asarray(d.pose_R, dtype=np.float64)
                t = np.asarray(d.pose_t, dtype=np.float64).reshape(3)
                tag_dict = {"id": d.tag_id, "R": R, "t": t, "corners": np.asarray(d.corners, dtype=np.float64)}
                all_tags.append(tag_dict)
                if d.tag_id == args.tag_id:
                    ref_tag = tag_dict

                corners = np.asarray(d.corners, dtype=int)
                col = (0, 255, 255) if (d.tag_id == args.tag_id) else (255, 180, 0)
                cv2.polylines(annotated_color, [corners], True, col, 2)
                cv2.circle(annotated_color, (int(corners[0, 0]), int(corners[0, 1])), 5, (0, 0, 255), -1)
                cv2.putText(annotated_color, f"Tag #{d.tag_id}", (corners[0, 0], corners[0, 1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

            # YOLO Object Detection
            yolo_res = yolo_model.predict(source=color_img, conf=args.conf, verbose=False)[0]
            objects_list: list[dict] = []
            poses_list: list[dict] = []

            # 1. Camera 3D Pose in Tag Corner Origin
            if ref_tag:
                s = args.tag_size
                p_cam_in_tag = -ref_tag["R"].T @ ref_tag["t"]
                c_rel_x = float(-(p_cam_in_tag[0] - s / 2.0))
                c_rel_y = float(p_cam_in_tag[1] - (-s / 2.0))
                c_rel_z = float(-p_cam_in_tag[2])
                rpy_cam = rotation_matrix_to_euler_deg(ref_tag["R"].T)
                poses_list.append({
                    "id": "C01",
                    "name": "Camera",
                    "xyz_cm": [c_rel_x * 100.0, c_rel_y * 100.0, c_rel_z * 100.0],
                    "rpy_deg": rpy_cam,
                    "is_cam": True,
                })

            # 2. Populate AprilTag poses with T01, T02... IDs
            for t in all_tags:
                rpy = rotation_matrix_to_euler_deg(t["R"])
                tag_id_str = f"T{t['id']:02d}"
                if ref_tag:
                    p_in_ref = ref_tag["R"].T @ (t["t"] - ref_tag["t"])
                    s = args.tag_size
                    rel_x = float(-(p_in_ref[0] - s / 2.0))
                    rel_y = float(p_in_ref[1] - (-s / 2.0))
                    rel_z = float(-p_in_ref[2])
                    poses_list.append({
                        "id": tag_id_str,
                        "name": f"Tag #{t['id']}",
                        "xyz_cm": [rel_x * 100.0, rel_y * 100.0, rel_z * 100.0],
                        "rpy_deg": rpy,
                        "is_tag": True,
                    })
                else:
                    poses_list.append({
                        "id": tag_id_str,
                        "name": f"Tag #{t['id']}",
                        "xyz_cm": [t["t"][0] * 100.0, t["t"][1] * 100.0, t["t"][2] * 100.0],
                        "rpy_deg": rpy,
                        "is_tag": True,
                    })

            for box in yolo_res.boxes:
                cls_id = int(box.cls[0].item())
                cname = yolo_res.names[cls_id].lower()
                if cname in TARGET_CLASSES:
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].cpu().numpy()]
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    conf = float(box.conf[0].item())

                    measured_depth = get_robust_median_depth(depth_frame, cx, cy)
                    has_3d = measured_depth > 0

                    cls_prefix = "b" if cname == "bottle" else ("c" if cname == "can" else ("A" if cname == "apple" else cname[0]))
                    obj_id = f"{cls_prefix}{len(objects_list) + 1:02d}"

                    obj_info = {"id": obj_id, "class": cname, "confidence": conf, "depth_m": measured_depth, "has_3d": has_3d, "euclidean_dist_cm": None}

                    if has_3d:
                        # 3D surface point on front face
                        p_cam_surface = backproject_to_3d(cx, cy, measured_depth, K)
                        # Compensate cylinder radius along viewing ray to find cylinder axis center
                        ray_norm = np.linalg.norm(p_cam_surface)
                        if ray_norm > 1e-6 and args.bottle_radius > 0:
                            p_cam = p_cam_surface + (p_cam_surface / ray_norm) * args.bottle_radius
                        else:
                            p_cam = p_cam_surface

                        if ref_tag:
                            # Express 3D location in AprilTag reference frame
                            p_in_tag = ref_tag["R"].T @ (p_cam - ref_tag["t"])
                            s = args.tag_size
                            # Exact one_cam_depth_cans_3d transformation relative to tag corner origin:
                            # Origin at [+s/2, -s/2, 0] in tag frame
                            rel_x = float(-(p_in_tag[0] - s / 2.0))
                            rel_y = float(p_in_tag[1] - (-s / 2.0))
                            rel_z = float(-p_in_tag[2])
                            p_rel = np.array([rel_x, rel_y, rel_z], dtype=np.float64)

                            euclid_dist = float(np.linalg.norm(p_rel)) * 100.0
                            obj_info["euclidean_dist_cm"] = euclid_dist
                            poses_list.append({
                                "id": obj_id,
                                "name": cname[:10],
                                "xyz_cm": [rel_x * 100.0, rel_y * 100.0, rel_z * 100.0],
                                "rpy_deg": (0.0, 0.0, 0.0),
                                "is_tag": False,
                            })
                            # Format label just like one_cam_depth_cans_3d
                            label_top = f"[{obj_id}] {cname} (d={euclid_dist:.1f}cm)"
                            label_coords = f"x={rel_x:+.2f}, y={rel_y:+.2f}, z={rel_z:+.2f}m"
                        else:
                            euclid_dist = float(np.linalg.norm(p_cam)) * 100.0
                            obj_info["euclidean_dist_cm"] = euclid_dist
                            poses_list.append({
                                "id": obj_id,
                                "name": cname[:10],
                                "xyz_cm": [p_cam[0] * 100.0, p_cam[1] * 100.0, p_cam[2] * 100.0],
                                "rpy_deg": (0.0, 0.0, 0.0),
                                "is_tag": False,
                            })
                            label_top = f"[{obj_id}] {cname} (Cam Z={measured_depth:.2f}m)"
                            label_coords = f"x={p_cam[0]:+.2f}, y={p_cam[1]:+.2f}, z={p_cam[2]:+.2f}m"

                        # Draw bounding box and 3D coordinate annotations
                        cv2.rectangle(annotated_color, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.circle(annotated_color, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                        cv2.putText(annotated_color, label_top, (int(x1), max(18, int(y1) - 18)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.putText(annotated_color, label_coords, (int(x1), max(32, int(y1) - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
                    else:
                        # 2D only detection fallback
                        cv2.rectangle(annotated_color, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
                        cv2.circle(annotated_color, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                        cv2.putText(annotated_color, f"[{obj_id}] {cname} {conf:.2f} (No Depth)",
                                    (int(x1), max(16, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

                    objects_list.append(obj_info)

            # Generate Colormapped Depth image
            depth_color_frame = colorizer.colorize(depth_frame)
            depth_colormap = np.asanyarray(depth_color_frame.get_data())

            # Draw Depth Overlay targets
            annotated_depth = depth_colormap.copy()
            for o in objects_list:
                if o["has_3d"]:
                    cv2.putText(annotated_depth, f"{o['class']} {o['depth_m']:.2f}m",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # FPS measurement
            instant_fps = 1.0 / max(1e-4, time.time() - now)
            fps_smooth = 0.9 * fps_smooth + 0.1 * min(args.fps, instant_fps)

            cv2.putText(annotated_color, f"RealSense Color (720p) | Limiter: {args.fps:.0f} FPS", (16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated_depth, f"RealSense Depth (Z16) | Real-time: {fps_smooth:.1f} FPS", (16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if not args.no_gui:
                target_w, target_h = 640, 360
                cam1_view = cv2.resize(annotated_color, (target_w, target_h))
                cam2_view = cv2.resize(annotated_depth, (target_w, target_h))

                table_panel = render_table_panel(target_w, target_h, all_tags, objects_list, fps_smooth)
                poses_panel = render_poses_panel(target_w, target_h, poses_list, args.tag_id if ref_tag else None)

                top_row = np.hstack((cam1_view, table_panel))
                bot_row = np.hstack((cam2_view, poses_panel))
                dashboard = np.vstack((top_row, bot_row))

                cv2.imshow("RealSense D415 720p Depth & 6D Poses Dashboard", dashboard)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

            if args.frames and frame_idx >= args.frames:
                break

    finally:
        streamer.stop()
        if not args.no_gui:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Intel RealSense 720p Depth + YOLO Object Localization Demo")
    parser.add_argument("--calib", type=str, default=DEFAULT_CALIB_DIR,
                        help=f"Path to calibration directory or npz/json (default: {DEFAULT_CALIB_DIR})")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Target FPS limiter rate. Drops buffered frames to maintain zero-latency real-time processing (default: 10)")
    parser.add_argument("--yolo-model", type=str, default=DEFAULT_YOLO_MODEL,
                        help=f"Path to YOLO weights (default: {DEFAULT_YOLO_MODEL})")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO detection confidence threshold (default: 0.25)")
    parser.add_argument("--tag-id", type=int, default=1,
                        help="Reference AprilTag ID defining world coordinate origin (default: 1)")
    parser.add_argument("--tag-family", type=str, default="tag36h11",
                        help="AprilTag family (default: tag36h11)")
    parser.add_argument("--tag-size", type=float, default=DEFAULT_MARKER_SIZE,
                        help=f"AprilTag physical marker side length in meters (default: {DEFAULT_MARKER_SIZE}m)")
    parser.add_argument("--factory-calib", action="store_true", default=False,
                        help="Use RealSense factory color camera intrinsics from device EEPROM instead of custom calibrator")
    parser.add_argument("--bottle-radius", type=float, default=0.045,
                        help="Object radius in meters for center compensation along viewing ray (default: 0.045m = 9cm diameter)")
    parser.add_argument("--undistort", action="store_true",
                        help="Apply OpenCV lens undistortion using loaded calibration coefficients")
    parser.add_argument("--no-gui", action="store_true",
                        help="Run in headless/console mode without cv2.imshow")
    parser.add_argument("--frames", type=int, default=None,
                        help="Limit execution to N frames")
    args = parser.parse_args()

    run_realsense_demo(args)


if __name__ == "__main__":
    main()
