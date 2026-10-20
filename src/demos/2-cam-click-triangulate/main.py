#!/usr/bin/env python3
"""Two-Camera Interactive Multi-Point Click Triangulator with Realtime Feeds & Custom Calibrations.

Features:
  - Custom Calibrations (NPZ / JSON / YAML / Profile directory / Named profile):
      - Adheres to custom-calibrator-npz.md specifications.
      - Loads camera_matrix, dist_coeffs, image_size, rms_error, card_name, device_node, stream_type.
      - Automatically checks stream_type and resolves camera device node re-mapping.
      - Supports profile names (e.g. --calib0 econ-lap-0 --calib1 rl720) found across standard calibration paths.
  - Realtime Live Video Feeds (--realtime / --live):
      - Multi-threaded zero-latency V4L2 background grabbers (NonQueuingCameraReader).
      - Automatic V4L2 device node discovery matching camera card name and stream type.
      - Interactive live 10-30 FPS AprilTag detection, clicking, and 3D triangulation.
  - Multi-Point Interactive Workflow:
      - Point 1 ready immediately upon start.
      - Click Camera 1 (Top-Left) to set Cam1 pixel coordinates for the active point.
      - Click Camera 2 (Bottom-Left) to set Cam2 pixel coordinates for the active point.
      - Real-time 3D triangulation relative to the chosen AprilTag origin.
  - Keyboard Shortcuts:
      - 'n', 'a', Space, '+': Create new point.
      - 'p', '[', Left Arrow: Select PREVIOUS point.
      - ']', Tab, Right Arrow: Select NEXT point.
      - '1' - '9': Direct jump to point 1 to 9.
      - 'c': Clear clicks for current active point.
      - 'd', 'x', Delete: Delete current active point.
      - 's': Save all clicked points to JSON file immediately.
      - 'q', ESC: Quit application and auto-save all points to JSON.
  - Inspection Utilities:
      - --list-cams: Scan and print connected V4L2 camera cards and stream nodes, then exit.
      - --list-calibs: Scan and print all discovered custom calibration profiles, then exit.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import yaml
from pupil_apriltags import Detector

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("2cam-click-triangulator")

try:
    from common.markers import DEFAULT_MARKER_SIZES_M, DEFAULT_TAG_FAMILY, DEFAULT_TAG_SIZE_M
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from common.markers import DEFAULT_MARKER_SIZES_M, DEFAULT_TAG_FAMILY, DEFAULT_TAG_SIZE_M

# Standard calibration search directories
DEFAULT_CALIB_SEARCH_DIRS = [
    Path("calibrations"),
    Path("/home/mfk01/Projects/econ-systems/tools/calibrator/output"),
    Path.home() / ".local/share/econ-systems/calibrations",
    Path.home() / "Projects/econ-systems/demos/a2-01/calibrations",
]


def clean_card_name(card: str | None) -> str:
    """Removes USB bus/port suffixes for robust camera card matching."""
    if not card:
        return ""
    # Strip (usb-0000:00:14.0-5) or similar hardware location suffixes
    cleaned = re.sub(r"\s*\(usb-[0-9a-fA-F:.-]+\)", "", str(card))
    return cleaned.strip().lower()


def enumerate_v4l2_streams() -> list[dict]:
    """Enumerates all connected V4L2 device nodes, card names, and video capture capabilities."""
    streams = []
    v4l_dir = Path("/sys/class/video4linux")
    if not v4l_dir.is_dir():
        return streams

    for p in sorted(v4l_dir.iterdir(), key=lambda x: int(re.search(r"\d+", x.name).group()) if re.search(r"\d+", x.name) else 999):
        name_file = p / "name"
        if name_file.exists():
            card_name = name_file.read_text().strip()
            node = f"/dev/{p.name}"
            has_capture = False
            formats = []
            try:
                res = subprocess.run(["v4l2-ctl", "--device", node, "--list-formats"], capture_output=True, text=True, timeout=0.6)
                for line in res.stdout.splitlines():
                    if "Capture" in line or "[" in line:
                        formats.append(line.strip())
                if "Type: Video Capture" in res.stdout and any(fmt in res.stdout for fmt in ["YUYV", "MJPG", "UYVY", "RGB", "GREY", "Z16", "BA81"]):
                    has_capture = True
            except Exception:
                has_capture = True

            streams.append({
                "node": node,
                "card_name": card_name,
                "has_capture": has_capture,
                "formats": formats
            })
    return streams


@dataclass
class CameraCalibration:
    name: str
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]
    rms_error: float | None = None
    device_node: str | None = None
    card_name: str | None = None
    stream_type: str | None = None
    distortion_model: str = "standard"
    source_path: Path | None = None
    is_remapped: bool = False
    original_node: str | None = None

    @classmethod
    def load(cls, path_or_query: str | Path | None, default_name: str = "cam",
             expected_stream_type: str = "RGB / Color",
             search_dirs: Sequence[Path] | None = None) -> CameraCalibration:
        """Loads calibration from NPZ, JSON, YAML, directory, or named profile.
        Follows rules defined in custom-calibrator-npz.md."""
        if not path_or_query:
            return cls.fallback_fov(default_name)

        target_path = Path(path_or_query)
        effective_search_dirs = list(search_dirs or DEFAULT_CALIB_SEARCH_DIRS)

        # 1. Search candidate locations
        candidate_paths = [target_path, Path.cwd() / target_path]
        for sdir in effective_search_dirs:
            candidate_paths.append(sdir / target_path)
            candidate_paths.append(sdir / target_path.name)

        found_path: Path | None = None
        for cp in candidate_paths:
            if cp.exists():
                found_path = cp.resolve()
                break

        # If not found directly, search within subdirectories of search_dirs by name
        if not found_path:
            q_lower = str(path_or_query).lower()
            for sdir in effective_search_dirs:
                if sdir.is_dir():
                    for item in sdir.iterdir():
                        if item.name.lower() == q_lower or (item.is_dir() and item.name.lower() == q_lower):
                            found_path = item.resolve()
                            break
                if found_path:
                    break

        if not found_path:
            log.warning(f"Could not locate calibration profile for '{path_or_query}'. Falling back to default FOV.")
            return cls.fallback_fov(default_name)

        # 2. If path is directory, look for standard calibration outputs
        if found_path.is_dir():
            for fname in ["camera_calibration.npz", "camera_calibration.json", "camera_info.yaml"]:
                cand = found_path / fname
                if cand.is_file():
                    found_path = cand
                    break

        log.info(f"Loading custom calibration from: {found_path}")

        # 3. Parse by file extension
        suffix = found_path.suffix.lower()
        if suffix == ".npz":
            return cls._load_npz(found_path, default_name, expected_stream_type)
        elif suffix == ".json":
            return cls._load_json(found_path, default_name, expected_stream_type)
        elif suffix in (".yaml", ".yml"):
            return cls._load_yaml(found_path, default_name, expected_stream_type)
        else:
            raise ValueError(f"Unsupported calibration format: {found_path}")

    @classmethod
    def _load_npz(cls, path: Path, default_name: str, expected_stream_type: str) -> CameraCalibration:
        data = np.load(path, allow_pickle=True)
        K = np.asarray(data["camera_matrix"], dtype=np.float64)
        D = np.asarray(data["dist_coeffs"], dtype=np.float64)
        size = tuple(int(v) for v in data["image_size"])
        name = str(data.get("camera_name", path.parent.name if path.parent.name != "output" else default_name))
        node = str(data.get("device_node")) if "device_node" in data and str(data["device_node"]).strip() else None
        card = str(data.get("card_name")) if "card_name" in data and str(data["card_name"]).strip() else None
        stype = str(data.get("stream_type")) if "stream_type" in data and str(data["stream_type"]).strip() else None
        rms = float(data["rms_error"]) if "rms_error" in data else None
        model = str(data.get("distortion_model", "standard"))

        calib = cls(name=name, camera_matrix=K, dist_coeffs=D, image_size=size,
                    rms_error=rms, device_node=node, card_name=card, stream_type=stype,
                    distortion_model=model, source_path=path)
        cls._check_stream_type(calib, expected_stream_type)
        return calib

    @classmethod
    def _load_json(cls, path: Path, default_name: str, expected_stream_type: str) -> CameraCalibration:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        name = data.get("camera_name") or path.parent.name or default_name
        c_info = data.get("calibration_info", {})
        w = int(c_info.get("image_width", 1280))
        h = int(c_info.get("image_height", 720))
        rms = float(c_info["rms_reprojection_error_px"]) if "rms_reprojection_error_px" in c_info else None
        model = c_info.get("model", "standard")

        dev = data.get("device_info", {})
        cfg = data.get("config", {})
        card = dev.get("card_name") or cfg.get("camera_card")
        node = dev.get("device_node") or cfg.get("device_node")
        stype = dev.get("stream_type") or cfg.get("stream_type")

        mat = data.get("camera_matrix", {})
        if "matrix_3x3" in mat:
            K = np.array(mat["matrix_3x3"], dtype=np.float64)
        else:
            fx = float(mat.get("fx", 900.0))
            fy = float(mat.get("fy", 900.0))
            cx = float(mat.get("cx", w / 2.0))
            cy = float(mat.get("cy", h / 2.0))
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        D = np.array(data.get("distortion_coefficients", []), dtype=np.float64)

        calib = cls(name=name, camera_matrix=K, dist_coeffs=D, image_size=(w, h),
                    rms_error=rms, device_node=node, card_name=card, stream_type=stype,
                    distortion_model=model, source_path=path)
        cls._check_stream_type(calib, expected_stream_type)
        return calib

    @classmethod
    def _load_yaml(cls, path: Path, default_name: str, expected_stream_type: str) -> CameraCalibration:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        name = data.get("camera_name") or path.parent.name or default_name
        w = int(data.get("image_width", 1280))
        h = int(data.get("image_height", 720))
        k_data = data.get("camera_matrix", {}).get("data", [])
        K = np.array(k_data, dtype=np.float64).reshape((3, 3)) if len(k_data) == 9 else np.eye(3)
        d_data = data.get("distortion_coefficients", {}).get("data", [])
        D = np.array(d_data, dtype=np.float64)
        model = data.get("distortion_model", "plumb_bob")
        card = data.get("camera_card")
        node = data.get("device_node")
        stype = data.get("stream_type")

        calib = cls(name=name, camera_matrix=K, dist_coeffs=D, image_size=(w, h),
                    rms_error=None, device_node=node, card_name=card, stream_type=stype,
                    distortion_model=model, source_path=path)
        cls._check_stream_type(calib, expected_stream_type)
        return calib

    @classmethod
    def fallback_fov(cls, name: str, width: int = 1280, height: int = 720, hfov_deg: float = 60.0) -> CameraCalibration:
        fx = (width / 2.0) / np.tan(np.radians(hfov_deg / 2.0))
        K = np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        D = np.zeros(5, dtype=np.float64)
        return cls(name=name, camera_matrix=K, dist_coeffs=D, image_size=(width, height),
                   stream_type="RGB / Color (Fallback FOV)")

    def update_resolution(self, width: int, height: int, hfov_deg: float = 60.0):
        if self.image_size != (width, height):
            self.image_size = (width, height)
            fx = (width / 2.0) / np.tan(np.radians(hfov_deg / 2.0))
            self.camera_matrix = np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    def resolve_live_device_node(self, user_override: str | int | None = None) -> str:
        """Resolves active V4L2 device node matching card name & stream type.
        Implements camera node resolution logic from custom-calibrator-npz.md."""
        # 1. User Override Precedence
        if user_override is not None:
            log.info(f"Camera '{self.name}': Using explicit user override node -> '{user_override}'")
            return str(user_override)

        connected = enumerate_v4l2_streams()
        c_target_card = clean_card_name(self.card_name)

        # 2. Check exact recorded node match
        if self.device_node:
            for s in connected:
                if s["node"] == self.device_node and s["has_capture"]:
                    s_card_clean = clean_card_name(s["card_name"])
                    if not c_target_card or c_target_card in s_card_clean or s_card_clean in c_target_card:
                        log.info(f"Camera '{self.name}': Verified exact node match '{self.device_node}' ({s['card_name']})")
                        return self.device_node

        # 3. Fallback: Search connected streams for matching card_name + stream_type
        if c_target_card:
            for s in connected:
                if s["has_capture"]:
                    s_card_clean = clean_card_name(s["card_name"])
                    if c_target_card in s_card_clean or s_card_clean in c_target_card:
                        old_node = self.device_node or "None"
                        self.is_remapped = (old_node != s["node"])
                        self.original_node = old_node
                        self.device_node = s["node"]
                        if self.is_remapped:
                            log.warning(f"[CAMERA MOVED / RE-MAPPED] Camera '{self.name}' (Card: '{self.card_name}') "
                                        f"node changed from '{old_node}' -> '{s['node']}' ({s['card_name']})")
                        else:
                            log.info(f"Camera '{self.name}': Matched live device node '{s['node']}' ({s['card_name']})")
                        return s["node"]

        # 4. Fallback if not detected
        fallback = self.device_node or "/dev/video0"
        log.warning(f"Camera '{self.name}' (Card: '{self.card_name or 'Unknown'}') was not found in connected V4L2 devices. "
                    f"Falling back to '{fallback}'.")
        return fallback

    @staticmethod
    def _check_stream_type(calib: CameraCalibration, expected_stream_type: str | None = None) -> None:
        res_str = f" @ {calib.image_size[0]}x{calib.image_size[1]}" if calib.image_size else ""
        rms_str = f" (RMS: {calib.rms_error:.3f}px)" if calib.rms_error is not None else ""
        if calib.stream_type:
            log.info(f"Loaded calibration '{calib.name}' stream: '{calib.stream_type}'{res_str}{rms_str} (Card: {calib.card_name or 'Unknown'})")
            is_color_calib = any(c in calib.stream_type.lower() for c in ("color", "rgb", "bgr", "mjpeg", "yuyv"))
            is_expected_color = any(c in (expected_stream_type or "").lower() for c in ("color", "rgb", "bgr"))
            if is_expected_color and not is_color_calib:
                log.warning(f"Stream type mismatch for {calib.name}! Expected color stream, but got '{calib.stream_type}'.")
        else:
            log.info(f"Calibration '{calib.name}' does not specify 'stream_type'{res_str}{rms_str}.")


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
        if calib:
            node = calib.resolve_live_device_node()
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
    log.info(f"Camera '{cam_label}' stream opened at resolution: {actual_w}x{actual_h} on node: {node}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class MediaReader:
    """Unified reader that seamlessly supports static image files, video files, or live cameras."""

    def __init__(self, source: str | int | None, calib: CameraCalibration | None, default_index: int = 0, force_live: bool = False):
        self.source = source
        self.calib = calib
        self.is_static_image = False
        self.is_video_file = False
        self.static_frame: np.ndarray | None = None
        self.cap: cv2.VideoCapture | None = None
        self.grabber: NonQueuingCameraReader | None = None

        node_str = str(source).strip() if source is not None else ""
        img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}
        vid_exts = {".mp4", ".avi", ".mkv", ".mov", ".m4v"}

        path = Path(node_str) if node_str else None
        if not force_live and path and path.is_file() and path.suffix.lower() in img_exts:
            img = cv2.imread(str(path))
            if img is not None:
                self.is_static_image = True
                self.static_frame = img
                cam_name = calib.name if calib else path.stem
                log.info(f"Loaded static image source for '{cam_name}': {path} ({img.shape[1]}x{img.shape[0]})")
                if calib:
                    calib.update_resolution(img.shape[1], img.shape[0])
                return
            else:
                log.warning(f"Failed to read image file: {path}")

        if not force_live and path and path.is_file() and path.suffix.lower() in vid_exts:
            self.is_video_file = True
            self.cap = cv2.VideoCapture(str(path))
            cam_name = calib.name if calib else path.stem
            log.info(f"Opened video file source for '{cam_name}': {path}")
            return

        # Live realtime camera
        effective_node = calib.resolve_live_device_node(user_override=source) if calib else (source if source is not None else default_index)
        self.cap = open_camera_source(effective_node, calib)
        self.grabber = NonQueuingCameraReader(self.cap)

    def get_latest_frame(self) -> np.ndarray | None:
        if self.is_static_image:
            return None if self.static_frame is None else self.static_frame.copy()
        if self.is_video_file and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            return frame if ret else None
        if self.grabber:
            return self.grabber.get_latest_frame()
        return None

    def stop(self):
        if self.grabber:
            self.grabber.stop()
        elif self.cap:
            self.cap.release()


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


def select_best_common_tag(tags0: list[dict], tags1: list[dict], override_id: int | None = None) -> tuple[dict, dict] | tuple[None, None]:
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
    p_cam = R_c_w @ pt_w + t_c_w
    if p_cam[2] <= 0.05:
        return None
    u = (K[0, 0] * p_cam[0] / p_cam[2]) + K[0, 2]
    v = (K[1, 1] * p_cam[1] / p_cam[2]) + K[1, 2]
    return int(round(u)), int(round(v))


def draw_axis_measurement_lines(img: np.ndarray, pt_rel: np.ndarray, corner_offset: np.ndarray,
                                R_c_w: np.ndarray, t_c_w: np.ndarray, K: np.ndarray, label_prefix: str = ""):
    p_O = corner_offset
    p_X = corner_offset + np.array([pt_rel[0], 0.0, 0.0])
    p_XY = corner_offset + np.array([pt_rel[0], pt_rel[1], 0.0])
    p_XYZ = corner_offset + pt_rel

    px_O = project_point_3d(p_O, R_c_w, t_c_w, K)
    px_X = project_point_3d(p_X, R_c_w, t_c_w, K)
    px_XY = project_point_3d(p_XY, R_c_w, t_c_w, K)
    px_XYZ = project_point_3d(p_XYZ, R_c_w, t_c_w, K)

    # 1. Direct Euclidean Line
    if px_O and px_XYZ:
        dist_cm = float(np.linalg.norm(pt_rel)) * 100.0
        cv2.line(img, px_O, px_XYZ, (0, 255, 255), 2, cv2.LINE_AA)
        mid_x = (px_O[0] + px_XYZ[0]) // 2
        mid_y = (px_O[1] + px_XYZ[1]) // 2
        prefix = f"{label_prefix} " if label_prefix else ""
        cv2.putText(img, f"{prefix}d={dist_cm:.1f}cm", (mid_x + 8, mid_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2, cv2.LINE_AA)

    # 2. X-axis line (Red)
    if px_O and px_X:
        cv2.line(img, px_O, px_X, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(img, px_X, 4, (0, 0, 255), -1)

    # 3. Y-axis line (Green)
    if px_X and px_XY:
        cv2.line(img, px_X, px_XY, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(img, px_XY, 4, (0, 255, 0), -1)

    # 4. Z-axis line (Cyan)
    if px_XY and px_XYZ:
        cv2.line(img, px_XY, px_XYZ, (255, 200, 0), 2, cv2.LINE_AA)
        cv2.circle(img, px_XYZ, 4, (255, 200, 0), -1)


@dataclass
class PointAnnotation:
    id: int
    name: str
    pt0: tuple[float, float] | None = None
    pt1: tuple[float, float] | None = None
    result: dict | None = None


class MultiPointTracker:
    def __init__(self, panel_w: int, panel_h: int, orig_w0: int, orig_h0: int, orig_w1: int, orig_h1: int):
        self.panel_w = panel_w
        self.panel_h = panel_h
        self.orig_w0 = orig_w0
        self.orig_h0 = orig_h0
        self.orig_w1 = orig_w1
        self.orig_h1 = orig_h1

        self.points: list[PointAnnotation] = [PointAnnotation(id=1, name="Point 1")]
        self.active_idx: int = 0
        self.status_message: str = "Ready. Click Cam 1 & Cam 2 for Point 1."
        self.status_time: float = time.time()

    def update_actual_dimensions(self, w0: int, h0: int, w1: int, h1: int):
        self.orig_w0 = w0
        self.orig_h0 = h0
        self.orig_w1 = w1
        self.orig_h1 = h1

    @property
    def active_point(self) -> PointAnnotation:
        if not self.points:
            self.points.append(PointAnnotation(id=1, name="Point 1"))
            self.active_idx = 0
        return self.points[self.active_idx]

    def create_new_point(self) -> PointAnnotation:
        next_id = max((p.id for p in self.points), default=0) + 1
        new_pt = PointAnnotation(id=next_id, name=f"Point {next_id}")
        self.points.append(new_pt)
        self.active_idx = len(self.points) - 1
        self.notify(f"Created & Selected Point #{next_id}")
        log.info(f"Created new Point #{next_id} (Total: {len(self.points)})")
        return new_pt

    def select_prev_point(self):
        if not self.points:
            return
        self.active_idx = (self.active_idx - 1 + len(self.points)) % len(self.points)
        pt = self.active_point
        self.notify(f"Selected Point #{pt.id} ({self.active_idx + 1}/{len(self.points)})")
        log.info(f"Selected Point #{pt.id}")

    def select_next_point(self):
        if not self.points:
            return
        self.active_idx = (self.active_idx + 1) % len(self.points)
        pt = self.active_point
        self.notify(f"Selected Point #{pt.id} ({self.active_idx + 1}/{len(self.points)})")
        log.info(f"Selected Point #{pt.id}")

    def select_point_by_index(self, idx: int):
        if 0 <= idx < len(self.points):
            self.active_idx = idx
            self.notify(f"Jumped to Point #{self.points[idx].id}")

    def delete_current_point(self):
        if not self.points:
            return
        del_id = self.active_point.id
        if len(self.points) == 1:
            self.points[0].pt0 = None
            self.points[0].pt1 = None
            self.points[0].result = None
            self.notify(f"Cleared clicks for Point #{del_id}")
        else:
            self.points.pop(self.active_idx)
            self.active_idx = min(self.active_idx, len(self.points) - 1)
            self.notify(f"Deleted Point #{del_id}. Now on Point #{self.active_point.id}")
            log.info(f"Deleted Point #{del_id}")

    def clear_current_point(self):
        pt = self.active_point
        pt.pt0 = None
        pt.pt1 = None
        pt.result = None
        self.notify(f"Cleared clicks for Point #{pt.id}")
        log.info(f"Cleared clicks for Point #{pt.id}")

    def notify(self, msg: str):
        self.status_message = msg
        self.status_time = time.time()

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Camera 0 view (Top-Left)
            if 0 <= x < self.panel_w and 0 <= y < self.panel_h:
                orig_x = (x / float(self.panel_w)) * self.orig_w0
                orig_y = (y / float(self.panel_h)) * self.orig_h0
                pt = self.active_point
                pt.pt0 = (orig_x, orig_y)
                self.notify(f"Point #{pt.id}: Cam 1 clicked ({orig_x:.1f}, {orig_y:.1f})")
                log.info(f"Point #{pt.id}: Clicked Cam 1 ({orig_x:.1f}, {orig_y:.1f})")

            # Camera 1 view (Bottom-Left)
            elif 0 <= x < self.panel_w and self.panel_h <= y < self.panel_h * 2:
                local_y = y - self.panel_h
                orig_x = (x / float(self.panel_w)) * self.orig_w1
                orig_y = (local_y / float(self.panel_h)) * self.orig_h1
                pt = self.active_point
                pt.pt1 = (orig_x, orig_y)
                self.notify(f"Point #{pt.id}: Cam 2 clicked ({orig_x:.1f}, {orig_y:.1f})")
                log.info(f"Point #{pt.id}: Clicked Cam 2 ({orig_x:.1f}, {orig_y:.1f})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clear_current_point()


def export_points_json(output_path: Path, scene_name: str, points: list[PointAnnotation],
                       best_tag_info: dict | None, calib0: CameraCalibration, calib1: CameraCalibration) -> dict:
    completed_count = sum(1 for p in points if p.pt0 is not None and p.pt1 is not None)
    payload = {
        "scene": scene_name,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_points": len(points),
        "completed_points": completed_count,
        "reference_tag": {
            "id": best_tag_info["id"] if best_tag_info else None,
            "origin": "corner" if best_tag_info and best_tag_info.get("corner") else "center",
            "size_m": float(best_tag_info["t0"]["size"]) if best_tag_info else None,
        } if best_tag_info else None,
        "cameras": {
            "camera0": {
                "name": calib0.name,
                "card_name": calib0.card_name,
                "device_node": calib0.device_node,
                "stream_type": calib0.stream_type,
                "rms_error_px": calib0.rms_error,
                "resolution": list(calib0.image_size),
                "distortion_model": calib0.distortion_model,
            },
            "camera1": {
                "name": calib1.name,
                "card_name": calib1.card_name,
                "device_node": calib1.device_node,
                "stream_type": calib1.stream_type,
                "rms_error_px": calib1.rms_error,
                "resolution": list(calib1.image_size),
                "distortion_model": calib1.distortion_model,
            }
        },
        "points": []
    }

    for p in points:
        pt_entry = {
            "id": p.id,
            "name": p.name,
            "cam0_px": [round(float(p.pt0[0]), 2), round(float(p.pt0[1]), 2)] if p.pt0 else None,
            "cam1_px": [round(float(p.pt1[0]), 2), round(float(p.pt1[1]), 2)] if p.pt1 else None,
            "status": "complete" if (p.pt0 and p.pt1) else ("cam0_only" if p.pt0 else ("cam1_only" if p.pt1 else "empty")),
            "triangulated": None
        }
        if p.result:
            pt_entry["triangulated"] = {
                "x_cm": round(float(p.result["xyz_cm"][0]), 2),
                "y_cm": round(float(p.result["xyz_cm"][1]), 2),
                "z_cm": round(float(p.result["xyz_cm"][2]), 2),
                "euclidean_distance_cm": round(float(p.result["dist_cm"]), 2),
                "reproj_err_px": round(float(p.result["reproj_px"]), 2),
                "ray_skew_cm": round(float(p.result["ray_diff_cm"]), 2),
                "ray_angle_deg": round(float(p.result["ray_angle_deg"]), 1),
                "est_depth_err_cm": round(float(p.result["est_err_cm"]), 2),
                "reference_tag_id": int(p.result["tag_id"]) if "tag_id" in p.result else None,
            }
        payload["points"].append(pt_entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


def render_info_panel(width: int, height: int, best_tag_info: dict | None,
                      tracker: MultiPointTracker, fps: float,
                      calib0: CameraCalibration, calib1: CameraCalibration,
                      is_realtime: bool) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (24, 24, 28)

    # Header
    feed_badge = "LIVE REALTIME" if is_realtime else "STREAM"
    cv2.rectangle(panel, (0, 0), (width, 40), (40, 40, 48), -1)
    cv2.putText(panel, f"2-CAM 3D TRIANGULATOR [{feed_badge}] ({fps:.1f} FPS)", (14, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2, cv2.LINE_AA)

    y = 56
    # Section: Calibrations
    cv2.putText(panel, "CALIBRATION PROFILES & SENSORS", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1)
    y += 6
    cv2.line(panel, (14, y), (width - 14, y), (60, 60, 70), 1)
    y += 18

    # Cam 0 info
    rms0_str = f"RMS:{calib0.rms_error:.2f}px" if calib0.rms_error is not None else "FOV Fallback"
    remap0_str = " [RE-MAPPED]" if calib0.is_remapped else ""
    c0_line = f"C1: {calib0.name} ({calib0.image_size[0]}x{calib0.image_size[1]}) | {rms0_str} | Node: {calib0.device_node or '-'}{remap0_str}"
    cv2.putText(panel, c0_line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 200, 100) if calib0.is_remapped else (200, 230, 255), 1, cv2.LINE_AA)
    y += 16

    # Cam 1 info
    rms1_str = f"RMS:{calib1.rms_error:.2f}px" if calib1.rms_error is not None else "FOV Fallback"
    remap1_str = " [RE-MAPPED]" if calib1.is_remapped else ""
    c1_line = f"C2: {calib1.name} ({calib1.image_size[0]}x{calib1.image_size[1]}) | {rms1_str} | Node: {calib1.device_node or '-'}{remap1_str}"
    cv2.putText(panel, c1_line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 200, 100) if calib1.is_remapped else (200, 230, 255), 1, cv2.LINE_AA)
    y += 20

    # Section: Best AprilTag Origin
    cv2.putText(panel, "REFERENCE APRILTAG COORDINATES", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 50), 1)
    y += 6
    cv2.line(panel, (14, y), (width - 14, y), (60, 60, 70), 1)
    y += 18

    if best_tag_info:
        tid = best_tag_info["id"]
        t0 = best_tag_info["t0"]
        t1 = best_tag_info["t1"]
        d0_cm = float(np.linalg.norm(t0["t"])) * 100.0
        d1_cm = float(np.linalg.norm(t1["t"])) * 100.0
        fit0 = t0["pose_err"]
        txt1 = f"Tag #{tid:<2} (Auto Best) | Size: {t0['size']*100:.1f}cm | Origin: {'Corner' if best_tag_info['corner'] else 'Center'}"
        txt2 = f"Cam1 Dist: {d0_cm:5.1f}cm (fit:{fit0:.2f}) | Cam2 Dist: {d1_cm:5.1f}cm"
        cv2.putText(panel, txt1, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 240, 255), 1, cv2.LINE_AA)
        y += 16
        cv2.putText(panel, txt2, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 220, 240), 1, cv2.LINE_AA)
        y += 20
    else:
        cv2.putText(panel, "Looking for AprilTag visible in both cameras...", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 100, 255), 1)
        y += 20

    # Section: Active Selected Point
    active_pt = tracker.active_point
    total_pts = len(tracker.points)
    active_num = tracker.active_idx + 1
    cv2.putText(panel, f"ACTIVE POINT: Point #{active_pt.id} ({active_num}/{total_pts})", (14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 120), 1)
    y += 6
    cv2.line(panel, (14, y), (width - 14, y), (60, 60, 70), 1)
    y += 18

    st0 = f"({active_pt.pt0[0]:.1f}, {active_pt.pt0[1]:.1f})" if active_pt.pt0 else "[WAITING FOR CLICK]"
    st1 = f"({active_pt.pt1[0]:.1f}, {active_pt.pt1[1]:.1f})" if active_pt.pt1 else "[WAITING FOR CLICK]"
    c0_col = (100, 255, 150) if active_pt.pt0 else (140, 140, 150)
    c1_col = (100, 255, 150) if active_pt.pt1 else (140, 140, 150)

    cv2.putText(panel, f"Cam 1 (Top-Left):   {st0}", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, c0_col, 1)
    y += 16
    cv2.putText(panel, f"Cam 2 (Bot-Left):   {st1}", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, c1_col, 1)
    y += 20

    # Section: 3D Triangulation Result
    cv2.putText(panel, "3D POSITION & ERROR METRICS", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)
    y += 6
    cv2.line(panel, (14, y), (width - 14, y), (60, 60, 70), 1)
    y += 18

    if active_pt.result:
        x, y_pos, z = active_pt.result["xyz_cm"]
        dist = active_pt.result["dist_cm"]
        skew = active_pt.result["ray_diff_cm"]
        ang = active_pt.result["ray_angle_deg"]
        err = active_pt.result["est_err_cm"]
        reproj = active_pt.result["reproj_px"]

        box_color = (0, 255, 0) if err < 2.5 else ((0, 200, 255) if err < 5.0 else (0, 80, 255))
        res1 = f"3D Pos: X={x:+5.1f}cm, Y={y_pos:+5.1f}cm, Z={z:+5.1f}cm"
        res2 = f"Distance to Origin: {dist:5.1f} cm (Uncertainty: ±{err:4.1f} cm)"
        res3 = f"Skew: {skew:4.2f}cm | Angle: {ang:4.1f}° | Reproj: {reproj:.1f}px"

        cv2.putText(panel, res1, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        y += 18
        cv2.putText(panel, res2, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, box_color, 2)
        y += 18
        cv2.putText(panel, res3, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 220, 240), 1)
        y += 22
    else:
        cv2.putText(panel, "Click on both cameras to triangulate 3D point.", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 150), 1)
        y += 24

    # Action Notification Banner
    age = time.time() - tracker.status_time
    banner_color = (0, 255, 200) if age < 3.0 else (120, 160, 180)
    cv2.rectangle(panel, (12, y - 4), (width - 12, y + 18), (32, 38, 44), -1)
    cv2.putText(panel, f">> {tracker.status_message}", (16, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, banner_color, 1, cv2.LINE_AA)
    y += 28

    # Quick Keyboard Shortcut Guide
    cv2.putText(panel, "KEYS: [N] New Pt | [P]/[ [ ] Prev | [ ] ] Next | [1-9] Jump", (14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 200, 100), 1)
    y += 16
    cv2.putText(panel, "      [C] Clear  | [D] Del Pt  | [S] Save JSON | [Q] Exit", (14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)

    return panel


def render_history_panel(width: int, height: int, tracker: MultiPointTracker,
                         camera_poses: list[dict], output_path: Path) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (20, 20, 24)

    # Header
    cv2.rectangle(panel, (0, 0), (width, 40), (36, 36, 44), -1)
    cv2.putText(panel, "SCENE POINTS TABLE & JSON EXPORT", (14, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 2, cv2.LINE_AA)

    y = 56
    # Stereo camera poses
    if camera_poses:
        for cp in camera_poses:
            x, yp, z = cp["xyz_cm"]
            r, p, yw = cp["rpy_deg"]
            txt = f"{cp['name']} | [{x:+5.1f}, {yp:+5.1f}, {z:+5.1f}]cm | R:{r:+.0f}° P:{p:+.0f}° Y:{yw:+.0f}° | {cp.get('base','')}"
            cv2.putText(panel, txt, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 140, 255), 1, cv2.LINE_AA)
            y += 16
    else:
        cv2.putText(panel, "Stereo Baseline: Awaiting shared AprilTag pose...", (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 130), 1)
        y += 16

    y += 4
    cv2.line(panel, (14, y), (width - 14, y), (60, 60, 70), 1)
    y += 16

    # Table Header
    hdr = f"{'SEL':<3} | {'#':<3} | {'CAM1 (px)':<12} | {'CAM2 (px)':<12} | {'DIST':<7} | {'XYZ (cm)':<18} | {'ERR'}"
    cv2.putText(panel, hdr, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 190), 1)
    y += 6
    cv2.line(panel, (14, y), (width - 14, y), (50, 50, 60), 1)
    y += 16

    # List points
    total_pts = len(tracker.points)
    max_visible = 8
    start_idx = max(0, min(tracker.active_idx - max_visible // 2, total_pts - max_visible))
    end_idx = min(total_pts, start_idx + max_visible)

    for i in range(start_idx, end_idx):
        p = tracker.points[i]
        is_sel = (i == tracker.active_idx)
        sel_marker = ">" if is_sel else " "

        c1_str = f"{p.pt0[0]:.0f},{p.pt0[1]:.0f}" if p.pt0 else "-"
        c2_str = f"{p.pt1[0]:.0f},{p.pt1[1]:.0f}" if p.pt1 else "-"

        if p.result:
            dist_str = f"{p.result['dist_cm']:4.1f}cm"
            x, yp, z = p.result["xyz_cm"]
            xyz_str = f"[{x:+4.0f},{yp:+4.0f},{z:+4.0f}]"
            err = p.result["est_err_cm"]
            err_str = f"±{err:3.1f}cm"
            text_col = (0, 255, 120) if is_sel else (100, 220, 255)
        else:
            dist_str = "--"
            xyz_str = "--"
            err_str = "--"
            text_col = (0, 255, 255) if is_sel else (150, 150, 160)

        if is_sel:
            cv2.rectangle(panel, (12, y - 12), (width - 12, y + 4), (40, 50, 65), -1)

        row_txt = f"{sel_marker:<3} | {p.id:<3} | {c1_str:<12} | {c2_str:<12} | {dist_str:<7} | {xyz_str:<18} | {err_str}"
        cv2.putText(panel, row_txt, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, text_col, 1, cv2.LINE_AA)
        y += 18

    # Bottom status: JSON Export
    cv2.line(panel, (14, height - 32), (width - 14, height - 32), (60, 60, 70), 1)
    completed_pts = sum(1 for p in tracker.points if p.pt0 and p.pt1)
    status_txt = f"JSON Output: {output_path.name} | Points: {completed_pts}/{total_pts} Triangulated (Press 'S' to Save)"
    cv2.putText(panel, status_txt, (14, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 230, 255), 1, cv2.LINE_AA)

    return panel


def find_camera_file(directory: Path, prefix: str) -> str | None:
    patterns = [
        f"{prefix}_render*",
        f"{prefix}.*",
        f"{prefix}_*",
        f"{prefix}*",
    ]
    for pat in patterns:
        for p in sorted(directory.glob(pat)):
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".mp4", ".avi"}:
                if "annotated" not in p.name.lower():
                    return str(p)
    for p in sorted(directory.glob(f"{prefix}*")):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".mp4", ".avi"}:
            return str(p)
    return None


def resolve_scene_and_sources(scene_arg: str | None, cam0_arg: str | None, cam1_arg: str | None) -> tuple[str, str | None, str | None]:
    scene_name = "scene_a"
    resolved_cam0 = cam0_arg
    resolved_cam1 = cam1_arg

    if scene_arg:
        scene_path = Path(scene_arg)
        potential_dirs = [
            scene_path,
            Path("experiment/13_box_feature_point_matching") / scene_arg,
            Path("experiment/two_camera_rel_pose") / scene_arg,
        ]
        found_dir = None
        for d in potential_dirs:
            if d.is_dir():
                found_dir = d
                break

        if found_dir:
            scene_name = found_dir.name
            log.info(f"Resolved scene directory: {found_dir}")
            inputs_dir = found_dir / "inputs" if (found_dir / "inputs").is_dir() else found_dir

            c0 = find_camera_file(inputs_dir, "cam0") or find_camera_file(inputs_dir, "left")
            c1 = find_camera_file(inputs_dir, "cam1") or find_camera_file(inputs_dir, "right")

            if c0 and not resolved_cam0:
                resolved_cam0 = c0
            if c1 and not resolved_cam1:
                resolved_cam1 = c1
        else:
            scene_name = scene_path.stem if scene_path.suffix else scene_arg

    return scene_name, resolved_cam0, resolved_cam1


def list_available_calibrations(search_dirs: Sequence[Path]) -> list[dict]:
    profiles = []
    seen_names = set()
    for sdir in search_dirs:
        root = Path(sdir).expanduser().resolve()
        if not root.is_dir():
            continue
        for item in sorted(root.iterdir()):
            if not item.is_dir():
                continue
            npz = item / "camera_calibration.npz"
            if npz.is_file() and item.name not in seen_names:
                try:
                    data = np.load(npz, allow_pickle=True)
                    size = tuple(int(v) for v in data["image_size"])
                    rms = float(data["rms_error"]) if "rms_error" in data else None
                    card = str(data.get("card_name", ""))
                    stype = str(data.get("stream_type", ""))
                    node = str(data.get("device_node", ""))
                    profiles.append({
                        "name": item.name,
                        "path": str(npz),
                        "size": size,
                        "rms": rms,
                        "card": card,
                        "stype": stype,
                        "node": node
                    })
                    seen_names.add(item.name)
                except Exception:
                    pass
    return profiles


def print_camera_inspection():
    cams = enumerate_v4l2_streams()
    print("=" * 80)
    print("🎥 CONNECTED V4L2 VIDEO CAMERAS & STREAMS")
    print("=" * 80)
    if not cams:
        print("  No V4L2 camera devices detected on this system.")
    else:
        for c in cams:
            cap_badge = "[CAPTURE CAPABLE]" if c["has_capture"] else "[METADATA/OTHER]"
            print(f"  Node: {c['node']:<14} | Card: {c['card_name']} {cap_badge}")
            if c["formats"]:
                fmt_preview = ", ".join(c["formats"][:4])
                print(f"        Formats: {fmt_preview}")
    print("=" * 80)


def print_calib_inspection(search_dirs: Sequence[Path]):
    profs = list_available_calibrations(search_dirs)
    print("=" * 80)
    print("📐 DISCOVERED CUSTOM CAMERA CALIBRATION PROFILES")
    print("=" * 80)
    if not profs:
        print("  No calibration profiles found in search directories.")
    else:
        for p in profs:
            rms_str = f"RMS: {p['rms']:.3f} px" if p['rms'] is not None else "RMS: N/A"
            print(f"  Profile: {p['name']:<25} | {p['size'][0]}x{p['size'][1]} | {rms_str:<14} | {p['stype']:<15} | Card: {p['card']}")
    print("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="2-Camera Interactive Manual Point Triangulation with Realtime Feeds, Custom Calibrations & JSON Export",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Realtime & Live feed options
    parser.add_argument("--realtime", "--live", dest="realtime", action="store_true", default=False,
                        help="Force live realtime camera feeds using zero-latency V4L2 capture")
    parser.add_argument("--fps", type=float, default=10.0, help="Target FPS limiter rate")

    # Custom calibration options
    parser.add_argument("--calib0", default=None,
                        help="Camera 0 custom calibration: profile name (e.g. econ-lap-0, rl720) or path to .npz, .json, .yaml")
    parser.add_argument("--calib1", default=None,
                        help="Camera 1 custom calibration: profile name (e.g. astra720a, rl720) or path to .npz, .json, .yaml")
    parser.add_argument("--calib-dir", type=Path, default=None,
                        help="Additional directory to search for custom calibration profiles")
    parser.add_argument("--undistort", action="store_true", default=True,
                        help="Undistort camera frames using calibration dist_coeffs")

    # Camera & Scene inputs
    parser.add_argument("--scene", default="scene_a",
                        help="Scene name, scene directory, or dataset folder (default: scene_a)")
    parser.add_argument("--cam0", default=None,
                        help="Explicit Camera 0 device node/index, image file, or video file")
    parser.add_argument("--cam1", default=None,
                        help="Explicit Camera 1 device node/index, image file, or video file")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output JSON path to save all clicked points")

    # Inspection commands
    parser.add_argument("--list-cams", action="store_true",
                        help="Scan and list all connected V4L2 cameras and cards, then exit")
    parser.add_argument("--list-calibs", action="store_true",
                        help="Scan and list all available custom calibration profiles, then exit")

    # AprilTag settings
    parser.add_argument("--tag-id", type=int, default=None,
                        help="Manual reference Tag ID override (default: auto best tag)")
    parser.add_argument("--tag-size", type=float, default=0.0885,
                        help="Physical edge length of default AprilTag in meters")
    parser.add_argument("--marker-size", action="append",
                        help="Override marker size for specific ID, format: ID=METRES (e.g. 0=0.124)")
    parser.add_argument("--corner-origin", action="store_true", default=True,
                        help="Report relative positions from physical marker corner rather than center")
    parser.add_argument("--tag-family", default="tag36h11",
                        help="AprilTag dictionary family")

    args = parser.parse_args()

    # Search directories for calibrations
    search_dirs = list(DEFAULT_CALIB_SEARCH_DIRS)
    if args.calib_dir:
        search_dirs.insert(0, Path(args.calib_dir).expanduser().resolve())

    # Handle inspection flags
    if args.list_cams:
        print_camera_inspection()
        return 0

    if args.list_calibs:
        print_calib_inspection(search_dirs)
        return 0

    # 1. Load Custom Calibrations
    calib0 = CameraCalibration.load(args.calib0, default_name="cam0", expected_stream_type="RGB / Color", search_dirs=search_dirs)
    calib1 = CameraCalibration.load(args.calib1, default_name="cam1", expected_stream_type="RGB / Color", search_dirs=search_dirs)

    # 2. Resolve Scene and Sources
    scene_name, resolved_cam0, resolved_cam1 = resolve_scene_and_sources(args.scene, args.cam0, args.cam1)
    output_path = args.output if args.output is not None else Path(f"{scene_name}_points.json")

    # If --realtime is requested or if sources are not static files, resolve device nodes
    if args.realtime:
        resolved_cam0 = calib0.resolve_live_device_node(user_override=args.cam0)
        resolved_cam1 = calib1.resolve_live_device_node(user_override=args.cam1)
        log.info(f"REALTIME MODE ENABLED: Cam0 node -> {resolved_cam0} | Cam1 node -> {resolved_cam1}")

    log.info(f"Scene: '{scene_name}'")
    log.info(f"Cam0 source: {resolved_cam0 or 'default 0'} (Calibration: '{calib0.name}')")
    log.info(f"Cam1 source: {resolved_cam1 or 'default 1'} (Calibration: '{calib1.name}')")
    log.info(f"JSON export destination: {output_path.resolve()}")

    marker_sizes = parse_marker_sizes(args.marker_size, default_size=args.tag_size)
    detector = Detector(families=args.tag_family, nthreads=2, quad_decimate=1.0)

    reader0 = MediaReader(resolved_cam0, calib0, default_index=0, force_live=args.realtime)
    reader1 = MediaReader(resolved_cam1, calib1, default_index=1, force_live=args.realtime)

    panel_w, panel_h = 640, 360
    tracker = MultiPointTracker(panel_w, panel_h, calib0.image_size[0], calib0.image_size[1], calib1.image_size[0], calib1.image_size[1])

    window_name = f"2-Camera Multi-Point Click Triangulator - Scene: {scene_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, tracker.on_mouse)

    frame_interval = 1.0 / max(0.1, args.fps)
    last_process_time = 0.0
    fps_smooth = args.fps
    best_tag_info = None

    log.info("Starting Multi-Point Triangulation loop... Press 'n' for new point, '['/']' to cycle points, 's' to save JSON, 'q' to exit.")

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

            tracker.update_actual_dimensions(frame0.shape[1], frame0.shape[0], frame1.shape[1], frame1.shape[0])
            calib0.update_resolution(frame0.shape[1], frame0.shape[0])
            calib1.update_resolution(frame1.shape[1], frame1.shape[0])

            # Undistort using custom calibration coefficients if requested
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
            corner_offset = np.zeros(3)

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
                    "name": f"Cam0 ({calib0.name})",
                    "xyz_cm": [p_c0[0] * 100.0, p_c0[1] * 100.0, p_c0[2] * 100.0],
                    "rpy_deg": rotation_matrix_to_euler_deg(R_c0_t.T),
                    "base": f"Base: {baseline_m*100:4.1f}cm",
                })
                camera_poses.append({
                    "name": f"Cam1 ({calib1.name})",
                    "xyz_cm": [p_c1[0] * 100.0, p_c1[1] * 100.0, p_c1[2] * 100.0],
                    "rpy_deg": rotation_matrix_to_euler_deg(R_c1_t.T),
                    "base": f"Base: {baseline_m*100:4.1f}cm",
                })

                # Triangulate all points that have both cam0 and cam1 clicks
                for p in tracker.points:
                    if p.pt0 is not None and p.pt1 is not None:
                        pt_tag, reproj_px, ray_diff_cm, ray_angle_deg, est_depth_err_cm = triangulate_points(
                            p.pt0, p.pt1,
                            P0, P1, calib0.camera_matrix, calib1.camera_matrix,
                            R_c0_t, t_c0_t, R_c1_t, t_c1_t
                        )
                        pt_rel = pt_tag - corner_offset
                        dist_cm = float(np.linalg.norm(pt_rel)) * 100.0
                        p.result = {
                            "xyz_cm": [float(pt_rel[0] * 100.0), float(pt_rel[1] * 100.0), float(pt_rel[2] * 100.0)],
                            "dist_cm": dist_cm,
                            "ray_diff_cm": ray_diff_cm,
                            "ray_angle_deg": ray_angle_deg,
                            "est_err_cm": est_depth_err_cm,
                            "reproj_px": reproj_px,
                            "tag_id": int(best_tag0["id"]),
                        }
                    else:
                        p.result = None

            # Draw ALL points on Camera 1 View
            for idx, pt in enumerate(tracker.points):
                if pt.pt0 is not None:
                    cx0, cy0 = int(round(pt.pt0[0])), int(round(pt.pt0[1]))
                    is_active = (idx == tracker.active_idx)
                    if is_active:
                        cv2.circle(annotated0, (cx0, cy0), 12, (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.circle(annotated0, (cx0, cy0), 4, (0, 255, 0), -1, cv2.LINE_AA)
                        cv2.drawMarker(annotated0, (cx0, cy0), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
                        cv2.putText(annotated0, f"P{pt.id} [ACTIVE]", (cx0 + 14, max(20, cy0 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.circle(annotated0, (cx0, cy0), 7, (255, 200, 0), 2, cv2.LINE_AA)
                        cv2.circle(annotated0, (cx0, cy0), 3, (255, 200, 0), -1, cv2.LINE_AA)
                        cv2.putText(annotated0, f"P{pt.id}", (cx0 + 10, max(16, cy0 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)

            # Draw ALL points on Camera 2 View
            for idx, pt in enumerate(tracker.points):
                if pt.pt1 is not None:
                    cx1, cy1 = int(round(pt.pt1[0])), int(round(pt.pt1[1]))
                    is_active = (idx == tracker.active_idx)
                    if is_active:
                        cv2.circle(annotated1, (cx1, cy1), 12, (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.circle(annotated1, (cx1, cy1), 4, (0, 255, 0), -1, cv2.LINE_AA)
                        cv2.drawMarker(annotated1, (cx1, cy1), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
                        cv2.putText(annotated1, f"P{pt.id} [ACTIVE]", (cx1 + 14, max(20, cy1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.circle(annotated1, (cx1, cy1), 7, (255, 200, 0), 2, cv2.LINE_AA)
                        cv2.circle(annotated1, (cx1, cy1), 3, (255, 200, 0), -1, cv2.LINE_AA)
                        cv2.putText(annotated1, f"P{pt.id}", (cx1 + 10, max(16, cy1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)

            # Draw 3D measurement vector & axis lines for ACTIVE point
            active_pt = tracker.active_point
            if active_pt.result and best_tag0 and best_tag1:
                x, y_pos, z = active_pt.result["xyz_cm"]
                dist = active_pt.result["dist_cm"]
                err = active_pt.result["est_err_cm"]
                pt_rel_m = np.array([x / 100.0, y_pos / 100.0, z / 100.0])

                draw_axis_measurement_lines(annotated0, pt_rel_m, corner_offset,
                                            best_tag0["R"], best_tag0["t"], calib0.camera_matrix, f"P{active_pt.id}")
                draw_axis_measurement_lines(annotated1, pt_rel_m, corner_offset,
                                            best_tag1["R"], best_tag1["t"], calib1.camera_matrix, f"P{active_pt.id}")

                box_col = (0, 255, 0) if err < 2.5 else ((0, 200, 255) if err < 5.0 else (0, 80, 255))
                lbl_top = f"P{active_pt.id} 3D: [{x:+5.1f}, {y_pos:+5.1f}, {z:+5.1f}]cm (d={dist:.1f}cm ±{err:.1f}cm)"

                if active_pt.pt0 is not None:
                    cx0, cy0 = int(round(active_pt.pt0[0])), int(round(active_pt.pt0[1]))
                    cv2.putText(annotated0, lbl_top, (max(16, cx0 - 120), max(24, cy0 - 26)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, box_col, 2)

                if active_pt.pt1 is not None:
                    cx1, cy1 = int(round(active_pt.pt1[0])), int(round(active_pt.pt1[1]))
                    cv2.putText(annotated1, lbl_top, (max(16, cx1 - 120), max(24, cy1 - 26)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, box_col, 2)

            instant_fps = 1.0 / max(1e-4, time.time() - now)
            fps_smooth = 0.9 * fps_smooth + 0.1 * min(args.fps, instant_fps)

            # Titles on feeds
            cv2.putText(annotated0, f"Cam 1 ({calib0.name}) - Click P{active_pt.id}", (16, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(annotated1, f"Cam 2 ({calib1.name}) - Click P{active_pt.id}", (16, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # 2x2 Dashboard assembly
            cam1_view = cv2.resize(annotated0, (panel_w, panel_h))
            cam2_view = cv2.resize(annotated1, (panel_w, panel_h))

            info_panel = render_info_panel(panel_w, panel_h, best_tag_info, tracker, fps_smooth, calib0, calib1, args.realtime)
            history_panel = render_history_panel(panel_w, panel_h, tracker, camera_poses, output_path)

            top_row = np.hstack((cam1_view, info_panel))
            bot_row = np.hstack((cam2_view, history_panel))
            dashboard = np.vstack((top_row, bot_row))

            cv2.imshow(window_name, dashboard)

            # Key handling (support extended arrow keys)
            key = cv2.waitKeyEx(1)
            if key != -1:
                code = key & 0xFF
                if code in (ord('q'), ord('Q'), 27):
                    break
                elif code in (ord('n'), ord('N'), ord('a'), ord('A'), ord('+'), ord('='), 32):
                    tracker.create_new_point()
                elif code in (ord('p'), ord('P'), ord('[')) or key in (65361, 81, 2424832):
                    tracker.select_prev_point()
                elif code in (ord(']'), 9) or key in (65363, 83, 2555904):
                    tracker.select_next_point()
                elif code in (ord('c'), ord('C')):
                    tracker.clear_current_point()
                elif code in (ord('d'), ord('D'), ord('x'), ord('X'), 8, 127, 255) or key in (65535,):
                    tracker.delete_current_point()
                elif code in (ord('s'), ord('S')):
                    export_points_json(output_path, scene_name, tracker.points, best_tag_info, calib0, calib1)
                    tracker.notify(f"SAVED {len(tracker.points)} points to {output_path.name}!")
                    log.info(f"Explicit save: wrote {len(tracker.points)} points to {output_path.resolve()}")
                elif ord('1') <= code <= ord('9'):
                    target_idx = code - ord('1')
                    if target_idx < len(tracker.points):
                        tracker.select_point_by_index(target_idx)
                    else:
                        while len(tracker.points) <= target_idx:
                            tracker.create_new_point()
                        tracker.select_point_by_index(target_idx)

    finally:
        try:
            export_points_json(output_path, scene_name, tracker.points, best_tag_info, calib0, calib1)
            log.info(f"Auto-saved {len(tracker.points)} points to {output_path.resolve()} upon exit.")
        except Exception as e:
            log.error(f"Failed to auto-save points to JSON: {e}")

        reader0.stop()
        reader1.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
