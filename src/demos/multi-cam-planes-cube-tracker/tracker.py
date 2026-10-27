#!/usr/bin/env python3
"""Tracking layer for the Multi-Camera Planes & Cube Tracker.

- CameraCalibration: NPZ/JSON/YAML/named-profile loading + live V4L2 node resolution.
- VideoFeedSource / NonQueuingCameraReader: live webcam, video file or static image feeds for N cameras.
- Geometry: tag detection + PnP, Base-Frame extrinsics, RANSAC plane fit.
- MultiCameraPlanesCubePipeline: unified fusion of Base (0), Plane A (40,41),
  Plane B (42..45) and Cube (20..25) across arbitrary N cameras with auto-survey
  world map and real-time tracking loop.

Rendering lives in ``viz``; this module only calls into it.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
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

# Suppress low-level OpenCV console warnings
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

# Add src and demo dir to python path if needed
DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent.parent.parent
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(SRC_ROOT), str(DEMO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.markers import MarkerRegistry, DEFAULT_TAG_FAMILY
from common.calibrator import CubeConfig, estimate_cube_centroid_ransac

from viz import (
    annotate_camera_view,
    draw_base_frame_overlay,
    layout_camera_views,
    render_hud_dashboard,
    render_scene3d_panel,
    project_base_point,
    SessionLogger,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("multi-cam-planes-cube-tracker")

# Standard calibration search directories
DEFAULT_CALIB_SEARCH_DIRS = [
    REPO_ROOT / "calibrations",
    Path("/home/mfk01/Projects/econ-systems/tools/calibrator/output"),
    Path.home() / ".local/share/econ-systems/calibrations",
    REPO_ROOT / "calibrations" / "calibarotor0" / "outputs",
]

# Static reference tags for multi-tag bundle / auto-survey
STATIC_TAG_IDS = frozenset({0, 40, 41, 42, 43, 44, 45})
CUBE_TAG_IDS = frozenset(range(20, 32))

# Default Base Link transformation relative to Base Reference (Tag 0)
DEFAULT_BASE_LINK_ORIGIN_M = np.array([0.629164, 0.581773, 1.003308], dtype=np.float64)
DEFAULT_BASE_LINK_ROTATION = np.array([
    [ 0.648822, -0.458219,  0.607507],
    [ 0.462455,  0.871456,  0.163400],
    [-0.604289,  0.174927,  0.777326],
], dtype=np.float64)


def load_base_link_config(config_path: Path | str | None = None) -> tuple[np.ndarray, np.ndarray, bool]:
    """Loads base_link origin and rotation matrix relative to Tag 0 base frame."""
    origin = DEFAULT_BASE_LINK_ORIGIN_M.copy()
    R = DEFAULT_BASE_LINK_ROTATION.copy()
    auto_from_plane_a = False

    candidates = []
    if config_path:
        candidates.append(Path(config_path))
    candidates.extend([
        REPO_ROOT / "calibrations" / "base_link.yaml",
        REPO_ROOT / "calibrations" / "base_link.json",
    ])

    for p in candidates:
        p = Path(p).expanduser().resolve()
        if p.is_file():
            try:
                data = yaml.safe_load(p.read_text())
                bl = data.get("base_link", data)
                if "origin_tag0_m" in bl:
                    origin = np.array(bl["origin_tag0_m"], dtype=np.float64)
                elif "origin_m" in bl:
                    origin = np.array(bl["origin_m"], dtype=np.float64)
                elif "origin_base_m" in bl:
                    origin = np.array(bl["origin_base_m"], dtype=np.float64)

                if "rotation_matrix" in bl:
                    R = np.array(bl["rotation_matrix"], dtype=np.float64)
                elif "R_base_link_tag0" in bl:
                    R = np.array(bl["R_base_link_tag0"], dtype=np.float64)
                elif "R_custom_base" in bl:
                    R = np.array(bl["R_custom_base"], dtype=np.float64)

                if "auto_calculate_from_plane_a" in bl:
                    auto_from_plane_a = bool(bl["auto_calculate_from_plane_a"])

                log.info(f"Loaded base_link config from {p}: origin={origin.round(3).tolist()} auto_plane_a={auto_from_plane_a}")
                return origin, R, auto_from_plane_a
            except Exception as exc:
                log.warning(f"Could not load base_link config from {p}: {exc}. Using default constants.")
                break

    log.info(f"Using default base_link transformation constants: origin={origin.round(3).tolist()}")
    return origin, R, auto_from_plane_a


def clean_card_name(card: str | None) -> str:
    """Removes USB bus/port suffixes for robust camera card matching."""
    if not card:
        return ""
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
                cap = cv2.VideoCapture(node, cv2.CAP_V4L2)
                if cap.isOpened():
                    has_capture = True
                    cap.release()
            except Exception:
                pass
            streams.append({
                "node": node,
                "card_name": card_name,
                "has_capture": has_capture,
                "formats": formats,
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
    def load(
        cls,
        path_or_query: Path | str | None = None,
        default_name: str = "camera",
        search_dirs: Sequence[Path] | None = None,
        expected_stream_type: str = "color",
    ) -> CameraCalibration:
        """Loads calibration from NPZ, JSON, YAML, directory, or named profile."""
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

        if found_path.is_dir():
            for fname in ["camera_calibration.npz", "camera_calibration.json", "camera_info.yaml"]:
                cand = found_path / fname
                if cand.is_file():
                    found_path = cand
                    break

        log.info(f"Loading custom calibration from: {found_path}")

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

    def resolve_live_device_node(self, user_override: str | int | None = None) -> str:
        """Resolves active V4L2 device node matching card name & stream type."""
        if user_override is not None:
            log.info(f"Camera '{self.name}': Using explicit user override node -> '{user_override}'")
            return str(user_override)

        connected = enumerate_v4l2_streams()
        c_target_card = clean_card_name(self.card_name)

        if self.device_node:
            for s in connected:
                if s["node"] == self.device_node and s["has_capture"]:
                    s_card_clean = clean_card_name(s["card_name"])
                    if not c_target_card or c_target_card in s_card_clean or s_card_clean in c_target_card:
                        log.info(f"Camera '{self.name}': Verified exact node match '{self.device_node}' ({s['card_name']})")
                        return self.device_node

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
        else:
            log.info(f"Calibration '{calib.name}' does not specify 'stream_type'{res_str}{rms_str}.")


class NonQueuingCameraReader:
    """Spawns background grab thread that constantly drains V4L2 buffers,
    ensuring .get_latest_frame() always retrieves the zero-latency real-time frame."""

    def __init__(self, cap: cv2.VideoCapture, camera_name: str = "camera"):
        self.cap = cap
        self.camera_name = camera_name
        self.lock = threading.Lock()
        self.latest_frame: np.ndarray | None = None
        self.running = True
        self.consecutive_fails = 0
        self.error_msg: str | None = None
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
            except cv2.error:
                self.consecutive_fails += 1
                if self.consecutive_fails > 90:
                    self.error_msg = f"Camera '{self.camera_name}' fatal decode error."
                    break
                time.sleep(0.01)
                continue
            except Exception as exc:
                self.consecutive_fails += 1
                if self.consecutive_fails > 90:
                    self.error_msg = f"Camera '{self.camera_name}' read exception: {exc}"
                    break
                time.sleep(0.01)
                continue

            if not ret or frame is None:
                self.consecutive_fails += 1
                if self.consecutive_fails > 90:
                    self.error_msg = f"Camera '{self.camera_name}' stopped producing frames ({self.consecutive_fails} consecutive drops)."
                time.sleep(0.01)
                continue

            self.consecutive_fails = 0
            with self.lock:
                self.latest_frame = frame

    def get_latest_frame(self) -> np.ndarray | None:
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass


class VideoFeedSource:
    """Handles live webcam, video file, or static image feed with automatic node resolution."""

    def __init__(self, source_path_or_idx: str | int | np.ndarray | None, calib: CameraCalibration):
        self.calib = calib
        self.is_static_image = False
        self.static_frame: np.ndarray | None = None
        self.cap: cv2.VideoCapture | None = None
        self.grabber: NonQueuingCameraReader | None = None
        self.source_desc = ""
        self.error_msg: str | None = None

        # Check if direct numpy array passed (e.g. for offline testing / synthetic feeds)
        if isinstance(source_path_or_idx, np.ndarray):
            self.is_static_image = True
            self.static_frame = source_path_or_idx.copy()
            self.source_desc = f"Static Array ({self.static_frame.shape[1]}x{self.static_frame.shape[0]})"
            return

        # Check if mock / synthetic string
        if isinstance(source_path_or_idx, str) and source_path_or_idx.lower() in ("mock", "synthetic", "test", "dummy"):
            self.is_static_image = True
            w, h = calib.image_size if calib.image_size else (1280, 720)
            self.static_frame = np.zeros((h, w, 3), dtype=np.uint8)
            self.source_desc = f"Mock Canvas ({w}x{h})"
            return

        # Check if static image file
        if isinstance(source_path_or_idx, str) and Path(source_path_or_idx).is_file():
            s = str(source_path_or_idx).lower()
            if s.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                self.is_static_image = True
                self.static_frame = cv2.imread(str(source_path_or_idx))
                self.source_desc = f"Static Image ({source_path_or_idx})"
                if self.static_frame is None:
                    self.error_msg = f"Could not read static image file: {source_path_or_idx}"
                    log.error(f"[CAMERA ERROR] {self.error_msg}")
                    return
                log.info(f"Loaded static image for {calib.name}: {source_path_or_idx} ({self.static_frame.shape[1]}x{self.static_frame.shape[0]})")
                return

        live_node = calib.resolve_live_device_node(source_path_or_idx)

        backend = cv2.CAP_ANY
        if isinstance(live_node, int) or (isinstance(live_node, str) and live_node.isdigit()):
            target = int(live_node)
            backend = cv2.CAP_V4L2
        elif isinstance(live_node, str) and live_node.startswith("/dev/video"):
            target = live_node
            backend = cv2.CAP_V4L2
        else:
            target = live_node

        self.source_desc = f"{target} (backend: {'V4L2' if backend == cv2.CAP_V4L2 else 'ANY'})"

        try:
            self.cap = cv2.VideoCapture(target, backend) if backend != cv2.CAP_ANY else cv2.VideoCapture(target)
        except Exception as exc:
            self.error_msg = f"Exception opening device '{target}' for '{calib.name}': {exc}"
            log.error(f"[CAMERA OPEN ERROR] {self.error_msg}")
            return

        if not self.cap or not self.cap.isOpened():
            self.error_msg = f"Failed to open video capture device '{target}' for '{calib.name}'!"
            log.error(f"[CAMERA OPEN ERROR] {self.error_msg}")
            return

        req_w = 1920
        req_h = 1080
        if calib.image_size and calib.image_size[0] > 0 and calib.image_size[1] > 0:
            req_w, req_h = calib.image_size

        if backend == cv2.CAP_V4L2:
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            except Exception:
                pass

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        act_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        act_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if (act_w != req_w or act_h != req_h) and backend == cv2.CAP_V4L2:
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
                act_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                act_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            except Exception:
                pass

        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        test_frame = None
        for _ in range(10):
            ret, cand_frame = self.cap.read()
            if ret and cand_frame is not None:
                test_frame = cand_frame
                break
            time.sleep(0.05)

        if test_frame is None:
            self.error_msg = f"Camera '{calib.name}' on device '{target}' opened, but failed to retrieve frame!"
            log.error(f"[CAMERA STREAM ERROR] {self.error_msg}")
            self.release()
            return

        log.info(f"🎥 [CAMERA STREAM] '{calib.name}' on {target}: Streaming {act_w}x{act_h} @ {fps:.1f} FPS -> ACTIVE OK")

        if calib.image_size and (act_w != calib.image_size[0] or act_h != calib.image_size[1]):
            sx = act_w / float(calib.image_size[0])
            sy = act_h / float(calib.image_size[1])
            calib.camera_matrix[0, 0] *= sx
            calib.camera_matrix[0, 2] *= sx
            calib.camera_matrix[1, 1] *= sy
            calib.camera_matrix[1, 2] *= sy
            calib.image_size = (act_w, act_h)

        if backend == cv2.CAP_V4L2 or isinstance(target, int) or (isinstance(target, str) and target.startswith("/dev/video")):
            self.grabber = NonQueuingCameraReader(self.cap, camera_name=calib.name)
            self.grabber.latest_frame = test_frame

    def has_error(self) -> bool:
        if self.error_msg is not None:
            return True
        return bool(self.grabber and self.grabber.error_msg is not None)

    def get_error(self) -> str | None:
        if self.error_msg is not None:
            return self.error_msg
        return self.grabber.error_msg if self.grabber else None

    def get_frame(self) -> np.ndarray | None:
        if self.is_static_image:
            return None if self.static_frame is None else self.static_frame.copy()
        if self.grabber:
            return self.grabber.get_latest_frame()
        if self.cap:
            ret, frame = self.cap.read()
            return frame if ret else None
        return None

    def release(self):
        if self.grabber:
            self.grabber.stop()
        elif self.cap:
            try:
                self.cap.release()
            except Exception:
                pass


@dataclass
class DetectedTagPose:
    tag_id: int
    corners_px: np.ndarray       # (4, 2)
    center_px: np.ndarray        # (2,)
    R_c_t: np.ndarray            # (3, 3) Tag rotation in camera frame
    t_c_t: np.ndarray            # (3,) Tag translation in camera frame
    decision_margin: float


def detect_and_estimate_poses(
    gray: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    detector: Detector,
    registry: MarkerRegistry,
    active_ratio: float = 1.0,
) -> dict[int, DetectedTagPose]:
    """Detect tags and estimate 6D camera-relative poses."""
    raw_dets = detector.detect(gray)
    results = {}

    for d in raw_dets:
        tid = int(d.tag_id)
        nominal_size = registry.get_size(tid)
        eff_size = nominal_size * active_ratio
        h = eff_size / 2.0

        obj_pts = np.array([
            [-h,  h, 0.0],
            [ h,  h, 0.0],
            [ h, -h, 0.0],
            [-h, -h, 0.0]
        ], dtype=np.float64)

        img_pts = np.array(d.corners, dtype=np.float64)
        success, rvecs, tvecs, repr_errs = cv2.solvePnPGeneric(
            obj_pts,
            img_pts,
            K,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not success or len(rvecs) == 0:
            continue

        best_idx = 0
        best_err = 1e9
        for i, (r, t) in enumerate(zip(rvecs, tvecs)):
            if t[2, 0] > 0.01:
                err = float(repr_errs[i, 0]) if repr_errs is not None and len(repr_errs) > i else 0.0
                if err < best_err:
                    best_err = err
                    best_idx = i

        rvec = rvecs[best_idx]
        tvec = tvecs[best_idx]
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.ravel()

        results[tid] = DetectedTagPose(
            tag_id=tid,
            corners_px=img_pts,
            center_px=np.array(d.center, dtype=np.float64),
            R_c_t=R,
            t_c_t=t,
            decision_margin=float(getattr(d, "decision_margin", 0.0)),
        )

    return results


def solve_camera_extrinsics_from_base(
    base_pose: DetectedTagPose,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute camera pose (R_c_base, t_c_base) from Base Tag 0."""
    R_basis = np.diag([-1.0, 1.0, -1.0])
    R_c_base = base_pose.R_c_t @ R_basis
    t_c_base = base_pose.t_c_t.copy()
    return R_c_base, t_c_base


def project_tag_to_base(
    tag_pose: DetectedTagPose,
    R_c_base: np.ndarray,
    t_c_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform tag center and rotation into Base Tag 0 Frame."""
    R_b_c = R_c_base.T
    t_b_c = -R_b_c @ t_c_base

    tag_center_base = R_b_c @ tag_pose.t_c_t + t_b_c
    tag_R_base = R_b_c @ tag_pose.R_c_t
    return tag_center_base, tag_R_base


def fit_plane_ransac_points(
    points_m: np.ndarray,
    dist_thresh_m: float = 0.005,
    max_iter: int = 500,
) -> tuple[np.ndarray, float, float]:
    """Fit a 3D plane ax + by + cz + d = 0 with nz > 0."""
    if len(points_m) < 3:
        mean_z = float(np.mean(points_m[:, 2])) if len(points_m) > 0 else 0.0
        return np.array([0.0, 0.0, 1.0]), -mean_z, 0.0

    best_inliers = np.ones(len(points_m), dtype=bool)
    best_count = 0
    best_normal = np.array([0.0, 0.0, 1.0])
    best_d = 0.0

    n_pts = len(points_m)
    for _ in range(max_iter):
        idx = np.random.choice(n_pts, 3, replace=False)
        p1, p2, p3 = points_m[idx]

        v1 = p2 - p1
        v2 = p3 - p1
        norm = np.cross(v1, v2)
        norm_len = np.linalg.norm(norm)
        if norm_len < 1e-6:
            continue
        normal = norm / norm_len
        d = -float(np.dot(normal, p1))

        dists = np.abs(points_m @ normal + d)
        inliers = dists < dist_thresh_m
        cnt = int(np.sum(inliers))
        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers
            best_normal = normal
            best_d = d

    inlier_pts = points_m[best_inliers]
    if len(inlier_pts) >= 3:
        centroid = np.mean(inlier_pts, axis=0)
        _, _, vh = np.linalg.svd(inlier_pts - centroid)
        normal = vh[2, :]
        normal = normal / np.linalg.norm(normal)
        d = -float(np.dot(normal, centroid))
    else:
        normal = best_normal
        d = best_d

    if normal[2] < 0:
        normal = -normal
        d = -d

    residuals = np.abs(points_m @ normal + d)
    rmse_mm = float(np.sqrt(np.mean(residuals ** 2))) * 1000.0
    return normal, d, rmse_mm


class MultiCameraPlanesCubePipeline:
    """Full Tracking & Real-Time Visualization Pipeline supporting arbitrary N cameras."""

    def __init__(
        self,
        calibs: Sequence[CameraCalibration],
        sources: Sequence[str | int | None] | None = None,
        markers_yaml: Path | str | None = None,
        active_ratio: float = 1.0,
        base_link_config: Path | str | None = None,
        config: dict | str | Path | None = None,
    ):
        if not calibs:
            raise ValueError("MultiCameraPlanesCubePipeline requires at least one CameraCalibration profile.")

        self.calibs = list(calibs)
        self.num_cameras = len(self.calibs)

        if sources is None:
            self.sources = [None] * self.num_cameras
        else:
            self.sources = list(sources)
            if len(self.sources) < self.num_cameras:
                self.sources.extend([None] * (self.num_cameras - len(self.sources)))

        self.active_ratio = active_ratio

        # Load markers registry
        self.registry = MarkerRegistry(markers_yaml)

        # Load config.yaml if provided or auto-discover from demo directory
        self.config = {}
        if isinstance(config, (str, Path)):
            p = Path(config).expanduser().resolve()
            if p.is_file():
                import yaml
                try:
                    self.config = yaml.safe_load(p.read_text()) or {}
                    log.info(f"Loaded pipeline config from: {p}")
                except Exception as exc:
                    log.warning(f"Failed to parse config YAML {p}: {exc}")
        elif isinstance(config, dict):
            self.config = config
        else:
            default_cfg_p = Path(__file__).resolve().parent / "config.yaml"
            if default_cfg_p.is_file():
                import yaml
                try:
                    self.config = yaml.safe_load(default_cfg_p.read_text()) or {}
                    log.info(f"Auto-loaded pipeline config from: {default_cfg_p}")
                except Exception:
                    pass

        # Static reference tags for multi-tag bundle / auto-survey
        cfg_static = self.config.get("static_tags", {})
        if "all_static" in cfg_static:
            self.static_tag_ids = frozenset(cfg_static["all_static"])
        else:
            # Default: 0, 1, 4, 5, 54, 55 + platform tags 42..45, 50..53 + baseline 40, 41
            self.static_tag_ids = frozenset({0, 1, 4, 5, 40, 41, 42, 43, 44, 45, 50, 51, 52, 53, 54, 55})

        # Moveable cube configuration (5 cm cube, Tags 20..25)
        cube_cfg = self.config.get("moveable_cube", {})
        cube_tag_ids = cube_cfg.get("tag_ids", list(range(20, 26)))
        self.cube_tag_ids = frozenset(cube_tag_ids)
        self.cube_name = cube_cfg.get("name", "box")
        cube_side_length_m = cube_cfg.get("side_length_m", 0.050)
        face_normals_cfg = cube_cfg.get("face_normals", {})
        if face_normals_cfg:
            face_normals = {int(k): np.array(v, dtype=np.float64) for k, v in face_normals_cfg.items()}
        else:
            face_normals = {
                20: np.array([ 0.0,  0.0,  1.0]),  # Face 20: Top
                21: np.array([ 0.0,  0.0, -1.0]),  # Face 21: Bottom
                22: np.array([ 0.0,  1.0,  0.0]),  # Face 22: Front
                23: np.array([ 0.0, -1.0,  0.0]),  # Face 23: Back
                24: np.array([ 1.0,  0.0,  0.0]),  # Face 24: Right
                25: np.array([-1.0,  0.0,  0.0]),  # Face 25: Left
            }
        self.cube_config = CubeConfig(
            side_length_m=cube_side_length_m,
            face_normals=face_normals,
        )
        # Optional configured parent platform where moveable cube is located (e.g. "box", "box2", or None)
        self.cube_parent_platform = cube_cfg.get("parent_platform") or cube_cfg.get("platform")

        # Platform boxes configuration (box: 42..45, box2: 50..53)
        self.boxes_config = self.config.get("boxes", {
            "box": {
                "id": "platform_box",
                "name": "box",
                "type": "quad_plane",
                "tag_ids": [42, 43, 44, 45],
                "tag_size_m": 0.066,
                "description": "Source platform (tags 42..45, 66 mm)",
            },
            "box2": {
                "id": "box2",
                "name": "box2",
                "type": "quad_plane",
                "tag_ids": [50, 51, 52, 53],
                "tag_size_m": 0.070,
                "description": "Destination platform (tags 50..53, 70 mm)",
            },
        })

        # Initialize detector for each camera
        self.detectors = [
            Detector(families=DEFAULT_TAG_FAMILY, nthreads=2, quad_decimate=1.0)
            for _ in range(self.num_cameras)
        ]

        # Feeds
        self.feeds = [
            VideoFeedSource(src, calib)
            for src, calib in zip(self.sources, self.calibs)
        ]

        # State storage
        self.last_results: dict = {}
        self.lock = threading.Lock()

        # Custom Baseline Coordinate Frame Feature (Key 'H')
        self.custom_h_frame_active: bool = False
        self.custom_h_frame_info: dict | None = None

        # 3D scene panel visibility (key 'V')
        self.show_scene3d: bool = True

        # Multi-Tag PnP Auto-Survey World Map
        # World map: tag_id -> ndarray(4, 3) 3D corner positions in base frame
        self.world_map: dict[int, np.ndarray] = {}
        self.world_map_obs: dict[int, int] = {}

        # Pre-seed Tag 0 analytically (defines world origin)
        _h0 = self.registry.get_size(0) / 2.0
        _R_basis = np.diag([-1.0, 1.0, -1.0])
        _lc0 = np.array([
            [-_h0,  _h0, 0.0],
            [ _h0,  _h0, 0.0],
            [ _h0, -_h0, 0.0],
            [-_h0, -_h0, 0.0],
        ], dtype=np.float64)
        self.world_map[0] = (_R_basis @ _lc0.T).T
        self.world_map_obs[0] = 9999

        # Pre-seed all static anchor landmarks from most-scene.json / scene.json if available
        self._seed_static_landmarks_from_scene()

        # Base Link configuration and transformation constants
        self.base_link_origin_m, self.base_link_R, self.auto_base_link = load_base_link_config(base_link_config)

    def _seed_static_landmarks_from_scene(self) -> None:
        """Pre-seed static tag landmarks from most-scene.json into the multi-tag world map."""
        scene_candidates = [
            REPO_ROOT / "output-dobots" / "scenes" / "most-scene.json",
            REPO_ROOT / "outputs" / "scenes" / "most-scene.json",
            REPO_ROOT.parent.parent / "cr5a-moveit" / "DOBOT_6Axis_ROS2_V4_fresh" / "output-a201" / "scenes" / "scene.json",
        ]
        for sc_p in scene_candidates:
            if sc_p.is_file():
                try:
                    with open(sc_p, "r", encoding="utf-8") as f:
                        sc_data = json.load(f)
                    landmarks = sc_data.get("static_landmarks", {})
                    if "0" not in landmarks or "T_base_tag" not in landmarks["0"]:
                        continue

                    T_base_tag0 = np.array(landmarks["0"]["T_base_tag"])
                    R_base_tag0 = T_base_tag0[:3, :3]
                    t_base_tag0 = T_base_tag0[:3, 3]

                    R_basis = np.diag([-1.0, 1.0, -1.0])
                    R_custom = R_basis @ R_base_tag0.T

                    seeded_count = 0
                    for tid_str, lm in landmarks.items():
                        try:
                            tid = int(tid_str)
                        except ValueError:
                            continue
                        corners_base = lm.get("corners_base_m")
                        if not corners_base or len(corners_base) != 4:
                            continue

                        c_base = np.array(corners_base, dtype=np.float64)
                        c_map = (R_custom @ (c_base.T - t_base_tag0.reshape(3, 1))).T
                        self.world_map[tid] = c_map
                        self.world_map_obs[tid] = 9999
                        seeded_count += 1

                    log.info(f"Loaded {seeded_count} static anchor markers from {sc_p.name}: {sorted(self.world_map.keys())}")
                    break
                except Exception as exc:
                    log.warning(f"Could not seed static landmarks from {sc_p}: {exc}")

    def _survey_tags(
        self,
        dets: dict[int, DetectedTagPose],
        R_c_base: np.ndarray,
        t_c_base: np.ndarray,
    ) -> None:
        """Update world map with static tag positions observed in current frame."""
        R_b_c = R_c_base.T
        t_b_c = -R_b_c @ t_c_base

        for tid, pose in dets.items():
            if tid not in self.static_tag_ids or tid == 0:
                continue

            h = self.registry.get_size(tid) / 2.0
            lc = np.array([
                [-h,  h, 0.0],
                [ h,  h, 0.0],
                [ h, -h, 0.0],
                [-h, -h, 0.0],
            ], dtype=np.float64)

            c_cam = (pose.R_c_t @ lc.T + pose.t_c_t.reshape(3, 1)).T
            c_world = (R_b_c @ c_cam.T + t_b_c.reshape(3, 1)).T

            n = self.world_map_obs.get(tid, 0)
            if n == 0:
                self.world_map[tid] = c_world
            else:
                alpha = 1.0 / min(n + 1, 50)
                self.world_map[tid] = (1.0 - alpha) * self.world_map[tid] + alpha * c_world
            self.world_map_obs[tid] = n + 1

    def _multi_tag_pnp(
        self,
        dets: dict[int, DetectedTagPose],
        K: np.ndarray,
        dist: np.ndarray,
        init_R: np.ndarray | None = None,
        init_t: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Solve camera extrinsics (R_c_base, t_c_base) using all surveyed static tags."""
        obj_pts = []
        img_pts = []
        tags_used = []

        for tid, pose in dets.items():
            if tid in self.cube_tag_ids or tid not in self.world_map:
                continue
            if self.world_map_obs.get(tid, 0) < 1:
                continue
            obj_pts.append(self.world_map[tid])
            img_pts.append(pose.corners_px)
            tags_used.append(tid)

        if len(tags_used) < 2:
            return None

        obj = np.vstack(obj_pts).astype(np.float64)
        img = np.vstack(img_pts).astype(np.float64)

        try:
            if init_R is not None and init_t is not None:
                rvec_guess, _ = cv2.Rodrigues(init_R)
                tvec_guess = init_t.copy().reshape(3, 1).astype(np.float64)
                ok, rvec, tvec = cv2.solvePnP(
                    obj, img, K, dist,
                    rvec=rvec_guess, tvec=tvec_guess,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            else:
                ok, rvec, tvec = cv2.solvePnP(
                    obj, img, K, dist,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            if not ok:
                return None
            R, _ = cv2.Rodrigues(rvec)
            return R, tvec.ravel()
        except cv2.error:
            return None

    def _track_quad_plane(
        self,
        all_dets: list[dict[int, DetectedTagPose]],
        cam_extrinsics: dict[int, tuple[np.ndarray, np.ndarray]],
        tag_ids: Sequence[int],
    ) -> dict | None:
        """Estimate consensus plane normal, centroid, and coplanarity RMSE for quad platform tags."""
        raw_pts = []
        centers = {}
        normals = {}
        tag_corners = {}

        for tid in tag_ids:
            cand_centers = []
            cand_normals = []
            cand_corners = []
            half = self.registry.get_size(tid) / 2.0
            local_corners = np.array([
                [-half,  half, 0.0],
                [ half,  half, 0.0],
                [ half, -half, 0.0],
                [-half, -half, 0.0],
            ], dtype=np.float64)

            for c_id, dets in enumerate(all_dets):
                if c_id in cam_extrinsics and tid in dets:
                    R_c_base, t_c_base = cam_extrinsics[c_id]
                    c_base, R_t_base = project_tag_to_base(dets[tid], R_c_base, t_c_base)
                    cand_centers.append(c_base)
                    n_tag = R_t_base[:, 2].copy()
                    if n_tag[2] < 0:
                        n_tag = -n_tag
                    cand_normals.append(n_tag)
                    corners_base = c_base + (R_t_base @ local_corners.T).T
                    cand_corners.append(corners_base)

            if cand_centers:
                avg_c = np.mean(cand_centers, axis=0)
                centers[tid] = avg_c
                avg_n = np.mean(cand_normals, axis=0)
                normals[tid] = avg_n / max(1e-6, np.linalg.norm(avg_n))
                avg_corners = np.mean(cand_corners, axis=0)
                tag_corners[tid] = avg_corners
                raw_pts.append(avg_c)
                for pt in avg_corners:
                    raw_pts.append(pt)

        if not normals:
            return None

        normals_list = list(normals.values())
        med_n = np.median(normals_list, axis=0)
        med_n = med_n / max(1e-6, np.linalg.norm(med_n))
        inlier_normals = [
            n for n in normals_list
            if float(np.dot(n, med_n)) >= np.cos(np.radians(25.0))
        ]
        if inlier_normals:
            plane_normal = np.mean(inlier_normals, axis=0)
            plane_normal = plane_normal / max(1e-6, np.linalg.norm(plane_normal))
        else:
            plane_normal = med_n
        if plane_normal[2] < 0:
            plane_normal = -plane_normal

        plane_d = -float(np.median([np.dot(plane_normal, p) for p in raw_pts]))
        residuals = np.abs(np.array([np.dot(plane_normal, p) + plane_d for p in raw_pts]))
        rmse_mm = float(np.sqrt(np.mean(residuals ** 2))) * 1000.0

        proj_centers = {}
        for tid, pt in centers.items():
            pt_proj = pt - (np.dot(plane_normal, pt) + plane_d) * plane_normal
            proj_centers[tid] = pt_proj

        proj_corners = {}
        for tid, crn_arr in tag_corners.items():
            proj_crn = [pt - (np.dot(plane_normal, pt) + plane_d) * plane_normal for pt in crn_arr]
            proj_corners[tid] = np.mean(proj_crn, axis=0)

        center_plane = np.mean(list(proj_centers.values()), axis=0)
        return {
            "detected_tags": list(proj_centers.keys()),
            "normal": plane_normal.tolist(),
            "d_offset_m": plane_d,
            "center_mm": (center_plane * 1000.0).tolist(),
            "z_height_mm": float(center_plane[2] * 1000.0),
            "coplanarity_rmse_mm": rmse_mm,
            "centers_mm": {tid: (pt * 1000.0).tolist() for tid, pt in proj_centers.items()},
            "corners_mm": {tid: (pt * 1000.0).tolist() for tid, pt in proj_centers.items()},
        }

    def process_frames(self, frames: Sequence[np.ndarray]) -> dict:
        """Process frames from all cameras and compute fused geometric features."""
        num_cams = len(frames)
        all_dets: list[dict[int, DetectedTagPose]] = []

        for c_id, frame in enumerate(frames):
            calib = self.calibs[c_id]
            det = self.detectors[c_id]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            dets = detect_and_estimate_poses(
                gray, calib.camera_matrix, calib.dist_coeffs,
                det, self.registry, self.active_ratio
            )
            all_dets.append(dets)

        results: dict = {
            "timestamp": time.time(),
            "world_map_size": len(self.world_map),
            "world_map_tags": sorted(self.world_map.keys()),
            "cube_centroid": None,
            "plane_a": None,
            "plane_b": None,
            "boxes": {},
            "cam_tags": {c_id: list(all_dets[c_id].keys()) for c_id in range(num_cams)},
        }

        for c_id in range(num_cams):
            results[f"cam{c_id}_has_base"] = (0 in all_dets[c_id])
            results[f"cam{c_id}_tags"] = list(all_dets[c_id].keys())

        # 1. Camera Extrinsics: Bootstrap from Tag 0
        bootstrap_ext: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for c_id in range(num_cams):
            if 0 in all_dets[c_id]:
                bootstrap_ext[c_id] = solve_camera_extrinsics_from_base(all_dets[c_id][0])

        # 2. Survey static tags into world map using bootstrap extrinsics
        for c_id, ext in bootstrap_ext.items():
            self._survey_tags(all_dets[c_id], *ext)

        results["world_map_size"] = len(self.world_map)
        results["world_map_tags"] = sorted(self.world_map.keys())

        # 3. Multi-tag PnP for each camera (with fallback to bootstrap)
        cam_extrinsics: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        cam_pnp_methods: dict[int, str] = {}

        for c_id in range(num_cams):
            calib = self.calibs[c_id]
            init_R, init_t = bootstrap_ext.get(c_id, (None, None))
            pnp_res = self._multi_tag_pnp(
                all_dets[c_id], calib.camera_matrix, calib.dist_coeffs,
                init_R=init_R, init_t=init_t
            )
            if pnp_res is not None:
                cam_extrinsics[c_id] = pnp_res
                cam_pnp_methods[c_id] = "multi_tag_pnp"
                # If this camera wasn't in bootstrap_ext, survey its observed tags now
                if c_id not in bootstrap_ext:
                    self._survey_tags(all_dets[c_id], *pnp_res)
            elif c_id in bootstrap_ext:
                cam_extrinsics[c_id] = bootstrap_ext[c_id]
                cam_pnp_methods[c_id] = "bootstrap_tag0"

        if not cam_extrinsics:
            return results

        # Expose camera extrinsics
        results["cam_extrinsics"] = {
            c_id: {
                "R_c_base": R.tolist(),
                "t_c_base_m": t.tolist(),
                "position_base_mm": ((-R.T @ t) * 1000.0).tolist(),
                "method": cam_pnp_methods.get(c_id, "unknown"),
            }
            for c_id, (R, t) in cam_extrinsics.items()
        }

        # 4. Track Platform Boxes (box: 42..45, box2: 50..53, etc.)
        results["boxes"] = {}
        for box_name, box_def in self.boxes_config.items():
            if box_def.get("type") == "quad_plane":
                tag_ids = box_def.get("tag_ids", [])
                box_res = self._track_quad_plane(all_dets, cam_extrinsics, tag_ids)
                if box_res:
                    box_res["id"] = box_def.get("id", box_name)
                    box_res["name"] = box_name
                    results["boxes"][box_name] = box_res

        # Backward compatibility: plane_b points to "box" (or first detected quad plane)
        results["plane_b"] = results["boxes"].get("box") or results["boxes"].get("platform_box")
        if not results["plane_b"] and results["boxes"]:
            results["plane_b"] = next(iter(results["boxes"].values()))

        # 5. Track Cube Centroid (Tags 20..25 across all cameras)
        cube_observations = []
        for c_id, dets in enumerate(all_dets):
            if c_id not in cam_extrinsics:
                continue
            R_c_base, t_c_base = cam_extrinsics[c_id]
            for tid in self.cube_tag_ids:
                if tid in dets:
                    cube_observations.append((
                        tid,
                        dets[tid].R_c_t,
                        dets[tid].t_c_t,
                        R_c_base,
                        t_c_base
                    ))

        cube_face_corners: dict[int, list[np.ndarray]] = {}
        for tid, R_c_t, t_c_t, R_c_base, t_c_base in cube_observations:
            R_b_c = R_c_base.T
            t_b_c = -R_b_c @ t_c_base
            center_b = R_b_c @ t_c_t.ravel() + t_b_c
            R_b_t = R_b_c @ R_c_t
            half = self.registry.get_size(tid) / 2.0
            local_corners = np.array([
                [-half,  half, 0.0],
                [ half,  half, 0.0],
                [ half, -half, 0.0],
                [-half, -half, 0.0],
            ], dtype=np.float64)
            cube_face_corners.setdefault(tid, []).append(center_b + (R_b_t @ local_corners.T).T)

        if cube_face_corners:
            results["cube_faces_mm"] = {
                tid: (np.mean(np.stack(cl), axis=0) * 1000.0).tolist()
                for tid, cl in cube_face_corners.items()
            }

        if cube_observations:
            # Active supporting plane for cube:
            # Check configured parent_platform first, otherwise auto-detect closest platform
            supp_n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            supp_d = 0.0
            parent_plat_name = None

            # 1. Check if parent_platform explicitly configured and present in results["boxes"]
            if self.cube_parent_platform and self.cube_parent_platform in results.get("boxes", {}):
                cfg_box = results["boxes"][self.cube_parent_platform]
                supp_n = np.array(cfg_box["normal"], dtype=np.float64)
                supp_d = float(cfg_box["d_offset_m"])
                parent_plat_name = self.cube_parent_platform
            elif results.get("boxes"):
                # 2. Auto-detect closest platform based on preliminary XY centroid distance
                prelim_c = np.mean([
                    R_c_base.T @ t_c_t.ravel() - R_c_base.T @ t_c_base
                    for _, _, t_c_t, R_c_base, t_c_base in cube_observations
                ], axis=0)
                best_box = None
                best_dist = 999.0
                best_name = None
                for b_name, b_data in results["boxes"].items():
                    b_center = np.array(b_data["center_mm"]) * 0.001
                    dist_xy = float(np.linalg.norm(prelim_c[:2] - b_center[:2]))
                    if dist_xy < best_dist:
                        best_dist = dist_xy
                        best_box = b_data
                        best_name = b_name
                if best_box and best_dist < 0.35:
                    supp_n = np.array(best_box["normal"], dtype=np.float64)
                    supp_d = float(best_box["d_offset_m"])
                    parent_plat_name = best_name
            elif results.get("plane_b"):
                supp_n = np.array(results["plane_b"]["normal"], dtype=np.float64)
                supp_d = float(results["plane_b"]["d_offset_m"])
                parent_plat_name = "box"

            cube_size = self.cube_config.side_length_m
            half_L = cube_size / 2.0
            centroid_cands = []
            face_cands: dict[int, list[np.ndarray]] = {}

            for tid, R_c_t, t_c_t, R_c_base, t_c_base in cube_observations:
                R_b_c = R_c_base.T
                t_b_c = -R_b_c @ t_c_base
                center_b = R_b_c @ t_c_t.ravel() + t_b_c
                R_b_t = R_b_c @ R_c_t
                n_face = R_b_t[:, 2].copy()
                if np.linalg.norm(n_face) > 1e-6:
                    n_face = n_face / np.linalg.norm(n_face)

                dot_val = float(np.dot(n_face, supp_n))
                if dot_val > np.cos(np.radians(35.0)) or dot_val < -np.cos(np.radians(35.0)):
                    cand_c = center_b - (np.dot(supp_n, center_b) + supp_d - half_L) * supp_n
                else:
                    n_lat = n_face - np.dot(n_face, supp_n) * supp_n
                    n_lat_len = np.linalg.norm(n_lat)
                    n_lat = n_lat / n_lat_len if n_lat_len > 1e-6 else n_face
                    pt_h = center_b - (np.dot(supp_n, center_b) + supp_d - half_L) * supp_n
                    cand_c = pt_h - half_L * n_lat

                centroid_cands.append(cand_c)
                face_cands.setdefault(tid, []).append(cand_c)

            if centroid_cands:
                pts_cands = np.array(centroid_cands, dtype=np.float64)
                refined_centroid = np.median(pts_cands, axis=0)
                refined_centroid = refined_centroid - (np.dot(supp_n, refined_centroid) + supp_d - half_L) * supp_n
                spread_rmse = float(np.sqrt(np.mean(np.linalg.norm(pts_cands - refined_centroid, axis=1) ** 2))) * 1000.0

                results["cube_centroid"] = {
                    "xyz_mm": (refined_centroid * 1000.0).tolist(),
                    "visible_faces": sorted(face_cands.keys()),
                    "num_obs": len(centroid_cands),
                    "residual_spread_mm": spread_rmse,
                    "parent_platform": parent_plat_name,
                }

        # 6. Track Plane A: Tags 40, 41 (Line / Pair Plane across all cameras)
        plane_a_centers = {}
        for tid in (40, 41):
            candidates = []
            for c_id, dets in enumerate(all_dets):
                if c_id in cam_extrinsics and tid in dets:
                    R_c_base, t_c_base = cam_extrinsics[c_id]
                    c_base, _ = project_tag_to_base(dets[tid], R_c_base, t_c_base)
                    candidates.append(c_base)
            if candidates:
                plane_a_centers[tid] = np.mean(candidates, axis=0)

        if 40 not in plane_a_centers and 40 in self.world_map:
            plane_a_centers[40] = np.mean(self.world_map[40], axis=0)
        if 41 not in plane_a_centers and 41 in self.world_map:
            plane_a_centers[41] = np.mean(self.world_map[41], axis=0)

        if 40 in plane_a_centers and 41 in plane_a_centers:
            p40 = plane_a_centers[40]
            p41 = plane_a_centers[41]
            dist_mm = float(np.linalg.norm(p41 - p40)) * 1000.0
            midpoint = (p40 + p41) / 2.0
            results["plane_a"] = {
                "tag_40_mm": (p40 * 1000.0).tolist(),
                "tag_41_mm": (p41 * 1000.0).tolist(),
                "baseline_dist_mm": dist_mm,
                "center_mm": (midpoint * 1000.0).tolist(),
                "z_height_mm": float(midpoint[2] * 1000.0),
            }

        # 7. Coordinate System Transformation into base_link
        R_custom = self.base_link_R
        origin_custom = self.base_link_origin_m.copy()

        if self.auto_base_link and results["plane_a"]:
            p40 = np.array(results["plane_a"]["tag_40_mm"]) / 1000.0
            p41 = np.array(results["plane_a"]["tag_41_mm"]) / 1000.0
            midpoint = (p40 + p41) / 2.0
            dx = p41 - p40
            dx_len = np.linalg.norm(dx)
            if dx_len > 1e-6:
                ux = dx / dx_len
                cam0_dir = np.array([0.0, 0.0, 1.0])
                uy = np.cross(cam0_dir, ux)
                uy_len = np.linalg.norm(uy)
                if uy_len > 1e-6:
                    uy = uy / uy_len
                else:
                    uy = np.array([0.0, 1.0, 0.0])
                uz = np.cross(ux, uy)
                uz = uz / max(1e-6, np.linalg.norm(uz))
                R_raw = np.column_stack([ux, uy, uz])
                R_rot_y_90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
                R_custom = R_raw @ R_rot_y_90
                origin_custom = midpoint + R_custom @ np.array([0.02, 0.0, 0.05])

        R_L = R_custom.T
        t_L = -R_L @ origin_custom

        base_link_info = {
            "active": True,
            "origin_tag0_m": origin_custom.tolist(),
            "origin_in_tag0_base_mm": (origin_custom * 1000.0).tolist(),
            "rotation_matrix": R_custom.tolist(),
            "R_base_link_tag0": R_custom.tolist(),
            "origin_base_m": origin_custom.tolist(),
            "origin_base_mm": (origin_custom * 1000.0).tolist(),
            "R_custom_base": R_custom.tolist(),
        }

        # Transform Tag 0 into base_link
        tag0_base_m = np.zeros(3)
        tag0_link_m = R_L @ tag0_base_m + t_L
        results["tag0"] = {
            "xyz_mm": (tag0_link_m * 1000.0).tolist(),
            "origin_tag0_frame": [0.0, 0.0, 0.0],
        }

        # Transform Cube Centroid into base_link
        if results["cube_centroid"]:
            cube_base_m = np.array(results["cube_centroid"]["xyz_mm"]) / 1000.0
            cube_link_m = R_L @ cube_base_m + t_L
            results["cube_centroid"]["xyz_mm"] = (cube_link_m * 1000.0).tolist()
            base_link_info["cube_centroid_in_h_mm"] = (cube_link_m * 1000.0).tolist()

        # Transform Cube Faces into base_link
        if results.get("cube_faces_mm"):
            cube_faces_link = {}
            for tid, f_corners in results["cube_faces_mm"].items():
                corners_m = np.array(f_corners) / 1000.0
                corners_link = (R_L @ corners_m.T + t_L.reshape(3, 1)).T
                cube_faces_link[tid] = (corners_link * 1000.0).tolist()
            results["cube_faces_mm"] = cube_faces_link

        # Transform Plane A into base_link
        if results["plane_a"]:
            pa = results["plane_a"]
            p40_link = R_L @ (np.array(pa["tag_40_mm"]) / 1000.0) + t_L
            p41_link = R_L @ (np.array(pa["tag_41_mm"]) / 1000.0) + t_L
            mid_link = (p40_link + p41_link) / 2.0
            pa["tag_40_mm"] = (p40_link * 1000.0).tolist()
            pa["tag_41_mm"] = (p41_link * 1000.0).tolist()
            pa["center_mm"] = (mid_link * 1000.0).tolist()
            pa["z_height_mm"] = float(mid_link[2] * 1000.0)

        # Transform Boxes into base_link
        for box_name, pb in results.get("boxes", {}).items():
            c_base_m = np.array(pb["center_mm"]) / 1000.0
            c_link_m = R_L @ c_base_m + t_L
            n_base = np.array(pb["normal"], dtype=np.float64)
            n_link = R_L @ n_base
            pb["normal"] = n_link.tolist()
            pb["center_mm"] = (c_link_m * 1000.0).tolist()
            pb["z_height_mm"] = float(c_link_m[2] * 1000.0)
            pb["d_offset_m"] = -float(np.dot(n_link, c_link_m))

            if "centers_mm" in pb:
                pb["centers_mm"] = {
                    tid: ((R_L @ (np.array(pt) / 1000.0) + t_L) * 1000.0).tolist()
                    for tid, pt in pb["centers_mm"].items()
                }
            if "corners_mm" in pb:
                pb["corners_mm"] = {
                    tid: ((R_L @ (np.array(pt) / 1000.0) + t_L) * 1000.0).tolist()
                    for tid, pt in pb["corners_mm"].items()
                }

        # Keep plane_b in sync with transformed box
        results["plane_b"] = results["boxes"].get("box") or results["boxes"].get("platform_box")
        if not results["plane_b"] and results["boxes"]:
            results["plane_b"] = next(iter(results["boxes"].values()))

        # Transform Camera Extrinsics into base_link
        for c_id in list(cam_extrinsics.keys()):
            R_c_base, t_c_base = cam_extrinsics[c_id]
            R_c_link = R_c_base @ R_custom
            t_c_link = R_c_base @ origin_custom + t_c_base
            pos_cam_link = -R_c_link.T @ t_c_link
            cam_extrinsics[c_id] = (R_c_link, t_c_link)
            results["cam_extrinsics"][c_id] = {
                "R_c_base": R_c_link.tolist(),
                "t_c_base_m": t_c_link.tolist(),
                "position_base_mm": (pos_cam_link * 1000.0).tolist(),
                "method": results["cam_extrinsics"].get(c_id, {}).get("method", "unknown"),
                "R_c_tag0": R_c_base.tolist(),
                "t_c_tag0_m": t_c_base.tolist(),
            }

        results["base_link"] = base_link_info
        results["custom_h_frame"] = base_link_info
        self.custom_h_frame_info = base_link_info
        results["origin_frame"] = "base_link"

        return results

    def annotate_camera_view(self, frame: np.ndarray, dets: dict[int, DetectedTagPose],
                             cam_label: str, results: dict | None = None,
                             cam_id: int = 0) -> np.ndarray:
        calib = self.calibs[cam_id]
        return annotate_camera_view(frame, dets, cam_label, results, cam_id,
                                    calib.camera_matrix, calib.dist_coeffs)

    def draw_base_frame_overlay(self, vis: np.ndarray, results: dict, cam_id: int) -> None:
        calib = self.calibs[cam_id]
        draw_base_frame_overlay(vis, results, cam_id,
                                calib.camera_matrix, calib.dist_coeffs)

    def render_scene3d_panel(self, results: dict, width: int = 640, height: int = 720) -> np.ndarray:
        return render_scene3d_panel(results, self.calibs, width, height)

    def render_hud_dashboard(self, results: dict, width: int = 640, height: int = 720) -> np.ndarray:
        return render_hud_dashboard(results, self.custom_h_frame_active, self.num_cameras, width, height)

    def save_snapshot_bundle(
        self,
        raw_frames: list[np.ndarray | None],
        annotated_frames: list[np.ndarray | None],
        combined_ui: np.ndarray | None,
        results: dict,
        experiment_name: str = "default",
        session_root: Path | str = "outputs",
    ) -> Path:
        """Save raw frames, annotated views for all N cameras, composite UI, and telemetry JSON."""
        stamp = time.strftime("%Y%m%d_%H%M%S")
        root = Path(session_root)
        exp_dir = root / (experiment_name or "default")
        time_dir = exp_dir / stamp
        time_dir.mkdir(parents=True, exist_ok=True)

        for c_id, f in enumerate(raw_frames):
            if f is not None:
                cv2.imwrite(str(time_dir / f"cam{c_id}.png"), f)
                cv2.imwrite(str(exp_dir / f"cam{c_id}.png"), f)

        for c_id, v in enumerate(annotated_frames):
            if v is not None:
                cv2.imwrite(str(time_dir / f"cam{c_id}_annotated.png"), v)
                cv2.imwrite(str(exp_dir / f"cam{c_id}_annotated.png"), v)

        if combined_ui is not None:
            cv2.imwrite(str(time_dir / "combined_ui.png"), combined_ui)
            cv2.imwrite(str(exp_dir / "combined_ui.png"), combined_ui)

        with open(time_dir / "live_tracking_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        with open(exp_dir / "live_tracking_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        try:
            root_json = root / "live_tracking_results.json"
            with open(root_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        except Exception:
            pass

        log.info(f"[SNAPSHOT SAVED] {time_dir}/")
        print(f"\n[SNAPSHOT SAVED] {time_dir}/")
        for c_id in range(len(raw_frames)):
            print(f"  ├── cam{c_id}.png + cam{c_id}_annotated.png")
        print(f"  ├── combined_ui.png")
        print(f"  └── live_tracking_results.json")
        return time_dir

    def run(self, output_json: Path | None = None, display: bool = True,
            session: SessionLogger | None = None,
            experiment_name: str = "default",
            session_dir: str | Path = "outputs"):
        """Main real-time tracking loop across all N cameras."""
        if output_json:
            out_path = Path(output_json)
        elif session is not None:
            out_path = session.session_dir / "live_tracking_results.json"
        else:
            out_path = Path(session_dir) / (experiment_name or "default") / "live_tracking_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        last_ui: np.ndarray | None = None
        empty_frames_count = 0

        try:
            while True:
                # Check for camera feed hardware disconnects
                error_feed = None
                for c_id, feed in enumerate(self.feeds):
                    if feed.has_error():
                        error_feed = (c_id, feed.get_error())
                        break
                if error_feed:
                    log.error(f"[CAMERA STREAM ERROR] Camera {error_feed[0]}: {error_feed[1]}. Closing application.")
                    break

                frames = [feed.get_frame() for feed in self.feeds]

                if all(f is None for f in frames):
                    empty_frames_count += 1
                    if empty_frames_count > 60:
                        log.error("[CAMERA STREAM ERROR] All cameras stopped delivering frames. Closing application.")
                        break
                    time.sleep(0.02)
                    continue
                empty_frames_count = 0

                # Fallback blank frame for any camera that dropped a frame
                for c_id in range(self.num_cameras):
                    if frames[c_id] is None:
                        calib = self.calibs[c_id]
                        frames[c_id] = np.zeros((calib.image_size[1], calib.image_size[0], 3), dtype=np.uint8)

                # Process geometric features across all camera feeds
                results = self.process_frames(frames)
                with self.lock:
                    self.last_results = results

                # Annotate camera views
                annotated_views = []
                for c_id, frame in enumerate(frames):
                    calib = self.calibs[c_id]
                    det = self.detectors[c_id]
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                    dets = detect_and_estimate_poses(gray, calib.camera_matrix, calib.dist_coeffs,
                                                      det, self.registry, self.active_ratio)
                    v = self.annotate_camera_view(frame, dets, f"Camera {c_id} ({calib.name})", results, c_id)
                    annotated_views.append(v)

                # Layout camera views into fixed 640x720 panel
                cam_column = layout_camera_views(annotated_views, width=640, height=720)
                hud_panel = self.render_hud_dashboard(results, width=640, height=720)

                panels = [cam_column]
                if self.show_scene3d:
                    panels.append(self.render_scene3d_panel(results, width=640, height=720))
                panels.append(hud_panel)
                combined_ui = np.hstack(panels)
                last_ui = combined_ui

                if session is not None:
                    session.record(results, combined_ui)

                if display:
                    cv2.imshow("Multi-Camera AprilTag Multi-Feature Tracker", combined_ui)
                    key = cv2.waitKey(20) & 0xFF
                    if key in (27, ord('q'), ord('Q')):
                        log.info("Quit command received.")
                        break
                    elif key in (ord('h'), ord('H')):
                        self.custom_h_frame_active = not self.custom_h_frame_active
                        mode_str = "ENABLED" if self.custom_h_frame_active else "DISABLED"
                        log.info(f"Custom Baseline H Frame: {mode_str}")
                    elif key in (ord('v'), ord('V')):
                        self.show_scene3d = not self.show_scene3d
                        log.info(f"3D Scene Panel: {'ENABLED' if self.show_scene3d else 'DISABLED'}")
                    elif key in (ord('s'), ord('S')):
                        self.save_snapshot_bundle(
                            frames, annotated_views, combined_ui, results,
                            experiment_name=experiment_name,
                            session_root=session_dir,
                        )
                        if output_json:
                            with open(out_path, "w", encoding="utf-8") as f:
                                json.dump(results, f, indent=2)
                else:
                    if output_json:
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(results, f, indent=2)
                        log.info(f"Saved headless snapshot to: {out_path}")
                    self.save_snapshot_bundle(
                        frames, annotated_views, combined_ui, results,
                        experiment_name=experiment_name,
                        session_root=session_dir,
                    )
                    time.sleep(0.05)
                    break

        finally:
            if session is not None:
                session.close(final_frame=last_ui)
            for feed in self.feeds:
                feed.release()
            if display:
                cv2.destroyAllWindows()
