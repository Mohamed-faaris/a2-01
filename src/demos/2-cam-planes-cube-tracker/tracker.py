#!/usr/bin/env python3
"""Tracking layer for the Two-Camera Planes & Cube Tracker.

- CameraCalibration: NPZ/JSON/YAML/named-profile loading + live V4L2 node
  resolution (per custom-calibrator-npz.md).
- VideoFeedSource / NonQueuingCameraReader: live webcam, video file or
  static image feed.
- Geometry: tag detection + PnP, Base-Frame extrinsics, RANSAC plane fit.
- TwoCameraPlanesCubePipeline: fusion of Base (0), Plane A (40,41),
  Plane B (42..45) and Cube (20..25) plus the realtime tracking loop.

Rendering lives in ``viz``; this module only calls into it.
"""

from __future__ import annotations

import json
import logging
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

# Add src (and this demo dir) to python path if needed
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
    render_hud_dashboard,
    render_scene3d_panel,
    draw_base_frame_overlay,
    project_base_point,
    SessionLogger,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("2cam-planes-cube-tracker")

# Standard calibration search directories
DEFAULT_CALIB_SEARCH_DIRS = [
    REPO_ROOT / "calibrations",
    Path("/home/mfk01/Projects/econ-systems/tools/calibrator/output"),
    Path.home() / ".local/share/econ-systems/calibrations",
    REPO_ROOT / "calibrations" / "calibarotor0" / "outputs",
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
            except FileNotFoundError:
                has_capture = True
            except Exception:
                has_capture = False

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


class VideoFeedSource:
    """Handles live webcam, video file, or static image feed with automatic node resolution."""

    def __init__(self, source_path_or_idx: str | int | None, calib: CameraCalibration):
        self.calib = calib
        self.is_static_image = False
        self.static_frame: np.ndarray | None = None
        self.cap: cv2.VideoCapture | None = None
        self.grabber: NonQueuingCameraReader | None = None

        # Check if static image first
        if isinstance(source_path_or_idx, str) and Path(source_path_or_idx).is_file():
            s = str(source_path_or_idx).lower()
            if s.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                self.is_static_image = True
                self.static_frame = cv2.imread(str(source_path_or_idx))
                log.info(f"Loaded static image: {source_path_or_idx}")
                return

        # Resolve live device node per custom-calibrator-npz.md
        live_node = calib.resolve_live_device_node(source_path_or_idx)
        target = int(live_node) if str(live_node).isdigit() else live_node

        self.cap = cv2.VideoCapture(target)
        if not self.cap.isOpened():
            log.warning(f"Could not open video capture for: {target}")
            return

        w, h = calib.image_size
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        if isinstance(target, int) or (isinstance(target, str) and target.startswith("/dev/video")):
            self.grabber = NonQueuingCameraReader(self.cap)
            log.info(f"Started real-time non-queuing grabber for {target}")

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
            self.cap.release()


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

        # Object points centered on tag surface
        obj_pts = np.array([
            [-h, -h, 0.0],
            [ h, -h, 0.0],
            [ h,  h, 0.0],
            [-h,  h, 0.0]
        ], dtype=np.float64)

        img_pts = np.array(d.corners, dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            K,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not success:
            continue

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
    """Compute camera pose (R_c_base, t_c_base) from Base Tag 0.

    Pupil-apriltags / OpenCV solvePnP tag coordinate system:
      +X: Right, +Y: Down, +Z: Pointing normal out of tag.
    Base reference coordinate system:
      Aligned such that +Z is UP normal to tabletop, +X is Right, +Y is Forward.
      R_c_base = R_c_t @ diag(-1, 1, -1)
    """
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
        # Fallback to horizontal plane through points
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

    # Refine with all inliers using SVD
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

    # Ensure +Z points upwards
    if normal[2] < 0:
        normal = -normal
        d = -d

    residuals = np.abs(points_m @ normal + d)
    rmse_mm = float(np.sqrt(np.mean(residuals ** 2))) * 1000.0
    return normal, d, rmse_mm


class TwoCameraPlanesCubePipeline:
    """Full Tracking & Real-Time Visualization Pipeline."""

    def __init__(
        self,
        calib0: CameraCalibration,
        calib1: CameraCalibration,
        source0: str | int | None = None,
        source1: str | int | None = None,
        markers_yaml: Path | str | None = None,
        active_ratio: float = 1.0,
    ):
        self.calib0 = calib0
        self.calib1 = calib1
        self.active_ratio = active_ratio

        # Load markers registry
        self.registry = MarkerRegistry(markers_yaml)

        # Configure cube for tags 20..25 (41 mm each)
        cube_size = self.registry.get_size(20)
        self.cube_config = CubeConfig(
            side_length_m=cube_size,
            face_normals={
                20: np.array([ 0.0,  0.0,  1.0]),  # Face 20: Top
                21: np.array([ 0.0,  0.0, -1.0]),  # Face 21: Bottom
                22: np.array([ 0.0,  1.0,  0.0]),  # Face 22: Front
                23: np.array([ 0.0, -1.0,  0.0]),  # Face 23: Back
                24: np.array([ 1.0,  0.0,  0.0]),  # Face 24: Right
                25: np.array([-1.0,  0.0,  0.0]),  # Face 25: Left
            }
        )

        # Detectors
        self.detector0 = Detector(families=DEFAULT_TAG_FAMILY, nthreads=2, quad_decimate=1.0)
        self.detector1 = Detector(families=DEFAULT_TAG_FAMILY, nthreads=2, quad_decimate=1.0)

        # Feeds
        self.feed0 = VideoFeedSource(source0, calib0)
        self.feed1 = VideoFeedSource(source1, calib1)

        # State storage
        self.last_results: dict = {}
        self.lock = threading.Lock()

        # Custom Baseline Coordinate Frame Feature (Key 'H')
        # Baseline top -> X axis, Right -> Y axis, Z -> towards camera
        # Followed by rotation 90 deg via Y axis (Z facing up)
        # Shift Z +5 cm (+0.05m), X +2 cm (+0.02m)
        self.custom_h_frame_active: bool = False
        self.custom_h_frame_info: dict | None = None

        # 3D scene panel visibility (key 'V')
        self.show_scene3d: bool = True

    def process_frames(self, frame0: np.ndarray, frame1: np.ndarray) -> dict:
        """Process a pair of frames and compute all geometric features."""
        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY) if frame0.ndim == 3 else frame0
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if frame1.ndim == 3 else frame1

        # 1. Detect tags in both cameras
        dets0 = detect_and_estimate_poses(
            gray0, self.calib0.camera_matrix, self.calib0.dist_coeffs,
            self.detector0, self.registry, self.active_ratio
        )
        dets1 = detect_and_estimate_poses(
            gray1, self.calib1.camera_matrix, self.calib1.dist_coeffs,
            self.detector1, self.registry, self.active_ratio
        )

        results = {
            "timestamp": time.time(),
            "cam0_has_base": (0 in dets0),
            "cam1_has_base": (0 in dets1),
            "cam0_tags": list(dets0.keys()),
            "cam1_tags": list(dets1.keys()),
            "cube_centroid": None,
            "plane_a": None,
            "plane_b": None,
        }

        # 2. Camera Extrinsics relative to Base Tag 0
        cam_extrinsics: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if 0 in dets0:
            cam_extrinsics[0] = solve_camera_extrinsics_from_base(dets0[0])
        if 0 in dets1:
            cam_extrinsics[1] = solve_camera_extrinsics_from_base(dets1[0])

        # If base is only seen in one camera, we can still track everything visible to that camera!
        if not cam_extrinsics:
            return results

        # Expose camera extrinsics so the 3D overlays can re-project Base-Frame
        # points back into each camera image (used to visually verify arrays).
        results["cam_extrinsics"] = {
            c_id: {
                "R_c_base": R.tolist(),
                "t_c_base_m": t.tolist(),
                "position_base_mm": ((-R.T @ t) * 1000.0).tolist(),
            }
            for c_id, (R, t) in cam_extrinsics.items()
        }

        # 3. Track Cube Centroid (Tags 20..25)
        cube_observations = []
        for c_id, dets in [(0, dets0), (1, dets1)]:
            if c_id not in cam_extrinsics:
                continue
            R_c_base, t_c_base = cam_extrinsics[c_id]
            for tid in range(20, 26):
                if tid in dets:
                    cube_observations.append((
                        tid,
                        dets[tid].R_c_t,
                        dets[tid].t_c_t,
                        R_c_base,
                        t_c_base
                    ))

        # Reprojectable cube face quads (Base Frame, mm): the fused pose of each
        # cube face, drawn back into BOTH camera views so it can be compared
        # against the raw 2D detections.
        cube_face_corners: dict[int, list[np.ndarray]] = {}
        for tid, R_c_t, t_c_t, R_c_base, t_c_base in cube_observations:
            R_b_c = R_c_base.T
            t_b_c = -R_b_c @ t_c_base
            center_b = R_b_c @ t_c_t.ravel() + t_b_c
            R_b_t = R_b_c @ R_c_t
            half = self.registry.get_size(tid) / 2.0
            local_corners = np.array([
                [-half, -half, 0.0],
                [ half, -half, 0.0],
                [ half,  half, 0.0],
                [-half,  half, 0.0],
            ], dtype=np.float64)
            cube_face_corners.setdefault(tid, []).append(center_b + (R_b_t @ local_corners.T).T)

        if cube_face_corners:
            results["cube_faces_mm"] = {
                tid: (np.mean(np.stack(cl), axis=0) * 1000.0).tolist()
                for tid, cl in cube_face_corners.items()
            }

        if cube_observations:
            cube_res = estimate_cube_centroid_ransac(self.cube_config, cube_observations)
            if cube_res:
                results["cube_centroid"] = {
                    "xyz_mm": (cube_res.centroid_base_m * 1000.0).tolist(),
                    "visible_faces": cube_res.visible_faces,
                    "num_obs": cube_res.num_observations,
                    "residual_spread_mm": float(cube_res.residual_spread_mm),
                }

        # 4. Track Plane A: Tags 40, 41 (Line / Pair Plane)
        plane_a_pts = []
        plane_a_centers = {}
        for tid in (40, 41):
            candidates = []
            for c_id, dets in [(0, dets0), (1, dets1)]:
                if c_id in cam_extrinsics and tid in dets:
                    R_c_base, t_c_base = cam_extrinsics[c_id]
                    c_base, _ = project_tag_to_base(dets[tid], R_c_base, t_c_base)
                    candidates.append(c_base)
            if candidates:
                avg_pt = np.mean(candidates, axis=0)
                plane_a_pts.append(avg_pt)
                plane_a_centers[tid] = avg_pt

        if len(plane_a_pts) >= 2:
            p40 = plane_a_centers[40]
            p41 = plane_a_centers[41]
            dist_mm = float(np.linalg.norm(p41 - p40)) * 1000.0
            mid_pt = (p40 + p41) / 2.0
            results["plane_a"] = {
                "tag_40_mm": (p40 * 1000.0).tolist(),
                "tag_41_mm": (p41 * 1000.0).tolist(),
                "baseline_dist_mm": dist_mm,
                "center_mm": (mid_pt * 1000.0).tolist(),
                "z_height_mm": float(mid_pt[2] * 1000.0),
            }

            # -------------------------------------------------------------
            # Custom Baseline Reference Frame (Key 'H' Feature):
            # 1. Baseline top is X-axis: x_init = (p41 - p40) / ||p41 - p40|| (pointing along baseline)
            # 2. Z is facing camera: z_init = camera view vector towards baseline
            # 3. Right is Y-axis: y_init = cross(z_init, x_init)
            # 4. Rotate 90 degrees via Y-axis so Z faces UP
            # 5. Move Z +5 cm (+0.05m) and X +2 cm (+0.02m)
            # -------------------------------------------------------------
            v_baseline = p41 - p40
            norm_b = np.linalg.norm(v_baseline)
            if norm_b > 1e-6:
                x_init = v_baseline / norm_b

                # Estimate average camera optical center in Base Frame
                cam_centers = []
                for c_id in cam_extrinsics:
                    R_c_base, t_c_base = cam_extrinsics[c_id]
                    # Camera optical center in base frame: -R_c_base.T @ t_c_base
                    c_center = -R_c_base.T @ t_c_base
                    cam_centers.append(c_center)

                avg_cam_center = np.mean(cam_centers, axis=0) if cam_centers else np.array([0.0, -0.5, 0.5])
                # Vector from baseline midpoint towards camera (Z facing camera)
                cam_ray = avg_cam_center - mid_pt
                cam_ray_norm = np.linalg.norm(cam_ray)
                z_init = cam_ray / max(1e-6, cam_ray_norm)

                # Gram-Schmidt: Orthogonalize z_init against x_init
                z_init = z_init - np.dot(z_init, x_init) * x_init
                z_init = z_init / max(1e-6, np.linalg.norm(z_init))

                # Right is Y-axis: y_init = z_init x x_init
                y_init = np.cross(z_init, x_init)
                y_init = y_init / max(1e-6, np.linalg.norm(y_init))

                # Initial rotation matrix R_init = [x_init, y_init, z_init]
                R_init = np.column_stack([x_init, y_init, z_init])

                # Rotate +90 / -90 deg around Y so Z faces UP (+Z in base frame)
                R_rot_y_pos = np.array([
                    [ 0.0, 0.0, 1.0],
                    [ 0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0]
                ], dtype=np.float64)

                R_rot_y_neg = np.array([
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0,  0.0],
                    [1.0, 0.0,  0.0]
                ], dtype=np.float64)

                R_cand_pos = R_init @ R_rot_y_pos
                R_cand_neg = R_init @ R_rot_y_neg

                # Pick candidate where Z faces UP (+Z in base frame)
                if R_cand_pos[2, 2] >= R_cand_neg[2, 2]:
                    R_custom = R_cand_pos
                else:
                    R_custom = R_cand_neg

                # Translation: Move Z axis 5 cm (0.05m) and X axis 2 cm (0.02m)
                # in the newly rotated custom frame coordinates
                origin_custom = mid_pt + R_custom[:, 0] * 0.02 + R_custom[:, 2] * 0.05

                h_frame_info = {
                    "origin_base_m": origin_custom.tolist(),
                    "origin_base_mm": (origin_custom * 1000.0).tolist(),
                    "R_custom_base": R_custom.tolist(),
                    "x_axis": R_custom[:, 0].tolist(),
                    "y_axis": R_custom[:, 1].tolist(),
                    "z_axis": R_custom[:, 2].tolist(),
                }

                # If Cube is tracked, express cube centroid in this custom H frame!
                if results["cube_centroid"]:
                    cube_pt = np.array(results["cube_centroid"]["xyz_mm"]) / 1000.0
                    cube_in_h = R_custom.T @ (cube_pt - origin_custom)
                    h_frame_info["cube_centroid_in_h_mm"] = (cube_in_h * 1000.0).tolist()

                results["custom_h_frame"] = h_frame_info
                self.custom_h_frame_info = h_frame_info

        # 5. Track Plane B: Tags 42, 43, 44, 45 (Platform Quad Plane)
        plane_b_pts = []
        plane_b_centers = {}
        plane_b_normals = []
        for tid in (42, 43, 44, 45):
            candidates = []
            cand_normals = []
            for c_id, dets in [(0, dets0), (1, dets1)]:
                if c_id in cam_extrinsics and tid in dets:
                    R_c_base, t_c_base = cam_extrinsics[c_id]
                    c_base, R_t_base = project_tag_to_base(dets[tid], R_c_base, t_c_base)
                    candidates.append(c_base)
                    # Tag local Z axis is normal to the tag plane
                    n_tag = R_t_base[:, 2].copy()
                    if n_tag[2] < 0:
                        n_tag = -n_tag
                    cand_normals.append(n_tag)
            if candidates:
                avg_pt = np.mean(candidates, axis=0)
                plane_b_pts.append(avg_pt)
                plane_b_centers[tid] = avg_pt
                if cand_normals:
                    avg_n = np.mean(cand_normals, axis=0)
                    norm_len = np.linalg.norm(avg_n)
                    if norm_len > 1e-6:
                        plane_b_normals.append(avg_n / norm_len)

        if len(plane_b_pts) >= 3:
            pts_arr = np.array(plane_b_pts, dtype=np.float64)
            normal, d, rmse_mm = fit_plane_ransac_points(pts_arr)
            center = np.mean(pts_arr, axis=0)
            results["plane_b"] = {
                "detected_tags": list(plane_b_centers.keys()),
                "normal": normal.tolist(),
                "center_mm": (center * 1000.0).tolist(),
                "z_height_mm": float(center[2] * 1000.0),
                "coplanarity_rmse_mm": rmse_mm,
                "centers_mm": {tid: (pt * 1000.0).tolist() for tid, pt in plane_b_centers.items()},
                "corners_mm": {tid: (pt * 1000.0).tolist() for tid, pt in plane_b_centers.items()},
            }
        elif len(plane_b_pts) > 0:
            pts_arr = np.array(plane_b_pts, dtype=np.float64)
            center = np.mean(pts_arr, axis=0)
            if plane_b_normals:
                normal = np.mean(plane_b_normals, axis=0)
                normal = normal / max(1e-6, np.linalg.norm(normal))
            else:
                normal = np.array([0.0, 0.0, 1.0])
            results["plane_b"] = {
                "detected_tags": list(plane_b_centers.keys()),
                "normal": normal.tolist(),
                "center_mm": (center * 1000.0).tolist(),
                "z_height_mm": float(center[2] * 1000.0),
                "coplanarity_rmse_mm": 0.0,
                "centers_mm": {tid: (pt * 1000.0).tolist() for tid, pt in plane_b_centers.items()},
                "corners_mm": {tid: (pt * 1000.0).tolist() for tid, pt in plane_b_centers.items()},
            }

        return results

    # ------------------------------------------------------------------
    # Thin rendering wrappers (delegates to `viz` with this pipeline's
    # calibrations; kept so existing callers/tests keep working).
    # ------------------------------------------------------------------

    def annotate_camera_view(self, frame: np.ndarray, dets: dict[int, DetectedTagPose],
                             cam_label: str, results: dict | None = None,
                             cam_id: int = 0) -> np.ndarray:
        """Annotate tag polygons, IDs and the re-projected 3D overlay."""
        calib = self.calib0 if cam_id == 0 else self.calib1
        return annotate_camera_view(frame, dets, cam_label, results, cam_id,
                                    calib.camera_matrix, calib.dist_coeffs)

    def draw_base_frame_overlay(self, vis: np.ndarray, results: dict, cam_id: int) -> None:
        """Re-project fused metric results into one camera view (in-place)."""
        calib = self.calib0 if cam_id == 0 else self.calib1
        draw_base_frame_overlay(vis, results, cam_id,
                                calib.camera_matrix, calib.dist_coeffs)

    def render_scene3d_panel(self, results: dict, width: int = 640, height: int = 720) -> np.ndarray:
        """Software-rendered 3D perspective view of the Base Frame scene."""
        return render_scene3d_panel(results, self.calib0, self.calib1, width, height)

    def render_hud_dashboard(self, results: dict, width: int = 640, height: int = 720) -> np.ndarray:
        """Render the telemetry dashboard (H-frame status from this pipeline)."""
        return render_hud_dashboard(results, self.custom_h_frame_active, width, height)

    def run(self, output_json: Path | None = None, display: bool = True,
            session: SessionLogger | None = None):
        """Main real-time tracking loop."""
        log.info("Starting Two-Camera Planes & Cube Tracker Demo. "
                 "Keys: 'q' quit, 'h' custom baseline frame, 'v' 3D scene view, 's' export JSON.")
        out_path = output_json or (REPO_ROOT / "outputs" / "live_tracking_results.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        last_ui: np.ndarray | None = None
        try:
            while True:
                f0 = self.feed0.get_frame()
                f1 = self.feed1.get_frame()

                if f0 is None and f1 is None:
                    time.sleep(0.02)
                    continue

                # Fallback blank frame if one camera dropped
                if f0 is None:
                    f0 = np.zeros((self.calib0.image_size[1], self.calib0.image_size[0], 3), dtype=np.uint8)
                if f1 is None:
                    f1 = np.zeros((self.calib1.image_size[1], self.calib1.image_size[0], 3), dtype=np.uint8)

                # Process geometric features
                results = self.process_frames(f0, f1)
                with self.lock:
                    self.last_results = results

                # Annotate camera views
                gray0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY) if f0.ndim == 3 else f0
                gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY) if f1.ndim == 3 else f1
                dets0 = detect_and_estimate_poses(gray0, self.calib0.camera_matrix, self.calib0.dist_coeffs,
                                                  self.detector0, self.registry, self.active_ratio)
                dets1 = detect_and_estimate_poses(gray1, self.calib1.camera_matrix, self.calib1.dist_coeffs,
                                                  self.detector1, self.registry, self.active_ratio)

                v0 = self.annotate_camera_view(f0, dets0, f"Camera 0 ({self.calib0.name})", results, 0)
                v1 = self.annotate_camera_view(f1, dets1, f"Camera 1 ({self.calib1.name})", results, 1)

                # Resize to standard stacked view
                target_w = 640
                target_h = 360
                v0_thumb = cv2.resize(v0, (target_w, target_h))
                v1_thumb = cv2.resize(v1, (target_w, target_h))

                cam_column = np.vstack([v0_thumb, v1_thumb])  # 640x720
                hud_panel = self.render_hud_dashboard(results, width=640, height=720)

                panels = [cam_column]
                if self.show_scene3d:
                    panels.append(self.render_scene3d_panel(results, width=640, height=720))
                panels.append(hud_panel)
                combined_ui = np.hstack(panels)  # 1280x720 or 1920x720
                last_ui = combined_ui

                # Time-series history log + periodic annotated-frame snapshots
                if session is not None:
                    session.record(results, combined_ui)

                if display:
                    cv2.imshow("Two-Camera AprilTag Multi-Feature Tracker", combined_ui)
                    key = cv2.waitKey(20) & 0xFF
                    if key in (27, ord('q'), ord('Q')):
                        log.info("Quit command received.")
                        break
                    elif key in (ord('h'), ord('H')):
                        self.custom_h_frame_active = not self.custom_h_frame_active
                        mode_str = "ENABLED" if self.custom_h_frame_active else "DISABLED"
                        log.info(f"Custom Baseline H Frame: {mode_str}")
                        if self.custom_h_frame_info:
                            log.info(f"Custom Frame Origin: {self.custom_h_frame_info['origin_base_mm']} mm")
                            if "cube_centroid_in_h_mm" in self.custom_h_frame_info:
                                log.info(f"Cube in Custom Frame: {self.custom_h_frame_info['cube_centroid_in_h_mm']} mm")
                    elif key in (ord('v'), ord('V')):
                        self.show_scene3d = not self.show_scene3d
                        log.info(f"3D Scene Panel: {'ENABLED' if self.show_scene3d else 'DISABLED'}")
                    elif key in (ord('s'), ord('S')):
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(results, f, indent=2)
                        log.info(f"Saved snapshot to: {out_path}")
                else:
                    # Non-GUI step: save output snapshot if requested
                    if output_json:
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(results, f, indent=2)
                        log.info(f"Saved headless snapshot to: {out_path}")
                    time.sleep(0.05)
                    break

        finally:
            if session is not None:
                session.close(final_frame=last_ui)
            self.feed0.release()
            self.feed1.release()
            if display:
                cv2.destroyAllWindows()
