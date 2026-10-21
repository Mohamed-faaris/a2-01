#!/usr/bin/env python3
"""Tracking layer for the Single-Camera Planes & Cube Tracker.

- CameraCalibration: NPZ/JSON/YAML/named-profile loading + live V4L2 node
  resolution (per custom-calibrator-npz.md).
- VideoFeedSource / NonQueuingCameraReader: live webcam, video file or
  static image feed.
- Geometry (physics-constrained):
  * Plane B tags 42..45 are *coplanar markers on one platform*, so they all
    share ONE normal. The plane normal is estimated from the tag orientations
    (averaged, hemisphere-aligned) instead of from 4 noisy centers, then the
    offset is the robust mean of the center distances along that normal.
    Coplanarity RMSE is measured along the shared normal.
  * The cube rests ON the platform, so every cube face normal must be either
    PARALLEL (top/bottom) or PERPENDICULAR (sides) to the plane normal.
    Each observed face normal is snapped to that allowed set first; faces
    that disagree by more than ``max_snap_deg`` are rejected as outliers.
    The on-plane gap (should be 0 when resting) is reported explicitly,
    plus a contact-corrected centroid for reference.
- SingleCameraPlanesCubePipeline: Base (0), Plane A (40,41),
  Plane B (42..45) and Cube (20..25) tracking + realtime loop.

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

from viz import (
    annotate_camera_view,
    render_hud_dashboard,
    render_scene3d_panel,
    render_detections_panel,
    draw_base_frame_overlay,
    SessionLogger,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("1cam-planes-cube-tracker")

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


# ---------------------------------------------------------------------------
# Physics-constrained geometry
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return v / n


def fit_platform_plane_constrained(
    tag_data: list[tuple[int, np.ndarray, np.ndarray]],
    inlier_thresh_m: float = 0.005,
) -> dict | None:
    """Fit the platform plane from coplanar markers that SHARE one normal.

    tag_data: list of (tag_id, center_base_m, normal_base_m).

    Normal: averaged tag orientations (each marker sits flat on the plane, so
    every observed normal is the same direction up to PnP noise). Orientations
    are hemisphere-aligned first; this is far less noisy than a plane fit
    through 4 center points.
    Offset: robust (median-gated) mean of the signed center distances along
    the shared normal, then each center is projected ONTO the plane along
    that normal so the reported centers are truly coplanar.
    """
    if not tag_data:
        return None

    # 1. Shared normal from tag orientations
    normals = []
    for _, _, nvec in tag_data:
        n = _unit(np.asarray(nvec, dtype=np.float64))
        if normals and float(np.dot(n, normals[0])) < 0.0:
            n = -n
        normals.append(n)
    normal = _unit(np.mean(normals, axis=0))
    if normal[2] < 0.0:          # base frame: +Z is up
        normal = -normal

    # 2. Robust offset along the shared normal
    d_i = np.array([float(np.dot(normal, np.asarray(c, dtype=np.float64)))
                    for _, c, _ in tag_data], dtype=np.float64)
    d_med = float(np.median(d_i))
    inliers = np.abs(d_i - d_med) < inlier_thresh_m
    if not np.any(inliers):
        inliers = np.ones_like(inliers, dtype=bool)
    offset = float(np.mean(d_i[inliers]))

    # 3. Project centers onto the plane along the normal (enforces coplanarity)
    centers_proj: dict[int, np.ndarray] = {}
    resid = []
    for (tid, c, _), d_ct, is_in in zip(tag_data, d_i, inliers):
        c = np.asarray(c, dtype=np.float64)
        centers_proj[tid] = c - (float(np.dot(normal, c)) - offset) * normal
        if is_in:
            resid.append(float(np.dot(normal, c)) - offset)
    rmse_mm = float(np.sqrt(np.mean(np.square(resid)))) * 1000.0 if resid else 0.0

    center = np.mean(list(centers_proj.values()), axis=0)
    inlier_tids = [tid for (tid, _, _), is_in in zip(tag_data, inliers) if is_in]

    return {
        "normal": normal,
        "offset_m": offset,
        "center": center,
        "rmse_mm": rmse_mm,
        "centers_proj": centers_proj,
        "inlier_tids": inlier_tids,
    }


def snap_face_normal_to_axes(
    face_normal_base: np.ndarray,
    plane_normal: np.ndarray,
) -> tuple[np.ndarray, float, str]:
    """Snap an observed cube-face normal to the axes a cube-on-plane allows.

    A cube resting on the platform has face normals that are either PARALLEL
    to the plane normal (top/bottom) or PERPENDICULAR to it (four sides).

    Returns (snapped_normal, snap_error_deg, kind) with kind in
    {"top", "bottom", "side"}.
    """
    m = _unit(np.asarray(face_normal_base, dtype=np.float64))
    n = _unit(np.asarray(plane_normal, dtype=np.float64))
    c = float(np.dot(m, n))

    if abs(c) >= np.cos(np.radians(45.0)):
        snapped = n * (1.0 if c >= 0 else -1.0)
        kind = "top" if c >= 0 else "bottom"
    else:
        snapped = _unit(m - c * n)   # tangential (side) direction
        kind = "side"

    cos_a = float(np.clip(np.dot(m, snapped), -1.0, 1.0))
    return snapped, float(np.degrees(np.arccos(cos_a))), kind


def estimate_cube_centroid_constrained(
    observations: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    side_m: float,
    plane_normal: np.ndarray,
    plane_offset_m: float,
    max_snap_deg: float = 25.0,
    dist_thresh_m: float = 0.008,
) -> dict | None:
    """Estimate the cube centroid using the cube-on-plane constraints.

    observations: list of (tag_id, R_c_t, t_c_t, R_c_base, t_c_base).

    For each visible face: its observed normal is snapped to the allowed set
    (parallel/perpendicular to the plane normal); faces that disagree by more
    than ``max_snap_deg`` are inconsistent with the model and dropped. The
    candidate centroid is the face center pulled inward along the snapped
    normal by half the cube side. Candidates are combined with a medoid gate
    so a single bad pose cannot drag the centroid.

    Reported:
      xyz_mm            measured centroid (snapped face normals, medoid-gated)
      on_plane_gap_mm   distance between cube bottom and the platform
                        surface along the plane normal (0 = resting exactly)
      xyz_contact_mm    measured centroid projected onto exact contact
      max_snap_deg      worst face/axis disagreement kept
      yaw_deg           cube yaw (side faces) in the base frame
      face_snaps        per-face kind + snap angle (debug)
    """
    if not observations:
        return None

    n = _unit(np.asarray(plane_normal, dtype=np.float64))
    candidates: list[np.ndarray] = []
    face_snaps: dict[int, dict] = {}
    side_dirs: list[np.ndarray] = []

    for tid, R_c_t, t_c_t, R_c_base, t_c_base in observations:
        R_b_c = R_c_base.T
        t_b_c = -R_b_c @ t_c_base
        center = R_b_c @ np.asarray(t_c_t, dtype=np.float64).ravel() + t_b_c
        R_b_t = R_b_c @ np.asarray(R_c_t, dtype=np.float64)
        raw_normal = R_b_t @ np.array([0.0, 0.0, 1.0])

        snapped, snap_deg, kind = snap_face_normal_to_axes(raw_normal, n)
        if snap_deg > max_snap_deg:
            continue  # face pose inconsistent with cube-on-plane model
        candidates.append(center - (side_m / 2.0) * snapped)
        face_snaps[tid] = {"kind": kind, "snap_deg": snap_deg}
        if kind == "side":
            side_dirs.append(snapped)

    if not candidates:
        return None

    pts = np.array(candidates, dtype=np.float64)
    if len(pts) == 1:
        centroid = pts[0]
        spread_mm = 0.0
        inlier_count = 1
    else:
        # Medoid gate: majority cluster of candidate centroids
        best_count, best_inliers, best_medoid = 0, np.ones(len(pts), dtype=bool), pts[0]
        for i in range(len(pts)):
            inliers = np.linalg.norm(pts - pts[i], axis=1) < dist_thresh_m
            cnt = int(np.sum(inliers))
            if cnt > best_count:
                best_count, best_inliers, best_medoid = cnt, inliers, pts[i]
        inlier_pts = pts[best_inliers]
        centroid = np.mean(inlier_pts, axis=0)
        spread_mm = float(np.sqrt(np.mean(np.linalg.norm(inlier_pts - centroid, axis=1) ** 2))) * 1000.0
        inlier_count = int(best_count)

    # On-plane gap: bottom face vs platform surface along the plane normal
    surface_under_cube = float(np.dot(centroid, n)) - side_m / 2.0
    gap_m = surface_under_cube - float(plane_offset_m)
    contact_centroid = centroid + ((float(plane_offset_m) + side_m / 2.0)
                                   - float(np.dot(centroid, n))) * n

    # Cube yaw from side-face directions (circular mean in the plane)
    yaw_deg = None
    if side_dirs:
        mean_side = _unit(np.mean(np.array(side_dirs), axis=0))
        yaw_deg = float(np.degrees(np.arctan2(mean_side[1], mean_side[0])))

    max_snap = max((s["snap_deg"] for s in face_snaps.values()), default=0.0)

    return {
        "xyz_mm": (centroid * 1000.0).tolist(),
        "xyz_contact_mm": (contact_centroid * 1000.0).tolist(),
        "on_plane_gap_mm": gap_m * 1000.0,
        "visible_faces": sorted(face_snaps.keys()),
        "num_obs": len(candidates),
        "inlier_count": inlier_count,
        "residual_spread_mm": spread_mm,
        "max_snap_deg": max_snap,
        "yaw_deg": yaw_deg,
        "face_snaps": face_snaps,
    }


def entity_of_tag(tid: int) -> str:
    """Human-readable entity name for a tag id (used by the detections panel)."""
    if tid == 0:
        return "BASE"
    if tid in (40, 41):
        return "Plane A"
    if tid in range(42, 46):
        return "Plane B"
    if tid in range(20, 26):
        return "Cube"
    return "Other"


class SingleCameraPlanesCubePipeline:
    """Full single-camera tracking & real-time visualization pipeline."""

    def __init__(
        self,
        calib: CameraCalibration,
        source: str | int | None = None,
        markers_yaml: Path | str | None = None,
        active_ratio: float = 1.0,
    ):
        self.calib = calib
        self.active_ratio = active_ratio

        # Load markers registry
        self.registry = MarkerRegistry(markers_yaml)

        # Cube side length (tags 20..25 share one edge length)
        self.cube_side_m = self.registry.get_size(20)

        # Detector
        self.detector = Detector(families=DEFAULT_TAG_FAMILY, nthreads=2, quad_decimate=1.0)

        # Feed
        self.feed = VideoFeedSource(source, calib)

        # State storage
        self.last_results: dict = {}
        self.lock = threading.Lock()

        # Custom Baseline Coordinate Frame Feature (Key 'H')
        self.custom_h_frame_active: bool = False
        self.custom_h_frame_info: dict | None = None

        # 3D scene panel visibility (key 'V')
        self.show_scene3d: bool = True

    def process_frame(self, frame: np.ndarray) -> dict:
        """Process one frame and compute all geometric features."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        # 1. Detect tags
        dets = detect_and_estimate_poses(
            gray, self.calib.camera_matrix, self.calib.dist_coeffs,
            self.detector, self.registry, self.active_ratio
        )

        results = {
            "timestamp": time.time(),
            "cam_has_base": (0 in dets),
            "cam_tags": list(dets.keys()),
            "detections": [
                {"id": tid, "margin": pose.decision_margin, "entity": entity_of_tag(tid)}
                for tid, pose in sorted(dets.items())
            ],
            "cube_centroid": None,
            "plane_a": None,
            "plane_b": None,
        }

        # 2. Camera extrinsics relative to Base Tag 0
        if 0 not in dets:
            return results

        R_c_base, t_c_base = solve_camera_extrinsics_from_base(dets[0])
        cam_extrinsics = {0: (R_c_base, t_c_base)}

        # Expose camera extrinsics so the 3D overlays can re-project Base-Frame
        # points back into the camera image (used to visually verify arrays).
        results["cam_extrinsics"] = {
            0: {
                "R_c_base": R_c_base.tolist(),
                "t_c_base_m": t_c_base.tolist(),
                "position_base_mm": ((-R_c_base.T @ t_c_base) * 1000.0).tolist(),
            }
        }

        # 3. Reprojectable cube face quads (Base Frame, mm)
        cube_observations = []
        cube_face_corners: dict[int, list[np.ndarray]] = {}
        for tid in range(20, 26):
            if tid in dets:
                c_b, R_b_t = project_tag_to_base(dets[tid], R_c_base, t_c_base)
                cube_observations.append((tid, dets[tid].R_c_t, dets[tid].t_c_t,
                                          R_c_base, t_c_base))
                half = self.registry.get_size(tid) / 2.0
                local_corners = np.array([
                    [-half, -half, 0.0],
                    [ half, -half, 0.0],
                    [ half,  half, 0.0],
                    [-half,  half, 0.0],
                ], dtype=np.float64)
                cube_face_corners.setdefault(tid, []).append(c_b + (R_b_t @ local_corners.T).T)

        if cube_face_corners:
            results["cube_faces_mm"] = {
                tid: (np.mean(np.stack(cl), axis=0) * 1000.0).tolist()
                for tid, cl in cube_face_corners.items()
            }

        # 4. Track Plane A: Tags 40, 41 (Line / Pair Plane)
        plane_a_pts = []
        plane_a_centers = {}
        for tid in (40, 41):
            if tid in dets:
                c_base, _ = project_tag_to_base(dets[tid], R_c_base, t_c_base)
                plane_a_pts.append(c_base)
                plane_a_centers[tid] = c_base

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
            # 1. Baseline top is X-axis: x_init = (p41 - p40) / ||p41 - p40||
            # 2. Z is facing camera, 3. Right is Y-axis (cross)
            # 4. Rotate 90 deg around Y so Z faces UP
            # 5. Move Z +5 cm, X +2 cm
            # -------------------------------------------------------------
            v_baseline = p41 - p40
            norm_b = np.linalg.norm(v_baseline)
            if norm_b > 1e-6:
                x_init = v_baseline / norm_b

                # Camera optical center in base frame (single camera)
                cam_center = -R_c_base.T @ t_c_base
                cam_ray = cam_center - mid_pt
                z_init = cam_ray / max(1e-6, np.linalg.norm(cam_ray))

                # Gram-Schmidt: Orthogonalize z_init against x_init
                z_init = z_init - np.dot(z_init, x_init) * x_init
                z_init = z_init / max(1e-6, np.linalg.norm(z_init))

                y_init = np.cross(z_init, x_init)
                y_init = y_init / max(1e-6, np.linalg.norm(y_init))
                R_init = np.column_stack([x_init, y_init, z_init])

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
                R_custom = R_cand_pos if R_cand_pos[2, 2] >= R_cand_neg[2, 2] else R_cand_neg

                origin_custom = mid_pt + R_custom[:, 0] * 0.02 + R_custom[:, 2] * 0.05

                h_frame_info = {
                    "origin_base_m": origin_custom.tolist(),
                    "origin_base_mm": (origin_custom * 1000.0).tolist(),
                    "R_custom_base": R_custom.tolist(),
                    "x_axis": R_custom[:, 0].tolist(),
                    "y_axis": R_custom[:, 1].tolist(),
                    "z_axis": R_custom[:, 2].tolist(),
                }
                self.custom_h_frame_info = h_frame_info
                # cube_centroid added after the constrained cube estimate below

        # 5. Track Plane B: Tags 42..45 — coplanar markers, SHARED normal
        plane_b_data = []
        for tid in (42, 43, 44, 45):
            if tid in dets:
                c_base, R_t_base = project_tag_to_base(dets[tid], R_c_base, t_c_base)
                n_tag = R_t_base[:, 2].copy()   # tag local Z = surface normal
                plane_b_data.append((tid, c_base, n_tag))

        plane_fit = fit_platform_plane_constrained(plane_b_data)
        if plane_fit:
            results["plane_b"] = {
                "detected_tags": [tid for tid, _, _ in plane_b_data],
                "inlier_tags": plane_fit["inlier_tids"],
                "normal": plane_fit["normal"].tolist(),              # shared normal
                "shared_normal": True,
                "offset_m": plane_fit["offset_m"],
                "center_mm": (plane_fit["center"] * 1000.0).tolist(),
                "z_height_mm": float(plane_fit["center"][2] * 1000.0),
                "coplanarity_rmse_mm": plane_fit["rmse_mm"],
                "centers_mm": {tid: (c * 1000.0).tolist()
                               for tid, c, _ in plane_b_data},
                "centers_projected_mm": {tid: (c * 1000.0).tolist()
                                         for tid, c in plane_fit["centers_proj"].items()},
                "corners_mm": {tid: (c * 1000.0).tolist()
                               for tid, c in plane_b_data},
            }

        # 6. Constrained cube estimate (needs the plane's shared normal)
        if cube_observations:
            if plane_fit:
                n_pl = plane_fit["normal"]
                d_pl = plane_fit["offset_m"]
            else:
                n_pl = np.array([0.0, 0.0, 1.0])   # base table (tag 0 plane)
                d_pl = 0.0
            cube_res = estimate_cube_centroid_constrained(
                cube_observations, self.cube_side_m, n_pl, d_pl
            )
            if cube_res:
                results["cube_centroid"] = cube_res

        # 7. Finish the H-frame payload now that the cube is estimated
        if results["plane_a"] and self.custom_h_frame_info:
            h_frame_info = dict(self.custom_h_frame_info)
            if results["cube_centroid"]:
                cube_pt = np.array(results["cube_centroid"]["xyz_mm"]) / 1000.0
                R_custom = np.asarray(h_frame_info["R_custom_base"], dtype=np.float64)
                origin_custom = np.asarray(h_frame_info["origin_base_m"], dtype=np.float64)
                cube_in_h = R_custom.T @ (cube_pt - origin_custom)
                h_frame_info["cube_centroid_in_h_mm"] = (cube_in_h * 1000.0).tolist()
            results["custom_h_frame"] = h_frame_info

        return results

    # ------------------------------------------------------------------
    # Thin rendering wrappers (delegates to `viz` with this pipeline's
    # calibration; kept so callers/tests keep working).
    # ------------------------------------------------------------------

    def annotate_camera_view(self, frame: np.ndarray, dets: dict[int, DetectedTagPose],
                             cam_label: str, results: dict | None = None) -> np.ndarray:
        """Annotate tag polygons, IDs and the re-projected 3D overlay."""
        return annotate_camera_view(frame, dets, cam_label, results,
                                    self.calib.camera_matrix, self.calib.dist_coeffs)

    def draw_base_frame_overlay(self, vis: np.ndarray, results: dict) -> None:
        """Re-project fused metric results into the camera view (in-place)."""
        draw_base_frame_overlay(vis, results,
                                self.calib.camera_matrix, self.calib.dist_coeffs)

    def render_scene3d_panel(self, results: dict, width: int = 640, height: int = 720) -> np.ndarray:
        """Software-rendered 3D perspective view of the Base Frame scene."""
        return render_scene3d_panel(results, self.calib, width, height)

    def render_hud_dashboard(self, results: dict, width: int = 640, height: int = 720) -> np.ndarray:
        """Render the telemetry dashboard (H-frame status from this pipeline)."""
        return render_hud_dashboard(results, self.custom_h_frame_active, width, height)

    def render_detections_panel(self, results: dict, width: int = 640, height: int = 360) -> np.ndarray:
        """Render the live tag-detection list panel."""
        return render_detections_panel(results.get("detections") or [], results,
                                       width, height)

    def run(self, output_json: Path | None = None, display: bool = True,
            session: SessionLogger | None = None):
        """Main real-time tracking loop.

        Saves (per session layout ``outputs/<name>/<time>/``):
          - ``live_tracking_results.json``  latest telemetry snapshot
          - ``frames/*.png``                camera UI photos ('S' + interval)
          - ``history.csv``                 time series
        """
        if output_json:
            out_path = Path(output_json)
        elif session is not None:
            out_path = session.session_dir / "live_tracking_results.json"
        else:
            out_path = REPO_ROOT / "outputs" / "live_tracking_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        last_ui: np.ndarray | None = None
        try:
            while True:
                # Grab with a short deadline (first live frame may lag)
                f = self.feed.get_frame()
                waited = 0.0
                while f is None and waited < 5.0:
                    time.sleep(0.02)
                    waited += 0.02
                    f = self.feed.get_frame()
                if f is None:
                    log.warning("No frame received from source; stopping tracking loop.")
                    break

                # Process geometric features
                results = self.process_frame(f)
                with self.lock:
                    self.last_results = results

                # Annotate camera view
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
                dets = detect_and_estimate_poses(gray, self.calib.camera_matrix,
                                                 self.calib.dist_coeffs,
                                                 self.detector, self.registry,
                                                 self.active_ratio)
                v = self.annotate_camera_view(f, dets, f"Camera 0 ({self.calib.name})", results)

                # Composite: [cam 640x360 + detections 640x360 | 3D scene | HUD]
                cam_column = np.vstack([
                    cv2.resize(v, (640, 360)),
                    self.render_detections_panel(results, width=640, height=360),
                ])
                hud_panel = self.render_hud_dashboard(results, width=640, height=720)

                panels = [cam_column]
                if self.show_scene3d:
                    panels.append(self.render_scene3d_panel(results, width=640, height=720))
                panels.append(hud_panel)
                combined_ui = np.hstack(panels)  # 1920x720
                last_ui = combined_ui

                # Time-series history log + periodic annotated-frame snapshots
                if session is not None:
                    session.record(results, combined_ui)

                if display:
                    cv2.imshow("Single-Camera AprilTag Multi-Feature Tracker", combined_ui)
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
                        with open(out_path, "w", encoding="utf-8") as fjs:
                            json.dump(results, fjs, indent=2)
                        log.info(f"Saved snapshot to: {out_path}")
                        # Save photo snapshot on 'S' keypress
                        photo_path = out_path.with_suffix(".png")
                        try:
                            cv2.imwrite(str(photo_path), combined_ui)
                            log.info(f"Saved snapshot photo to: {photo_path}")
                        except Exception as exc:
                            log.warning(f"Could not save snapshot photo {photo_path}: {exc}")
                        if session is not None:
                            stamp_str = time.strftime("%Y%m%d_%H%M%S")
                            session.save_photo(combined_ui, f"snapshot_{session._frame_idx:06d}_{stamp_str}.png")
                else:
                    # Headless: always save the live results JSON
                    with open(out_path, "w", encoding="utf-8") as fjs:
                        json.dump(results, fjs, indent=2)
                    log.info(f"Saved headless snapshot to: {out_path}")
                    time.sleep(0.05)
                    break

        finally:
            if session is not None:
                session.close(final_frame=last_ui)
            self.feed.release()
            if display:
                cv2.destroyAllWindows()
