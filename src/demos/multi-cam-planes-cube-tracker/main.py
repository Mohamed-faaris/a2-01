#!/usr/bin/env python3
"""Multi-Camera Multi-Feature Realtime Tracker (CLI entrypoint):

- Reference Base: Tag 0 (origin and coordinate basis loaded from markers.yaml / base_link.yaml)
- Plane A (Line / Pair Plane): Tags 40, 41
- Plane B (Quad Platform Plane): Tags 42, 43, 44, 45 (RANSAC normal & coordinates)
- 6-Faced AprilTag Cube: Tags 20..25 (RANSAC 3D internal centroid tracking)
- Multi-Camera Fusion: Arbitrary N Cameras (1, 2, 3, 4, ... N) with interactive HUD,
  3D metric tracking, software 3D scene panel, and multi-camera export.

Modules:
- ``tracker``: calibration loading, video feeds, geometry, multi-camera PnP pipeline.
- ``viz``: 3D overlays, 3D scene panel, HUD dashboard, session logs.
- ``viewer3d``: interactive WebGL 3D visualizer for recorded or live sessions.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

# Suppress Qt font, Wayland, and OpenCV internal V4L2 debug warnings in console
os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false;qt.qpa.*=false;qt.text.*=false")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")

import cv2
import numpy as np
import yaml

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

# Add src and this demo dir to python path
DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent.parent.parent
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(SRC_ROOT), str(DEMO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tracker import (  # noqa: E402
    CameraCalibration,
    DEFAULT_CALIB_SEARCH_DIRS,
    MultiCameraPlanesCubePipeline,
    enumerate_v4l2_streams,
)
from viz import SessionLogger  # noqa: E402


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
        print("  No custom calibration archives found in search directories.")
    else:
        for p in profs:
            rms_str = f"RMS: {p['rms']:.3f}px" if p["rms"] is not None else "RMS: N/A"
            res_str = f"{p['size'][0]}x{p['size'][1]}"
            print(f"  Profile: {p['name']:<18} | {res_str:<10} | {rms_str:<14} | Card: {p['card'] or 'Unknown'}")
            print(f"           Path: {p['path']}")
    print("=" * 80)


def build_camera_and_calib_lists(args, config: dict | None = None) -> tuple[list[str | int | None], list[CameraCalibration]]:
    """Resolve camera sources and calibration profiles from args and config.yaml."""
    # 1. Gather camera sources
    sources: list[str | int | None] = []
    if getattr(args, "cams", None):
        sources = list(args.cams)

    # Check explicit numbered flags (--cam0, --cam1, --cam2, ...)
    direct_cams = [
        getattr(args, "cam0", None),
        getattr(args, "cam1", None),
        getattr(args, "cam2", None),
        getattr(args, "cam3", None),
        getattr(args, "cam4", None),
        getattr(args, "cam5", None),
    ]
    for idx, c in enumerate(direct_cams):
        if c is not None:
            while len(sources) <= idx:
                sources.append(None)
            sources[idx] = c

    # 2. Gather calibration queries
    calib_queries: list[str | None] = []
    if getattr(args, "calibs", None):
        calib_queries = list(args.calibs)

    direct_calibs = [
        getattr(args, "calib0", None),
        getattr(args, "calib1", None),
        getattr(args, "calib2", None),
        getattr(args, "calib3", None),
        getattr(args, "calib4", None),
        getattr(args, "calib5", None),
    ]
    for idx, c in enumerate(direct_calibs):
        if c is not None:
            while len(calib_queries) <= idx:
                calib_queries.append(None)
            calib_queries[idx] = c

    # 3. If neither CLI sources nor calibs provided, fallback to config.yaml if present
    if not sources and not calib_queries and config and "cameras" in config:
        cfg_cams = config["cameras"]
        if isinstance(cfg_cams, dict):
            sources = list(cfg_cams.get("sources", []))
            calib_queries = list(cfg_cams.get("calibs", []))
        elif isinstance(cfg_cams, list):
            sources = [c.get("source") for c in cfg_cams if isinstance(c, dict)]
            calib_queries = [c.get("calibration") or c.get("calib") for c in cfg_cams if isinstance(c, dict)]

    # Determine total camera count
    n_cams = max(len(sources), len(calib_queries))
    if n_cams == 0:
        # Default fallback: 2 cameras (cam 0, cam 1)
        n_cams = 2
        sources = [None, None]
        calib_queries = [None, None]

    while len(sources) < n_cams:
        sources.append(None)
    while len(calib_queries) < n_cams:
        calib_queries.append(None)

    # Load calibrations using unified CameraCalibration loader
    calibs: list[CameraCalibration] = []
    for idx in range(n_cams):
        q = calib_queries[idx]
        calib = CameraCalibration.load(q, default_name=f"Camera {idx}")
        calibs.append(calib)

    return sources, calibs


def main():
    import re
    # Normalize accidental 3+ leading dashes (e.g. ---calibs -> --calibs)
    sys.argv = [re.sub(r"^-{3,}", "--", a) for a in sys.argv]

    parser = argparse.ArgumentParser(description="Multi-Camera Planes (40,41 and 42..45) & Cube (20..25) Tracker Demo")
    parser.add_argument("--cams", nargs="+", type=str, default=None,
                        help="List of camera sources (e.g. --cams 0 1 2 or --cams /dev/video0 /dev/video2)")
    parser.add_argument("--calibs", nargs="+", type=str, default=None,
                        help="List of calibration profiles or paths (e.g. --calibs cam0 cam1 cam2)")

    # Direct numbered camera source arguments for compatibility
    parser.add_argument("--cam0", type=str, default=None, help="Camera 0 source")
    parser.add_argument("--cam1", type=str, default=None, help="Camera 1 source")
    parser.add_argument("--cam2", type=str, default=None, help="Camera 2 source")
    parser.add_argument("--cam3", type=str, default=None, help="Camera 3 source")
    parser.add_argument("--cam4", type=str, default=None, help="Camera 4 source")
    parser.add_argument("--cam5", type=str, default=None, help="Camera 5 source")

    # Direct numbered calibration arguments for compatibility
    parser.add_argument("--calib0", type=str, default=None, help="Camera 0 calibration profile or .npz path")
    parser.add_argument("--calib1", type=str, default=None, help="Camera 1 calibration profile or .npz path")
    parser.add_argument("--calib2", type=str, default=None, help="Camera 2 calibration profile or .npz path")
    parser.add_argument("--calib3", type=str, default=None, help="Camera 3 calibration profile or .npz path")
    parser.add_argument("--calib4", type=str, default=None, help="Camera 4 calibration profile or .npz path")
    parser.add_argument("--calib5", type=str, default=None, help="Camera 5 calibration profile or .npz path")

    parser.add_argument("--config", type=str, default=str(DEMO_DIR / "config.yaml"),
                        help="Path to tracker config.yaml (defining static_tags, moveable_cube, boxes, and cameras)")
    parser.add_argument("--markers-yaml", type=str, default="calibrations/markers.yaml", help="Path to markers.yaml")
    parser.add_argument("--active-ratio", type=float, default=1.0, help="Ratio of active tag to border (default 1.0)")
    parser.add_argument("--no-display", action="store_true", help="Run headlessly without opening an OpenCV window")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save output JSON telemetry snapshot")
    parser.add_argument("--session-dir", type=str, default=str(REPO_ROOT / "outputs"),
                        help="Root directory for session logs: outputs/<name>/<time>/...")
    parser.add_argument("--history-interval", type=float, default=0.1,
                        help="Seconds between history.csv time-series rows (default 0.1)")
    parser.add_argument("--photo-interval", type=float, default=0.0,
                        help="Seconds between automatic saved UI frames (default 0.0, disabled; photos saved on 'S' keypress)")
    parser.add_argument("--no-session", action="store_true",
                        help="Disable the session time-series log and photo snapshots")
    parser.add_argument("--name", "--experiment", "--exp", "--experiment-name", type=str, default="default",
                        help="Experiment name subfolder under outputs/ (outputs/<name>/<time>/...)")
    parser.add_argument("--no-timestamp", action="store_true",
                        help="Save directly into outputs/<experiment-name>/ without creating a timestamp subfolder")
    parser.add_argument("--base-link-config", type=str, default="calibrations/base_link.yaml",
                        help="Path to base_link YAML/JSON config file (default: calibrations/base_link.yaml)")
    parser.add_argument("--list-cams", action="store_true", help="Scan and print connected V4L2 camera cards and stream nodes")
    parser.add_argument("--list-calibs", action="store_true", help="Scan and print discovered custom calibration profiles")
    args = parser.parse_args()

    if args.list_cams:
        print_camera_inspection()
        return

    if args.list_calibs:
        print_calib_inspection(DEFAULT_CALIB_SEARCH_DIRS)
        return

    # Load configuration YAML
    config_data = {}
    if args.config:
        cfg_p = Path(args.config).expanduser().resolve()
        if cfg_p.is_file():
            try:
                config_data = yaml.safe_load(cfg_p.read_text()) or {}
                print(f"📖 Loaded tracker configuration from: {cfg_p}")
            except Exception as exc:
                print(f"⚠️  Warning: could not parse config file '{cfg_p}': {exc}")

    sources, calibs = build_camera_and_calib_lists(args, config=config_data)

    print("=" * 80)
    print(f"🚀 INITIALIZING MULTI-CAMERA PLANES & CUBE TRACKER ({len(calibs)} CAMERAS)")
    print("=" * 80)
    for idx, (src, cal) in enumerate(zip(sources, calibs)):
        src_desc = src if src is not None else "(auto-resolved)"
        print(f"  Camera {idx}: Source={src_desc} | Calib Profile='{cal.name}' ({cal.image_size[0]}x{cal.image_size[1]})")
    print("=" * 80)

    pipeline = MultiCameraPlanesCubePipeline(
        calibs=calibs,
        sources=sources,
        markers_yaml=args.markers_yaml,
        active_ratio=args.active_ratio,
        base_link_config=args.base_link_config,
        config=config_data,
    )

    out_p = Path(args.output_json) if args.output_json else None
    exp_name = args.name or "default"

    session = None
    if not args.no_session:
        exp_root = Path(args.session_dir) / exp_name
        session = SessionLogger(
            exp_root,
            history_interval_s=args.history_interval,
            photo_interval_s=args.photo_interval,
            use_timestamp_subfolder=(not args.no_timestamp),
        )

    try:
        pipeline.run(
            output_json=out_p,
            display=(not args.no_display),
            session=session,
            experiment_name=exp_name,
            session_dir=args.session_dir,
        )
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    main()
