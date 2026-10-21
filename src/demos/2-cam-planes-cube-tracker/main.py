#!/usr/bin/env python3
"""Two-Camera Multi-Feature Realtime Tracker (CLI entrypoint):

- Reference Base: Tag 0 (origin and coordinate basis loaded from markers.yaml)
- Plane A (Line / Pair Plane): Tags 40, 41
- Plane B (Quad Platform Plane): Tags 42, 43, 44, 45 (RANSAC normal & coordinates)
- 6-Faced AprilTag Cube: Tags 20..25 (RANSAC 3D internal centroid tracking)
- Multi-Camera Fusion: 2 Cameras with interactive HUD, 3D metric tracking, and export.

Modules:
- ``tracker``: calibration loading, video feeds, geometry, tracking pipeline.
- ``viz``: 3D overlays, 3D scene panel, HUD dashboard, session logs.
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

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

# Add src (and this demo dir) to python path if needed
DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent.parent.parent
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(SRC_ROOT), str(DEMO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tracker import (  # noqa: E402
    CameraCalibration,
    DEFAULT_CALIB_SEARCH_DIRS,
    TwoCameraPlanesCubePipeline,
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


def main():
    parser = argparse.ArgumentParser(description="Two-Camera Planes (40,41 and 42..45) & Cube (20..25) Tracker Demo")
    parser.add_argument("--cam0", type=str, default=None, help="Camera 0 source (index, /dev/videoN, or image path)")
    parser.add_argument("--cam1", type=str, default=None, help="Camera 1 source (index, /dev/videoN, or image path)")
    parser.add_argument("--calib0", type=str, default=None, help="Camera 0 calibration profile or .npz path")
    parser.add_argument("--calib1", type=str, default=None, help="Camera 1 calibration profile or .npz path")
    parser.add_argument("--markers-yaml", type=str, default="calibrations/markers.yaml", help="Path to markers.yaml")
    parser.add_argument("--active-ratio", type=float, default=1.0, help="Ratio of active tag to border (default 1.0, 0.8 for padded benchmarks)")
    parser.add_argument("--no-display", action="store_true", help="Run headlessly without opening an OpenCV window")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save output JSON telemetry snapshot")
    parser.add_argument("--session-dir", type=str, default=str(REPO_ROOT / "outputs"),
                        help="Root directory for session logs: outputs/session_<ts>/history.csv + frames/*.png")
    parser.add_argument("--history-interval", type=float, default=0.1,
                        help="Seconds between history.csv time-series rows (default 0.1)")
    parser.add_argument("--photo-interval", type=float, default=0.0,
                        help="Seconds between automatic saved UI frames (default 0.0, disabled; photos saved on 'S' keypress)")
    parser.add_argument("--no-session", action="store_true",
                        help="Disable the session time-series log and photo snapshots")
    parser.add_argument("--name", "--experiment", "--exp", "--experiment-name", type=str, default="default",
                        help="Experiment name subfolder under outputs/ (outputs/<name>/<time>/...)")
    parser.add_argument("--no-timestamp", action="store_true",
                        help="Save directly into outputs/<experiment-name>/ without creating a session_<ts> subfolder")
    parser.add_argument("--list-cams", action="store_true", help="Scan and print connected V4L2 camera cards and stream nodes")
    parser.add_argument("--list-calibs", action="store_true", help="Scan and print discovered custom calibration profiles")
    args = parser.parse_args()

    if args.list_cams:
        print_camera_inspection()
        return

    if args.list_calibs:
        print_calib_inspection(DEFAULT_CALIB_SEARCH_DIRS)
        return

    calib0 = CameraCalibration.load(args.calib0, "Camera 0")
    calib1 = CameraCalibration.load(args.calib1, "Camera 1")

    pipeline = TwoCameraPlanesCubePipeline(
        calib0=calib0,
        calib1=calib1,
        source0=args.cam0,
        source1=args.cam1,
        markers_yaml=args.markers_yaml,
        active_ratio=args.active_ratio,
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
