#!/usr/bin/env python3
"""Single-Camera Multi-Feature Realtime Tracker (CLI entrypoint):

- Reference Base: Tag 0 (origin and coordinate basis loaded from markers.yaml)
- Plane A (Line / Pair Plane): Tags 40, 41
- Plane B (Quad Platform Plane): Tags 42, 43, 44, 45
  -> coplanar markers: ONE shared normal (estimated from tag orientations)
- 6-Faced AprilTag Cube: Tags 20..25
  -> faces snapped parallel/perpendicular to the plane normal,
     on-plane gap reported (0 = resting exactly on the platform)

Session layout (per request): outputs/<name>/<time>/
    live_tracking_results.json   latest telemetry snapshot
    history.csv                  time series
    frames/*.png                 camera UI photos ('S' key + photo-interval)

Modules:
- ``tracker``: calibration loading, video feed, geometry, tracking pipeline.
- ``viz``: 3D overlays, detections panel, 3D scene panel, HUD, session logs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

# Suppress Qt font and Wayland warnings in console
os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false;qt.qpa.*=false;qt.text.*=false")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import numpy as np

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
    SingleCameraPlanesCubePipeline,
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
    parser = argparse.ArgumentParser(description="Single-Camera Planes (40,41 and 42..45) & Cube (20..25) Tracker Demo")
    parser.add_argument("--cam", type=str, default=None, help="Camera source (index, /dev/videoN, video file, or image path)")
    parser.add_argument("--calib", type=str, default=None, help="Camera calibration profile or .npz path")
    parser.add_argument("--markers-yaml", type=str, default="calibrations/markers.yaml", help="Path to markers.yaml")
    parser.add_argument("--active-ratio", type=float, default=1.0, help="Ratio of active tag to border (default 1.0, 0.8 for padded benchmarks)")
    parser.add_argument("--no-display", action="store_true", help="Run headlessly without opening an OpenCV window")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save output JSON telemetry snapshot (default: <session>/live_tracking_results.json)")
    parser.add_argument("--session-dir", type=str, default=str(REPO_ROOT / "outputs"),
                        help="Root directory for session logs: <session-dir>/<name>/<time>/history.csv + frames/*.png")
    parser.add_argument("--history-interval", type=float, default=0.1,
                        help="Seconds between history.csv time-series rows (default 0.1)")
    parser.add_argument("--photo-interval", type=float, default=0.0,
                        help="Seconds between automatic saved UI frames (default 0.0, disabled; photos saved on 'S' keypress)")
    parser.add_argument("--no-session", action="store_true",
                        help="Disable the session time-series log and photo snapshots")
    parser.add_argument("--name", "--experiment", "--exp", "--experiment-name", type=str, default="default",
                        help="Experiment name folder under outputs/: outputs/<name>/<time>/... (e.g. demo_run)")
    parser.add_argument("--no-timestamp", action="store_true",
                        help="Save directly into outputs/<name>/ without creating the <time> subfolder")
    parser.add_argument("--list-cams", action="store_true", help="Scan and print connected V4L2 camera cards and stream nodes")
    parser.add_argument("--list-calibs", action="store_true", help="Scan and print discovered custom calibration profiles")
    args = parser.parse_args()

    if args.list_cams:
        print_camera_inspection()
        return

    if args.list_calibs:
        print_calib_inspection(DEFAULT_CALIB_SEARCH_DIRS)
        return

    calib = CameraCalibration.load(args.calib, "Camera 0")

    pipeline = SingleCameraPlanesCubePipeline(
        calib=calib,
        source=args.cam,
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
