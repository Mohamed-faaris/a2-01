#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""realworld_connection.py - Real-Time Multi-Camera AprilTag Tracker & Robot Bridge.

Integrates real-world multi-camera tracking (AprilTags 20..25 cube, 42..45 platform,
Tag 4 ground) with the DOBOT CR5 MoveIt pick-and-place stack:
1. Continuously captures live frames from multiple cameras (or mock data).
2. Computes PnP fusion in base_link frame (ground from Tag 4, platforms, 5cm cube).
3. Streams live 3D coordinates over TCP (port 9876) to the robot bridge.
4. Detects tracking stability (variance < 2 mm) and allows manual ('T' key) or
   automatic (--auto-trigger) pick-and-place execution.
5. Continuously writes atomic live_tracking_results.json.

Usage:
  # Run live with auto-discovered cameras & stream on port 9876:
  ./.venv/bin/python src/demos/multi-cam-planes-cube-tracker/realworld_connection.py

  # Run with mock tracker (no physical cameras needed):
  ./.venv/bin/python src/demos/multi-cam-planes-cube-tracker/realworld_connection.py --mock

  # Headless mode with auto-trigger on stable detection:
  ./.venv/bin/python src/demos/multi-cam-planes-cube-tracker/realworld_connection.py --no-display --auto-trigger
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Sequence, Dict, Any, List, Optional

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("realworld_connection")

# Path resolving
DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent.parent.parent
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(SRC_ROOT), str(DEMO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np

try:
    from tracker import (
        CameraCalibration,
        DEFAULT_CALIB_SEARCH_DIRS,
        MultiCameraPlanesCubePipeline,
        enumerate_v4l2_streams,
    )
    from viz import layout_camera_views
except ImportError:
    log.warning("pupil_apriltags or tracker modules not fully available in current interpreter. Mock mode will still work.")


DEFAULT_DOBOT_WS = Path("/home/mfk01/Projects/econ-systems/cr5a-moveit/DOBOT_6Axis_ROS2_V4_fresh")
DEFAULT_SCENE_JSON = DEFAULT_DOBOT_WS / "output-a201" / "scenes" / "scene.json"


# ── POSE NOISE FILTER ──────────────────────────────────────────────────
class PoseNoiseFilter:
    """Rolling multi-sample median filter for 3D coordinates to eliminate camera jitter."""

    def __init__(self, window_size: int = 3):
        self.window_size = max(1, int(window_size))
        self.buffers: Dict[str, collections.deque] = {}

    def filter_point(self, key: str, pt: Sequence[float]) -> List[float]:
        if self.window_size <= 1:
            return list(pt)
        if key not in self.buffers:
            self.buffers[key] = collections.deque(maxlen=self.window_size)
        self.buffers[key].append(list(pt))
        arr = np.array(self.buffers[key], dtype=np.float64)
        # Median filter: rejects single-frame outliers / edge noise
        return np.median(arr, axis=0).tolist()


# ── TCP STREAMING BROADCASTER ──────────────────────────────────────────
class LiveStreamBroadcaster:
    """Non-blocking TCP streaming server for live tracking telemetry."""

    def __init__(self, host: str = "0.0.0.0", port: int = 9876):
        self.host = host
        self.port = port
        self.running = False
        self._server_sock: Optional[socket.socket] = None
        self._clients: List[socket.socket] = []
        self._lock = threading.Lock()
        self._latest_payload: Optional[str] = None

    def start(self):
        self.running = True
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(5)
        self._server_sock.settimeout(0.2)
        log.info(f"📡 [TCP SERVER] Streaming tracking data on {self.host}:{self.port}")

        threading.Thread(target=self._accept_loop, daemon=True).start()
        threading.Thread(target=self._broadcast_loop, daemon=True).start()

    def update_payload(self, data: dict):
        with self._lock:
            self._latest_payload = json.dumps(data) + "\n"

    def _accept_loop(self):
        while self.running:
            try:
                client, addr = self._server_sock.accept()
                log.info(f"  [CLIENT CONNECTED] {addr}")
                with self._lock:
                    self._clients.append(client)
            except socket.timeout:
                continue
            except OSError:
                break

    def _broadcast_loop(self):
        while self.running:
            payload = None
            with self._lock:
                payload = self._latest_payload

            if payload and self._clients:
                encoded = payload.encode("utf-8")
                dead = []
                with self._lock:
                    for c in self._clients:
                        try:
                            c.sendall(encoded)
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            dead.append(c)
                    for d in dead:
                        if d in self._clients:
                            self._clients.remove(d)
            time.sleep(0.05)  # 20 Hz broadcast

    def stop(self):
        self.running = False
        if self._server_sock:
            self._server_sock.close()


# ── REALWORLD TRACKING & ROBOT BRIDGE ──────────────────────────────────
class RealWorldConnectionApp:
    def __init__(self, args):
        self.args = args
        self.pose_filter = PoseNoiseFilter(window_size=args.filter_samples)
        self.broadcaster = LiveStreamBroadcaster(host="0.0.0.0", port=args.port)
        self.broadcaster.start()
        log.info(f"Pose Noise Filter initialized with {self.pose_filter.window_size} sample rolling median.")

        # Stability tracking (for auto-trigger)
        self.history_buffer = collections.deque(maxlen=25)  # ~1 second window
        self.last_trigger_time = 0.0
        self.trigger_in_progress = False

        # Load reference calibration scene if available
        self.scene_ref = {}
        if DEFAULT_SCENE_JSON.is_file():
            try:
                with open(DEFAULT_SCENE_JSON, "r", encoding="utf-8") as f:
                    self.scene_ref = json.load(f)
                log.info(f"Loaded reference calibration from {DEFAULT_SCENE_JSON}")
            except Exception as e:
                log.warning(f"Could not load reference scene: {e}")

        # Ground level from Tag 4
        landmarks = self.scene_ref.get("static_landmarks", {})
        tag4 = landmarks.get("4", {})
        self.ground_z = float(tag4.get("center_base_m", [0, 0, -0.7429])[2])

    def trigger_robot_pick_place(self, execute: bool = True):
        """Invoke the ROS 2 MoveIt demo-tracker to trigger pick and place."""
        if self.trigger_in_progress:
            log.warning("Pick-and-place trigger already in progress.")
            return

        self.trigger_in_progress = True
        log.info("🤖 TRIGGERING ROBOT PICK-AND-PLACE VIA ROS 2...")

        def _run():
            try:
                exec_flag = "--execute" if execute else ""
                cmd = (
                    f"source /opt/ros/humble/setup.bash && "
                    f"source {DEFAULT_DOBOT_WS}/install/setup.bash && "
                    f"python3 {DEFAULT_DOBOT_WS}/scripts/demo_tracker.py --publish --run-task {exec_flag}"
                )
                res = subprocess.run(
                    ["bash", "-c", cmd],
                    capture_output=True,
                    text=True,
                    cwd=str(DEFAULT_DOBOT_WS)
                )
                log.info(f"Robot trigger finished (returncode {res.returncode}):\n{res.stdout}")
                if res.stderr:
                    log.warning(f"Robot trigger stderr:\n{res.stderr}")
            except Exception as exc:
                log.error(f"Failed to trigger robot: {exc}")
            finally:
                self.trigger_in_progress = False
                self.last_trigger_time = time.time()

        threading.Thread(target=_run, daemon=True).start()

    def check_stability_and_trigger(self, cube_xyz_m: Sequence[float]):
        """Detect when cube position is stationary (< 2mm variance for 1s)."""
        self.history_buffer.append(cube_xyz_m)
        if len(self.history_buffer) < 20:
            return False, 999.0

        pts = np.array(self.history_buffer)
        spread_mm = float(np.max(np.std(pts, axis=0)) * 1000.0)

        is_stable = spread_mm < 2.0  # under 2mm noise
        now = time.time()

        if is_stable and self.args.auto_trigger and (now - self.last_trigger_time > 30.0) and not self.trigger_in_progress:
            log.info(f"✨ CUBE IS LOCKED & STABLE (spread {spread_mm:.2f} mm). AUTO-TRIGGERING PICK-AND-PLACE!")
            self.trigger_robot_pick_place(execute=self.args.execute)

        return is_stable, spread_mm

    def format_telemetry_packet(self, cube_xyz_mm: Sequence[float], platform_surf_mm: Sequence[float]) -> dict:
        """Construct the unified live tracking JSON packet."""
        cx, cy, cz = [v / 1000.0 for v in cube_xyz_mm]
        px, py, pz = [v / 1000.0 for v in platform_surf_mm]

        # Heights extended down to ground
        src_height = pz - self.ground_z
        target_z = 0.0  # base_link level
        tgt_height = target_z - self.ground_z

        packet = {
            "timestamp": time.time(),
            "status": "TRACKING_ACTIVE",
            "ground_plane": {
                "id": "ground_plane",
                "surface_z_m": self.ground_z,
                "size_m": [10.0, 10.0, 0.05],
                "center_m": [0.0, 0.0, self.ground_z - 0.025],
            },
            "source_platform": {
                "id": "platform_box",
                "surface_z_m": pz,
                "size_m": [0.30, 0.30, src_height],
                "center_m": [px, py, self.ground_z + (src_height / 2.0)],
            },
            "target_platform": {
                "id": "box2",
                "surface_z_m": target_z,
                "size_m": [0.30, 0.30, tgt_height],
                "center_m": [0.116, -0.589, self.ground_z + (tgt_height / 2.0)],
            },
            "movable_box": {
                "id": "box",
                "size_m": [0.05, 0.05, 0.05],
                "center_m": [cx, cy, cz],
                "center_mm": [float(v) for v in cube_xyz_mm],
            }
        }
        return packet

    def run_mock(self):
        """Simulate real-world tracking using calibration data & realistic noise."""
        log.info("▶ Running in MOCK TRACKER mode (broadcasting on port 9876)...")
        wp_planes = self.scene_ref.get("workpiece_planes", {})
        box_plane = wp_planes.get("box", {})
        base_src_m = box_plane.get("center_m", [0.640, -0.114, -0.492])

        # Clamped reachable radius
        r_src = math.hypot(base_src_m[0], base_src_m[1])
        if r_src > 0.650:
            scale = 0.650 / r_src
            src_x = base_src_m[0] * scale
            src_y = base_src_m[1] * scale
        else:
            src_x, src_y = base_src_m[0], base_src_m[1]
        src_z = base_src_m[2]

        cube_base_mm = [src_x * 1000.0, src_y * 1000.0, (src_z + 0.027) * 1000.0]
        plat_surf_mm = [src_x * 1000.0, src_y * 1000.0, src_z * 1000.0]

        log.info("Press 'T' in terminal or invoke trigger to perform pick and place.")
        log.info(f"Target object 'box' (5cm cube) sitting at [{src_x:.3f}, {src_y:.3f}, {src_z+0.027:.3f}] m")

        out_dir = REPO_ROOT / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        live_json_path = out_dir / "live_tracking_results.json"

        try:
            while True:
                # Add 0.3mm simulated sensor noise
                noise = np.random.normal(0, 0.3, 3)
                raw_cube_mm = [cube_base_mm[i] + float(noise[i]) for i in range(3)]
                
                # Apply N-sample pose filter to reduce noise
                curr_cube_mm = self.pose_filter.filter_point("cube", raw_cube_mm)
                curr_cube_m = [v / 1000.0 for v in curr_cube_mm]

                stable, spread = self.check_stability_and_trigger(curr_cube_m)
                packet = self.format_telemetry_packet(curr_cube_mm, plat_surf_mm)
                packet["stable"] = stable
                packet["spread_mm"] = spread

                self.broadcaster.update_payload(packet)

                # Write atomic JSON snapshot
                tmp_json = out_dir / "live_tracking_results.tmp"
                with open(tmp_json, "w", encoding="utf-8") as f:
                    json.dump(packet, f, indent=2)
                tmp_json.replace(live_json_path)

                time.sleep(0.05)  # 20 Hz
        except KeyboardInterrupt:
            log.info("\nShutting down tracker.")
        finally:
            self.broadcaster.stop()

    def run_live(self):
        """Run with physical camera pipeline."""
        sources = self.args.cams
        if not sources:
            log.info("Auto-detecting connected V4L2 cameras...")
            detected = enumerate_v4l2_streams()
            sources = [c["node"] for c in detected if c.get("has_capture")][:2]
            if not sources:
                log.error("No physical cameras found! Switch to --mock mode to test without cameras.")
                return

        log.info(f"Initializing MultiCamera pipeline with cameras: {sources}")
        calibs = [CameraCalibration.load(c, default_name=f"Cam{i}") for i, c in enumerate(sources)]
        pipeline = MultiCameraPlanesCubePipeline(
            calibs=calibs,
            sources=sources,
            markers_yaml=str(REPO_ROOT / "calibrations" / "markers.yaml"),
            base_link_config=str(REPO_ROOT / "calibrations" / "base_link.yaml"),
        )

        out_dir = REPO_ROOT / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        live_json_path = out_dir / "live_tracking_results.json"

        try:
            while True:
                frames = [feed.get_frame() for feed in pipeline.feeds]
                if all(f is None for f in frames):
                    time.sleep(0.02)
                    continue

                for c_id in range(pipeline.num_cameras):
                    if frames[c_id] is None:
                        calib = pipeline.calibs[c_id]
                        frames[c_id] = np.zeros((calib.image_size[1], calib.image_size[0], 3), dtype=np.uint8)

                results = pipeline.process_frames(frames)
                cube = results.get("cube_centroid")
                plane_b = results.get("plane_b")

                if cube and "xyz_mm" in cube and plane_b and "center_mm" in plane_b:
                    raw_cube_mm = cube["xyz_mm"]
                    raw_plat_mm = plane_b["center_mm"]
                    
                    # Apply N-sample pose filter to reduce camera noise
                    cube_mm = self.pose_filter.filter_point("cube", raw_cube_mm)
                    plat_mm = self.pose_filter.filter_point("platform", raw_plat_mm)
                    cube_m = [v / 1000.0 for v in cube_mm]

                    stable, spread = self.check_stability_and_trigger(cube_m)
                    packet = self.format_telemetry_packet(cube_mm, plat_mm)
                    packet["stable"] = stable
                    packet["spread_mm"] = spread

                    self.broadcaster.update_payload(packet)

                    with open(out_dir / "live_tracking_results.tmp", "w", encoding="utf-8") as f:
                        json.dump(packet, f, indent=2)
                    (out_dir / "live_tracking_results.tmp").replace(live_json_path)

                if not self.args.no_display:
                    cam_col = layout_camera_views(frames, width=640, height=720)
                    cv2.imshow("RealWorld Tracker Connection (Press 'T' for Pick&Place, 'Q' to quit)", cam_col)
                    key = cv2.waitKey(20) & 0xFF
                    if key in (27, ord('q'), ord('Q')):
                        break
                    elif key in (ord('t'), ord('T')):
                        log.info("Key 'T' pressed - Triggering Pick-and-Place!")
                        self.trigger_robot_pick_place(execute=self.args.execute)
        finally:
            self.broadcaster.stop()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mock", action="store_true", help="Run with simulated real-world tracking data (no physical cameras needed)")
    parser.add_argument("--filter-samples", "--smooth-samples", "--replicate-pose", type=int, default=3,
                        help="Number of consecutive pose samples to aggregate/filter to reduce camera noise (default: 3)")
    parser.add_argument("--cams", nargs="+", type=str, default=None, help="Camera video nodes (e.g. /dev/video0 /dev/video2)")
    parser.add_argument("--port", type=int, default=9876, help="TCP streaming port (default 9876)")
    parser.add_argument("--no-display", action="store_true", help="Run headlessly without GUI window")
    parser.add_argument("--auto-trigger", action="store_true", help="Automatically trigger robot pick-and-place once tracking stabilizes")
    parser.add_argument("--execute", action="store_true", default=True, help="Execute robot motion upon trigger (default: True)")
    args = parser.parse_args()

    app = RealWorldConnectionApp(args)
    if args.mock:
        app.run_mock()
    else:
        app.run_live()


if __name__ == "__main__":
    main()
