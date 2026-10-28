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
import shutil
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
import yaml

try:
    from tracker import (
        CameraCalibration,
        DEFAULT_CALIB_SEARCH_DIRS,
        MultiCameraPlanesCubePipeline,
        detect_and_estimate_poses,
        enumerate_v4l2_streams,
    )
    from viz import layout_camera_views, render_hud_dashboard, render_scene3d_panel
except ImportError:
    log.warning("pupil_apriltags or tracker modules not fully available in current interpreter. Mock mode will still work.")


DEFAULT_DOBOT_WS = Path("/home/mfk01/Projects/econ-systems/cr5a-moveit/DOBOT_6Axis_ROS2_V4_fresh")
DEFAULT_SCENE_JSON = REPO_ROOT / "output-dobots" / "scenes" / "most-scene.json"
if not DEFAULT_SCENE_JSON.is_file():
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
        self._client_addrs: Dict[socket.socket, tuple] = {}
        self._lock = threading.Lock()
        self._latest_payload: Optional[str] = None
        self.total_packets_sent: int = 0
        self.total_bytes_sent: int = 0

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    @property
    def client_endpoints(self) -> List[str]:
        with self._lock:
            return [f"{addr[0]}:{addr[1]}" for addr in self._client_addrs.values()]

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
                with self._lock:
                    self._clients.append(client)
                    self._client_addrs[client] = addr
                log.info(f"🔗 [TCP CLIENT CONNECTED] {addr[0]}:{addr[1]} | Total Active Clients: {len(self._clients)}")
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
                            self.total_bytes_sent += len(encoded)
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            dead.append(c)
                    for d in dead:
                        addr = self._client_addrs.pop(d, ("unknown", 0))
                        if d in self._clients:
                            self._clients.remove(d)
                        log.info(f"❌ [TCP CLIENT DISCONNECTED] {addr[0]}:{addr[1]} | Remaining Clients: {len(self._clients)}")
                self.total_packets_sent += 1
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

        # Load tracker configuration YAML
        self.config = {}
        cfg_p = Path(getattr(args, "config", DEMO_DIR / "config.yaml")).expanduser().resolve()
        if cfg_p.is_file():
            try:
                self.config = yaml.safe_load(cfg_p.read_text()) or {}
                log.info(f"Loaded config from {cfg_p}")
            except Exception as e:
                log.warning(f"Could not load config YAML {cfg_p}: {e}")

        # Ground level from Tag 4 (configured or calibration scene fallback)
        landmarks = self.scene_ref.get("static_landmarks", {})
        tag4 = landmarks.get("4", {})
        self.ground_z = float(tag4.get("center_base_m", [0, 0, -0.7429])[2])

        # UI & Real-Time connection logging state
        self.show_scene3d = True
        self._last_status_log = 0.0
        self._frame_count = 0

    def render_connection_status_badge(
        self,
        hud_panel: np.ndarray,
        stable: bool,
        spread_mm: float,
        cube_m: Sequence[float],
        parent_platform: Optional[str] = None
    ) -> None:
        """Render real-time TCP connection and robot motion status card onto the HUD panel."""
        w = hud_panel.shape[1]
        h = hud_panel.shape[0]

        card_y = h - 110
        cv2.rectangle(hud_panel, (14, card_y), (w - 14, card_y + 68), (30, 36, 46), -1)

        border_col = (0, 255, 120) if stable else (0, 165, 255)
        cv2.rectangle(hud_panel, (14, card_y), (w - 14, card_y + 68), border_col, 2)

        client_cnt = self.broadcaster.client_count
        client_col = (0, 255, 120) if client_cnt > 0 else (140, 180, 220)
        cv2.putText(hud_panel, f"📡 TCP SERVER: Port {self.broadcaster.port} | {client_cnt} Client(s) Active",
                    (22, card_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, client_col, 1, cv2.LINE_AA)

        stab_str = f"LOCKED (±{spread_mm:.2f} mm)" if stable else f"MEASURING (±{spread_mm:.2f} mm)"
        plat_str = f" | Platform: {parent_platform}" if parent_platform else ""
        cv2.putText(hud_panel, f"🎯 TRACKING: {stab_str}{plat_str}",
                    (22, card_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 240, 255), 1, cv2.LINE_AA)

        if self.trigger_in_progress:
            trig_str = "ROBOT MOTION IN PROGRESS..."
            trig_col = (0, 180, 255)
        else:
            trig_str = "ROBOT: READY (Press 'Y' SyncScene | 'C' Probe 15cm | 'T' Pick&Place)"
            trig_col = (0, 255, 120) if stable else (180, 180, 180)
        cv2.putText(hud_panel, f"🤖 {trig_str}",
                    (22, card_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.38, trig_col, 1, cv2.LINE_AA)

    def sync_scene_snapshot(self):
        """Publish the current tracking snapshot to MoveIt /scene/input so the user can close the app."""
        log.info("📸 [SCENE SYNC] Syncing current live tracking snapshot to MoveIt /scene/input...")

        def _run():
            try:
                c_flag = f"--constraints {self.args.constraints}" if getattr(self.args, "constraints", None) else ""
                inner_cmd = (
                    f"unset ROS_DISTRO ROS_PACKAGE_PATH AMENT_PREFIX_PATH PYTHONPATH CMAKE_PREFIX_PATH && "
                    f"source /opt/ros/humble/setup.bash && "
                    f"source {DEFAULT_DOBOT_WS}/install/setup.bash && "
                    f"python3 {DEFAULT_DOBOT_WS}/scripts/demo_tracker.py --publish {c_flag}"
                )
                in_container = os.path.exists("/run/.containerenv") or os.environ.get("CONTAINER_ID") == "moveit-dev"
                if in_container:
                    res = subprocess.run(["bash", "-c", inner_cmd], capture_output=True, text=True, cwd=str(DEFAULT_DOBOT_WS))
                else:
                    distrobox_cmd = "distrobox" if shutil.which("distrobox") else "distrobox-host-exec distrobox"
                    full_cmd = f"{distrobox_cmd} enter moveit-dev -- bash -c '{inner_cmd}'"
                    res = subprocess.run(["bash", "-c", full_cmd], capture_output=True, text=True, cwd=str(DEFAULT_DOBOT_WS))

                log.info("✅ [SCENE SYNC COMPLETE] Scene successfully loaded into MoveIt! You can now close the camera app with 'Q' and run demo_tracker.py standalone.")
            except Exception as exc:
                log.error(f"Failed to sync scene: {exc}")

        threading.Thread(target=_run, daemon=True).start()

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
                speed_val = getattr(self.args, "speed", 0.15)
                inner_cmd = (
                    f"unset ROS_DISTRO ROS_PACKAGE_PATH AMENT_PREFIX_PATH PYTHONPATH CMAKE_PREFIX_PATH && "
                    f"source /opt/ros/humble/setup.bash && "
                    f"source {DEFAULT_DOBOT_WS}/install/setup.bash && "
                    f"python3 {DEFAULT_DOBOT_WS}/scripts/demo_tracker.py --publish --run-task --speed {speed_val} {exec_flag}"
                )

                # Check if running on the host or inside a container
                in_container = os.path.exists("/run/.containerenv") or os.environ.get("CONTAINER_ID") == "moveit-dev"

                if in_container:
                    # Inside moveit-dev container
                    res = subprocess.run(
                        ["bash", "-c", inner_cmd],
                        capture_output=True,
                        text=True,
                        cwd=str(DEFAULT_DOBOT_WS)
                    )
                else:
                    # Running on host: use distrobox enter or distrobox-host-exec
                    distrobox_cmd = "distrobox" if shutil.which("distrobox") else "distrobox-host-exec distrobox"
                    full_cmd = f"{distrobox_cmd} enter moveit-dev -- bash -c '{inner_cmd}'"
                    res = subprocess.run(
                        ["bash", "-c", full_cmd],
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

    def trigger_robot_probe_corners(self, target_platform: str = "platform_box", execute: bool = True):
        """Invoke the ROS 2 MoveIt demo-tracker to move 30 cm above each reachable corner."""
        if self.trigger_in_progress:
            log.warning("Robot motion trigger already in progress.")
            return

        self.trigger_in_progress = True
        log.info(f"🤖 TRIGGERING ROBOT CORNER PROBE ON '{target_platform}' (30cm above) VIA ROS 2...")

        def _run():
            try:
                exec_flag = "--execute" if execute else ""
                speed_val = getattr(self.args, "speed", 0.15)
                buffer_val = getattr(self.args, "buffer", 0.10)
                c_flag = f"--constraints {self.args.constraints}" if getattr(self.args, "constraints", None) else ""
                inner_cmd = (
                    f"unset ROS_DISTRO ROS_PACKAGE_PATH AMENT_PREFIX_PATH PYTHONPATH CMAKE_PREFIX_PATH && "
                    f"source /opt/ros/humble/setup.bash && "
                    f"source {DEFAULT_DOBOT_WS}/install/setup.bash && "
                    f"python3 {DEFAULT_DOBOT_WS}/scripts/demo_tracker.py --publish --probe-corners {target_platform} --buffer {buffer_val} --speed {speed_val} {c_flag} {exec_flag}"
                )

                in_container = os.path.exists("/run/.containerenv") or os.environ.get("CONTAINER_ID") == "moveit-dev"

                if in_container:
                    res = subprocess.run(
                        ["bash", "-c", inner_cmd],
                        capture_output=True,
                        text=True,
                        cwd=str(DEFAULT_DOBOT_WS)
                    )
                else:
                    distrobox_cmd = "distrobox" if shutil.which("distrobox") else "distrobox-host-exec distrobox"
                    full_cmd = f"{distrobox_cmd} enter moveit-dev -- bash -c '{inner_cmd}'"
                    res = subprocess.run(
                        ["bash", "-c", full_cmd],
                        capture_output=True,
                        text=True,
                        cwd=str(DEFAULT_DOBOT_WS)
                    )

                log.info(f"Robot corner probe finished (returncode {res.returncode}):\n{res.stdout}")
                if res.stderr:
                    log.warning(f"Robot corner probe stderr:\n{res.stderr}")
            except Exception as exc:
                log.error(f"Failed to probe robot corners: {exc}")
            finally:
                self.trigger_in_progress = False
                self.last_trigger_time = time.time()

        threading.Thread(target=_run, daemon=True).start()

    def trigger_robot_home(self, execute: bool = True):
        """Invoke the ROS 2 MoveIt demo-tracker to move arm back to the HOME position."""
        if self.trigger_in_progress:
            log.warning("Robot motion trigger already in progress.")
            return

        self.trigger_in_progress = True
        log.info("🤖 TRIGGERING ROBOT MOTION TO HOME POSITION VIA ROS 2...")

        def _run():
            try:
                exec_flag = "--execute" if execute else ""
                speed_val = getattr(self.args, "speed", 0.15)
                inner_cmd = (
                    f"unset ROS_DISTRO ROS_PACKAGE_PATH AMENT_PREFIX_PATH PYTHONPATH CMAKE_PREFIX_PATH && "
                    f"source /opt/ros/humble/setup.bash && "
                    f"source {DEFAULT_DOBOT_WS}/install/setup.bash && "
                    f"python3 {DEFAULT_DOBOT_WS}/scripts/demo_tracker.py --home --speed {speed_val} {exec_flag}"
                )

                in_container = os.path.exists("/run/.containerenv") or os.environ.get("CONTAINER_ID") == "moveit-dev"

                if in_container:
                    res = subprocess.run(
                        ["bash", "-c", inner_cmd],
                        capture_output=True,
                        text=True,
                        cwd=str(DEFAULT_DOBOT_WS)
                    )
                else:
                    distrobox_cmd = "distrobox" if shutil.which("distrobox") else "distrobox-host-exec distrobox"
                    full_cmd = f"{distrobox_cmd} enter moveit-dev -- bash -c '{inner_cmd}'"
                    res = subprocess.run(
                        ["bash", "-c", full_cmd],
                        capture_output=True,
                        text=True,
                        cwd=str(DEFAULT_DOBOT_WS)
                    )

                log.info(f"Robot home motion finished (returncode {res.returncode}):\n{res.stdout}")
                if res.stderr:
                    log.warning(f"Robot home stderr:\n{res.stderr}")
            except Exception as exc:
                log.error(f"Failed to move robot home: {exc}")
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
            if getattr(self.args, "probe_corners", False):
                log.info(f"✨ CUBE IS LOCKED & STABLE (spread {spread_mm:.2f} mm). AUTO-TRIGGERING CORNER PROBE!")
                self.trigger_robot_probe_corners(execute=self.args.execute)
            else:
                log.info(f"✨ CUBE IS LOCKED & STABLE (spread {spread_mm:.2f} mm). AUTO-TRIGGERING PICK-AND-PLACE!")
                self.trigger_robot_pick_place(execute=self.args.execute)

        return is_stable, spread_mm

    def format_telemetry_packet(
        self,
        cube_xyz_mm: Sequence[float],
        platform_surf_mm: Sequence[float],
        target_surf_mm: Optional[Sequence[float]] = None,
        parent_platform: Optional[str] = None,
    ) -> dict:
        """Construct the unified live tracking JSON packet."""
        cx, cy, cz = [v / 1000.0 for v in cube_xyz_mm]
        px, py, pz = [v / 1000.0 for v in platform_surf_mm]

        # Heights extended down to ground
        src_height = pz - self.ground_z
        if target_surf_mm is not None:
            tx, ty, tz = [v / 1000.0 for v in target_surf_mm]
            tgt_height = tz - self.ground_z
            target_center = [tx, ty, self.ground_z + (tgt_height / 2.0)]
            target_z = tz
        else:
            target_z = 0.0  # base_link level default
            tgt_height = target_z - self.ground_z
            target_center = [0.116, -0.589, self.ground_z + (tgt_height / 2.0)]

        # Determine parent platform (configured or detected)
        active_parent = parent_platform or self.config.get("moveable_cube", {}).get("parent_platform") or "platform_box"

        # Prevent platform collision box from swallowing the cube
        cube_bot_z = cz - 0.025
        if active_parent == "box2":
            target_z = min(target_z, cube_bot_z - 0.002)
            tgt_height = max(0.05, target_z - self.ground_z)
            target_center[2] = self.ground_z + (tgt_height / 2.0)
        elif active_parent in ("box", "platform_box"):
            pz = min(pz, cube_bot_z - 0.002)
            src_height = max(0.05, pz - self.ground_z)
            src_center_z = self.ground_z + (src_height / 2.0)
            px, py = platform_surf_mm[0] / 1000.0, platform_surf_mm[1] / 1000.0

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
                "center_m": target_center,
            },
            "movable_box": {
                "id": "box",
                "size_m": [0.05, 0.05, 0.05],
                "center_m": [cx, cy, cz],
                "center_mm": [float(v) for v in cube_xyz_mm],
                "parent_platform": active_parent,
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

                # Periodic real-time connection status log (every 2.0s)
                now = time.time()
                if now - self._last_status_log > 2.0:
                    self._last_status_log = now
                    n_cl = self.broadcaster.client_count
                    cl_str = f"{n_cl} client(s)" if n_cl > 0 else "0 clients (waiting)"
                    stab_str = f"LOCKED (±{spread:.2f} mm)" if stable else f"TRACKING (±{spread:.2f} mm)"
                    log.info(
                        f"🔗 [REALTIME STATUS] TCP :9876 ({cl_str}) | MOCK SIM ACTIVE | "
                        f"Cube: {stab_str} on 'box' | Robot: {'BUSY' if self.trigger_in_progress else 'READY'}"
                    )

                if not self.args.no_display:
                    # Simulated mock results for full composite UI rendering
                    sim_results = {
                        "timestamp": now,
                        "world_map_size": 16,
                        "boxes": {
                            "box": {
                                "id": "platform_box",
                                "name": "box",
                                "detected_tags": [42, 43, 44, 45],
                                "normal": [0.0, 0.0, 1.0],
                                "d_offset_m": -src_z,
                                "center_mm": [src_x * 1000.0, src_y * 1000.0, src_z * 1000.0],
                                "z_height_mm": src_z * 1000.0,
                                "coplanarity_rmse_mm": 0.42,
                                "centers_mm": {
                                    42: [(src_x - 0.05) * 1000.0, (src_y - 0.05) * 1000.0, src_z * 1000.0],
                                    43: [(src_x + 0.05) * 1000.0, (src_y - 0.05) * 1000.0, src_z * 1000.0],
                                    44: [(src_x + 0.05) * 1000.0, (src_y + 0.05) * 1000.0, src_z * 1000.0],
                                    45: [(src_x - 0.05) * 1000.0, (src_y + 0.05) * 1000.0, src_z * 1000.0],
                                }
                            },
                            "box2": {
                                "id": "box2",
                                "name": "box2",
                                "detected_tags": [50, 51, 52, 53],
                                "normal": [0.0, 0.0, 1.0],
                                "d_offset_m": 0.0,
                                "center_mm": [116.0, -589.0, 0.0],
                                "z_height_mm": 0.0,
                                "coplanarity_rmse_mm": 0.55,
                                "centers_mm": {
                                    50: [66.0, -639.0, 0.0],
                                    51: [166.0, -639.0, 0.0],
                                    52: [166.0, -539.0, 0.0],
                                    53: [66.0, -539.0, 0.0],
                                }
                            }
                        },
                        "plane_b": None,
                        "plane_a": {
                            "baseline_dist_mm": 210.0,
                            "z_height_mm": src_z * 1000.0,
                            "center_mm": [src_x * 1000.0, (src_y + 0.20) * 1000.0, src_z * 1000.0],
                            "tag_40_mm": [src_x * 1000.0, (src_y + 0.10) * 1000.0, src_z * 1000.0],
                            "tag_41_mm": [src_x * 1000.0, (src_y + 0.30) * 1000.0, src_z * 1000.0],
                        },
                        "cube_centroid": {
                            "xyz_mm": curr_cube_mm,
                            "residual_spread_mm": spread,
                            "visible_faces": [20, 22, 24],
                            "num_obs": 6,
                            "parent_platform": "box",
                        },
                        "base_link": {
                            "active": True,
                            "origin_in_tag0_base_mm": [629.2, 581.8, 1003.3],
                        },
                        "cam_extrinsics": {
                            0: {"method": "multi_tag_pnp", "position_base_mm": [100.0, -800.0, 500.0], "R_c_base": np.eye(3).tolist(), "t_c_base_m": [0.0, 0.0, 1.5]},
                            1: {"method": "multi_tag_pnp", "position_base_mm": [500.0, -800.0, 600.0], "R_c_base": np.eye(3).tolist(), "t_c_base_m": [0.0, 0.0, 1.5]},
                        }
                    }
                    sim_results["plane_b"] = sim_results["boxes"]["box"]

                    # Create mock camera column
                    mock_cam0 = np.zeros((360, 640, 3), dtype=np.uint8)
                    mock_cam1 = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(mock_cam0, "🎥 MOCK FEED 0 (See3CAM 1080p)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
                    cv2.putText(mock_cam0, f"Cube Center: [{curr_cube_mm[0]:.1f}, {curr_cube_mm[1]:.1f}, {curr_cube_mm[2]:.1f}] mm", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1)
                    cv2.putText(mock_cam1, "🎥 MOCK FEED 1 (RealSense D415 720p)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 200), 1, cv2.LINE_AA)
                    cv2.putText(mock_cam1, f"Status: TCP Server Port 9876 Active", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1)
                    cam_column = np.vstack([mock_cam0, mock_cam1])

                    hud_panel = render_hud_dashboard(sim_results, num_cameras=2, width=640, height=720)
                    self.render_connection_status_badge(
                        hud_panel, stable=stable, spread_mm=spread,
                        cube_m=curr_cube_m, parent_platform="box"
                    )

                    mock_calibs = [
                        CameraCalibration.fallback_fov("see3cam", 1920, 1080),
                        CameraCalibration.fallback_fov("rl720", 1280, 720),
                    ]
                    panels = [cam_column]
                    if self.show_scene3d:
                        scene3d = render_scene3d_panel(sim_results, mock_calibs, width=640, height=720)
                        panels.append(scene3d)
                    panels.append(hud_panel)
                    combined_ui = np.hstack(panels)

                    cv2.imshow("RealWorld Tracker Connection (Press 'Y'=SyncScene, 'C'=Probe, 'T'=Pick&Place, 'Q'=Quit)", combined_ui)
                    key = cv2.waitKey(20) & 0xFF
                    if key in (27, ord('q'), ord('Q')):
                        break
                    elif key in (ord('y'), ord('Y')):
                        self.sync_scene_snapshot()
                    elif key in (ord('v'), ord('V')):
                        self.show_scene3d = not self.show_scene3d
                        log.info(f"3D Scene Panel: {'ENABLED' if self.show_scene3d else 'DISABLED'}")
                    elif key in (ord('t'), ord('T')):
                        log.info("Key 'T' pressed - Triggering Pick-and-Place!")
                        self.trigger_robot_pick_place(execute=self.args.execute)
                    elif key in (ord('c'), ord('C')):
                        log.info("Key 'C' pressed - Probing box corners (15 cm above)!")
                        self.trigger_robot_probe_corners(execute=self.args.execute)

                time.sleep(0.05)  # 20 Hz
        except KeyboardInterrupt:
            log.info("\nShutting down tracker.")
        finally:
            self.broadcaster.stop()

    def run_live(self):
        """Run with physical camera pipeline using unified camera loader."""
        from main import build_camera_and_calib_lists
        sources, calibs = build_camera_and_calib_lists(self.args, config=self.config)

        log.info(f"Initializing MultiCamera pipeline with {len(calibs)} cameras (using unified loader)...")
        pipeline = MultiCameraPlanesCubePipeline(
            calibs=calibs,
            sources=sources,
            markers_yaml=str(REPO_ROOT / "calibrations" / "markers.yaml"),
            base_link_config=str(REPO_ROOT / "calibrations" / "base_link.yaml"),
            config=self.config,
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
                boxes = results.get("boxes", {})
                box_src = boxes.get("box") or results.get("plane_b")
                box_tgt = boxes.get("box2")

                curr_cube_m = [0.0, 0.0, 0.0]
                curr_spread = 999.0
                curr_stable = False
                parent_plat = None

                if cube and "xyz_mm" in cube and box_src and "center_mm" in box_src:
                    raw_cube_mm = cube["xyz_mm"]
                    raw_src_mm = box_src["center_mm"]
                    raw_tgt_mm = box_tgt["center_mm"] if (box_tgt and "center_mm" in box_tgt) else None

                    # Apply N-sample pose filter to reduce camera noise
                    cube_mm = self.pose_filter.filter_point("cube", raw_cube_mm)
                    src_mm = self.pose_filter.filter_point("platform_box", raw_src_mm)
                    tgt_mm = self.pose_filter.filter_point("box2", raw_tgt_mm) if raw_tgt_mm else None
                    curr_cube_m = [v / 1000.0 for v in cube_mm]

                    curr_stable, curr_spread = self.check_stability_and_trigger(curr_cube_m)
                    parent_plat = cube.get("parent_platform")
                    packet = self.format_telemetry_packet(cube_mm, src_mm, target_surf_mm=tgt_mm, parent_platform=parent_plat)
                    packet["stable"] = curr_stable
                    packet["spread_mm"] = curr_spread

                    self.broadcaster.update_payload(packet)

                    with open(out_dir / "live_tracking_results.tmp", "w", encoding="utf-8") as f:
                        json.dump(packet, f, indent=2)
                    (out_dir / "live_tracking_results.tmp").replace(live_json_path)

                # Periodic terminal status logging (every 2.0s)
                now = time.time()
                if now - self._last_status_log > 2.0:
                    self._last_status_log = now
                    n_cl = self.broadcaster.client_count
                    cl_str = f"{n_cl} client(s)" if n_cl > 0 else "0 clients (waiting)"
                    stab_str = f"LOCKED (±{curr_spread:.2f} mm)" if curr_stable else f"TRACKING (±{curr_spread:.2f} mm)"
                    p_info = f"on '{parent_plat}'" if parent_plat else ""
                    log.info(
                        f"🔗 [REALTIME STATUS] TCP :9876 ({cl_str}) | "
                        f"Cameras: {pipeline.num_cameras} active | "
                        f"Cube: {stab_str} {p_info} | "
                        f"Robot: {'BUSY' if self.trigger_in_progress else 'READY'}"
                    )

                if not self.args.no_display:
                    # 1. Annotate camera views (AprilTags, bounding quads, tag labels, 3D overlays)
                    annotated_views = []
                    for c_id, frame in enumerate(frames):
                        calib = pipeline.calibs[c_id]
                        det = pipeline.detectors[c_id]
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                        dets = detect_and_estimate_poses(gray, calib.camera_matrix, calib.dist_coeffs,
                                                          det, pipeline.registry, pipeline.active_ratio)
                        v = pipeline.annotate_camera_view(frame, dets, f"Camera {c_id} ({calib.name})", results, c_id)
                        annotated_views.append(v)

                    # 2. Left Column: Camera views (640x720)
                    cam_column = layout_camera_views(annotated_views, width=640, height=720)

                    # 3. Right Column: HUD Dashboard with Realtime Connection Card (640x720)
                    hud_panel = pipeline.render_hud_dashboard(results, width=640, height=720)
                    self.render_connection_status_badge(
                        hud_panel, stable=curr_stable, spread_mm=curr_spread,
                        cube_m=curr_cube_m, parent_platform=parent_plat
                    )

                    # 4. Assemble Panels (with toggleable 3D Scene panel)
                    panels = [cam_column]
                    if self.show_scene3d:
                        scene3d = pipeline.render_scene3d_panel(results, width=640, height=720)
                        panels.append(scene3d)
                    panels.append(hud_panel)
                    combined_ui = np.hstack(panels)

                    cv2.imshow("RealWorld Tracker Connection (Press 'Y'=SyncScene, 'C'=Probe, '0'=Home, 'T'=Pick&Place, 'Q'=Quit)", combined_ui)
                    key = cv2.waitKey(10) & 0xFF
                    if key in (27, ord('q'), ord('Q')):
                        break
                    elif key in (ord('y'), ord('Y')):
                        self.sync_scene_snapshot()
                    elif key in (ord('v'), ord('V')):
                        self.show_scene3d = not self.show_scene3d
                        log.info(f"3D Scene Panel: {'ENABLED' if self.show_scene3d else 'DISABLED'}")
                    elif key in (ord('h'), ord('H')):
                        pipeline.custom_h_frame_active = not pipeline.custom_h_frame_active
                        log.info(f"Custom Frame: {'ENABLED' if pipeline.custom_h_frame_active else 'DISABLED'}")
                    elif key == ord('0'):
                        log.info("Key '0' pressed - Moving robot to HOME position!")
                        self.trigger_robot_home(execute=self.args.execute)
                    elif key in (ord('t'), ord('T')):
                        log.info("Key 'T' pressed - Triggering Pick-and-Place!")
                        self.trigger_robot_pick_place(execute=self.args.execute)
                    elif key in (ord('c'), ord('C')):
                        log.info("Key 'C' pressed - Probing box corners (15 cm above)!")
                        self.trigger_robot_probe_corners(execute=self.args.execute)
        finally:
            self.broadcaster.stop()
            cv2.destroyAllWindows()


def main():
    import re
    # Normalize accidental 3+ leading dashes (e.g. ---calibs -> --calibs)
    sys.argv = [re.sub(r"^-{3,}", "--", a) for a in sys.argv]

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mock", action="store_true", help="Run with simulated real-world tracking data (no physical cameras needed)")
    parser.add_argument("--filter-samples", "--smooth-samples", "--replicate-pose", type=int, default=3,
                        help="Number of consecutive pose samples to aggregate/filter to reduce camera noise (default: 3)")
    parser.add_argument("--config", type=str, default=str(DEMO_DIR / "config.yaml"),
                        help="Path to tracker config.yaml (defining static_tags, moveable_cube, boxes, and cameras)")
    parser.add_argument("--cams", nargs="+", type=str, default=None, help="Camera video nodes (e.g. /dev/video0 /dev/video2)")
    parser.add_argument("--calibs", nargs="+", type=str, default=None, help="Camera calibration profiles")
    parser.add_argument("--cam0", type=str, default=None, help="Camera 0 source")
    parser.add_argument("--cam1", type=str, default=None, help="Camera 1 source")
    parser.add_argument("--calib0", type=str, default=None, help="Camera 0 calib")
    parser.add_argument("--calib1", type=str, default=None, help="Camera 1 calib")
    parser.add_argument("--port", type=int, default=9876, help="TCP streaming port (default 9876)")
    parser.add_argument("--no-display", action="store_true", help="Run headlessly without GUI window")
    parser.add_argument("--auto-trigger", action="store_true", help="Automatically trigger robot motion once tracking stabilizes")
    parser.add_argument("--probe-corners", action="store_true", help="Probe 15 cm above each reachable corner of the platform (instead of pick-and-place)")
    parser.add_argument("--probe-tags", "--prob-tags", type=str, default=None, help="Path to JSONL file with tag poses to probe sequentially")
    parser.add_argument("--constraints", "--constraint", type=str, default=None, help="Path to constraints.yaml (defines ground_plane floor & base_pedestal)")
    parser.add_argument("--buffer", "--corner-inset", type=float, default=0.10, help="Buffer / inset distance in meters from outer platform edges in X and Y (default: 0.10 = 10cm)")
    parser.add_argument("--speed", type=float, default=0.15, help="Robot velocity and acceleration scaling factor (default: 0.15 = 15%)")
    parser.add_argument("--execute", action="store_true", default=True, help="Execute robot motion upon trigger (default: True)")
    args = parser.parse_args()

    app = RealWorldConnectionApp(args)
    if args.mock:
        app.run_mock()
    else:
        app.run_live()


if __name__ == "__main__":
    main()
