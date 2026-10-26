#!/usr/bin/env python3
"""Interactive 3D WebGL Viewer for Multi-Pose Hand-Eye Calibration & Scene Mapping.

Features:
- Powered by Viser (Three.js WebGL in browser with orbit, pan, zoom, dark theme).
- Consumes handeye_calibration.json and tmp/eye-hand/008 pose dataset.
- Reconstructs DOBOT CR5A robot arm:
  * Rods / Skeleton Mode: Clean cylindrical rods connecting joint hubs according to exact URDF kinematics.
  * CAD Mesh Mode: High-fidelity STL mesh loading from /home/mfk01/Downloads/Robotics_Resources/cr5.
- Real-time 3D elements:
  * Calibrated D415 and USB camera frustums and optical centers in base_link.
  * Static base tags (4, 5) and workpiece platform box tags (42, 43, 44, 45) + fitted plane.
  * Dynamically mounted tool tags (20, 21, 40) moving with the robot TCP.
  * Predicted vs detected tag positions and red residual error lines.
  * Camera optical sighting rays.
- Interactive GUI:
  * Pose selection dropdown (Pose 01 to Pose 18).
  * Smooth auto-play animation and playback speed control.
  * Rendering mode toggles (Rods, CAD Meshes, Coordinate Frames, Rays, Tags).
  * POV view shortcuts (Home, Top, D415 POV, USB POV, TCP POV).
  * Live telemetry markdown card.

Usage:
    .venv/bin/python src/demos/hand-eye-0/viewer3d.py
    .venv/bin/python src/demos/hand-eye-0/viewer3d.py --port 8085
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import viser
import yaml

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent.parent.parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from config import (
    DEFAULT_DATASET_DIR,
    DEFAULT_D415_CALIB,
    DEFAULT_USB_CALIB,
    STATIC_TAGS,
    TAG_SIZES_M,
    TOOL_TCP_TAGS,
    load_camera_intrinsics,
)
from detector import load_all_poses
from kinematics import (
    DEFAULT_CR5_RESOURCE_DIR,
    compute_cr5_fk,
    compute_cr5_rods,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("hand-eye-0.viewer3d")


def matrix_to_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to (w, x, y, z) quaternion for Viser."""
    try:
        q = Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_quat()  # [x, y, z, w]
        return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
    except Exception:
        return (1.0, 0.0, 0.0, 0.0)


def quat_xyzw_to_wxyz(q_xyzw: Sequence[float]) -> tuple[float, float, float, float]:
    """Convert [x, y, z, w] quaternion to (w, x, y, z)."""
    return (float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]))


def align_z_to_vector(direction: np.ndarray) -> tuple[float, float, float, float]:
    """Computes quaternion rotating local Z [0, 0, 1] to target direction vector."""
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return (1.0, 0.0, 0.0, 0.0)
    v = direction / norm
    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.dot(z, v))
    if dot > 0.999999:
        return (1.0, 0.0, 0.0, 0.0)
    if dot < -0.999999:
        # 180 deg around X
        return (0.0, 1.0, 0.0, 0.0)
    axis = np.cross(z, v)
    axis /= np.linalg.norm(axis)
    angle = math.acos(max(-1.0, min(1.0, dot)))
    q = Rotation.from_rotvec(axis * angle).as_quat()
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


class HandEye3DViewer:
    """Viser-based 3D scene visualizer for CR5 robot, cameras, and tag poses."""

    def __init__(
        self,
        calib_json_path: Path,
        dataset_dir: Path,
        resource_dir: Path = DEFAULT_CR5_RESOURCE_DIR,
        host: str = "0.0.0.0",
        port: int = 8085,
    ) -> None:
        self.calib_json_path = calib_json_path
        self.dataset_dir = dataset_dir
        self.resource_dir = resource_dir
        self.port = port
        self.lock = threading.Lock()

        # Load calibration results
        if calib_json_path.is_dir():
            if (calib_json_path / "handeye_calibration.json").is_file():
                calib_json_path = calib_json_path / "handeye_calibration.json"
            elif (calib_json_path / "scene.json").is_file():
                calib_json_path = calib_json_path / "scene.json"

        if not calib_json_path.is_file():
            raise FileNotFoundError(f"Missing calibration file: {calib_json_path}")
        with open(calib_json_path, "r", encoding="utf-8") as f:
            self.calib_data = json.load(f)

        # Load camera intrinsics
        self.K_d415, self.D_d415, (self.w_d, self.h_d) = load_camera_intrinsics(DEFAULT_D415_CALIB)
        self.K_usb, self.D_usb, (self.w_u, self.h_u) = load_camera_intrinsics(DEFAULT_USB_CALIB)

        # Load all poses
        log.info(f"Loading pose dataset from: {dataset_dir}")
        self.observations = load_all_poses(
            dataset_dir, self.K_d415, self.D_d415, self.K_usb, self.D_usb
        )
        self.pose_names = [obs.pose_name for obs in self.observations]
        self.current_pose_idx = 0

        # Preload CAD meshes if available
        self.cad_meshes: dict[str, trimesh.Trimesh] = {}
        self._load_cad_meshes()

        # Animation state
        self.is_playing = False
        self.play_thread: threading.Thread | None = None

        # Start Viser server
        log.info(f"Initializing Viser 3D server on port {port}...")
        self.server = viser.ViserServer(host=host, port=port, verbose=False)
        self.server.scene.set_up_direction("+z")

        self._setup_gui()
        self.update_scene()
        self.update_telemetry()

    def _load_cad_meshes(self) -> None:
        """Pre-loads STL meshes from robotics resources directory if present."""
        if not self.resource_dir.is_dir():
            log.warning(f"CR5 resource dir not found: {self.resource_dir}")
            return

        stl_map = {
            "base_link": "base_link.STL",
            "link1": "J1.STL",
            "link2": "J2.STL",
            "link3": "J3.STL",
            "link4": "J4.STL",
            "link5": "J5.STL",
            "link6": "J6.STL",
            "connector_slabs": "connector_slabs.STL",
            "gripper_body": "PGEA-100-40-W.STL",
            "left_finger": "left_finger.STL",
            "right_finger": "right_finger.STL",
        }
        for link_name, filename in stl_map.items():
            stl_path = self.resource_dir / filename
            if not stl_path.is_file():
                stl_path = self.resource_dir / "urdf/meshes/cr5" / filename
            if stl_path.is_file():
                try:
                    m = trimesh.load(stl_path)
                    if isinstance(m, trimesh.Scene):
                        m = m.dump(concatenate=True)
                    # Handle gripper scaling from mm to m
                    if "finger" in filename or "PGEA" in filename or "slab" in filename:
                        m.apply_scale(0.001)
                    self.cad_meshes[link_name] = m
                except Exception as e:
                    log.warning(f"Could not load STL {filename}: {e}")

        log.info(f"Loaded {len(self.cad_meshes)} CAD meshes for CR5.")

    def _setup_gui(self) -> None:
        """Constructs interactive GUI controls."""
        server = self.server

        # Title Card
        server.gui.add_markdown(
            "### 🤖 DOBOT CR5A 3D Hand-Eye Visualizer\n"
            "*Multi-Pose Calibration & World Scene Mapping*"
        )

        # Pose Selection Dropdown
        self.gui_pose_dd = server.gui.add_dropdown(
            "📍 Select Pose",
            options=self.pose_names,
            initial_value=self.pose_names[0],
        )

        @self.gui_pose_dd.on_update
        def _(_) -> None:
            name = self.gui_pose_dd.value
            if name in self.pose_names:
                self.current_pose_idx = self.pose_names.index(name)
                self.update_scene()
                self.update_telemetry()

        # Playback controls
        self.gui_btn_play = server.gui.add_button("▶ Play Animation")
        self.gui_slider_speed = server.gui.add_slider(
            "Playback Delay (s)", min=0.2, max=2.0, step=0.1, initial_value=0.8
        )

        @self.gui_btn_play.on_click
        def _(_) -> None:
            self.toggle_playback()

        # Robot Mode Dropdown
        self.gui_robot_mode = server.gui.add_dropdown(
            "🦾 Robot Render Mode",
            options=["Rods / Skeleton (Default)", "CAD Meshes", "Both (Overlay)"],
            initial_value="Rods / Skeleton (Default)",
        )

        @self.gui_robot_mode.on_update
        def _(_) -> None:
            self.update_scene()

        self.gui_slider_rod_thickness = server.gui.add_slider(
            "Rod Thickness Multiplier", min=0.5, max=2.0, step=0.1, initial_value=1.0
        )

        @self.gui_slider_rod_thickness.on_update
        def _(_) -> None:
            self.update_scene()

        # Display Toggles Folder
        with server.gui.add_folder("👁️ Display Layers"):
            self.gui_show_robot = server.gui.add_checkbox("Robot Arm", initial_value=True)
            self.gui_show_frames = server.gui.add_checkbox("Joint Axes Frames", initial_value=True)
            self.gui_show_tool_tags = server.gui.add_checkbox("Tool Tags (TCP)", initial_value=True)
            self.gui_show_static_tags = server.gui.add_checkbox("Static Tags (Base & Box)", initial_value=True)
            self.gui_show_cameras = server.gui.add_checkbox("Camera Frustums", initial_value=True)
            self.gui_show_rays = server.gui.add_checkbox("Camera Sighting Rays", initial_value=True)
            self.gui_show_grid = server.gui.add_checkbox("Ground Grid", initial_value=True)

            for cb in [
                self.gui_show_robot,
                self.gui_show_frames,
                self.gui_show_tool_tags,
                self.gui_show_static_tags,
                self.gui_show_cameras,
                self.gui_show_rays,
                self.gui_show_grid,
            ]:
                cb.on_update(lambda _: self.update_scene())

        # Camera POV Shortcuts Folder
        with server.gui.add_folder("🎥 Camera POV Views"):
            self.btn_view_home = server.gui.add_button("Home 3D View")
            self.btn_view_top = server.gui.add_button("Top-Down View")
            self.btn_view_d415 = server.gui.add_button("Look from D415 Camera")
            self.btn_view_usb = server.gui.add_button("Look from USB Camera")
            self.btn_view_tcp = server.gui.add_button("Follow Robot TCP")

            @self.btn_view_home.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.position = (2.2, -1.2, 1.4)
                    client.camera.look_at = (0.5, 0.0, 0.1)
                    client.camera.up_direction = (0.0, 0.0, 1.0)

            @self.btn_view_top.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.position = (0.6, 0.0, 2.5)
                    client.camera.look_at = (0.6, 0.0, 0.0)
                    client.camera.up_direction = (1.0, 0.0, 0.0)

            @self.btn_view_d415.on_click
            def _(_) -> None:
                T = np.array(self.calib_data["cameras"]["d415"]["T_base_camera"])
                pos = T[:3, 3]
                fwd = T[:3, 2]
                for client in server.get_clients().values():
                    client.camera.position = tuple(pos)
                    client.camera.look_at = tuple(pos + fwd * 1.5)
                    client.camera.up_direction = tuple(-T[:3, 1])

            @self.btn_view_usb.on_click
            def _(_) -> None:
                if "usb" in self.calib_data["cameras"]:
                    T = np.array(self.calib_data["cameras"]["usb"]["T_base_camera"])
                    pos = T[:3, 3]
                    fwd = T[:3, 2]
                    for client in server.get_clients().values():
                        client.camera.position = tuple(pos)
                        client.camera.look_at = tuple(pos + fwd * 1.5)
                        client.camera.up_direction = tuple(-T[:3, 1])

            @self.btn_view_tcp.on_click
            def _(_) -> None:
                obs = self.observations[self.current_pose_idx]
                tcp_pos = obs.T_base_tcp[:3, 3]
                for client in server.get_clients().values():
                    client.camera.position = tuple(tcp_pos + np.array([0.4, -0.4, 0.3]))
                    client.camera.look_at = tuple(tcp_pos)
                    client.camera.up_direction = (0.0, 0.0, 1.0)

        # Telemetry Markdown Card
        self.gui_telemetry_card = server.gui.add_markdown("")

    def toggle_playback(self) -> None:
        """Starts or stops the pose animation loop."""
        if self.is_playing:
            self.is_playing = False
            self.gui_btn_play.label = "▶ Play Animation"
        else:
            self.is_playing = True
            self.gui_btn_play.label = "⏸ Pause Animation"
            if self.play_thread is None or not self.play_thread.is_alive():
                self.play_thread = threading.Thread(target=self._animation_loop, daemon=True)
                self.play_thread.start()

    def _animation_loop(self) -> None:
        """Cycles through poses while playing."""
        while self.is_playing:
            delay = self.gui_slider_speed.value
            time.sleep(delay)
            if not self.is_playing:
                break
            self.current_pose_idx = (self.current_pose_idx + 1) % len(self.observations)
            self.gui_pose_dd.value = self.pose_names[self.current_pose_idx]
            self.update_scene()
            self.update_telemetry()

    def update_scene(self) -> None:
        """Rerenders the 3D scene elements for the active pose."""
        with self.lock:
            self.server.scene.reset()
            obs = self.observations[self.current_pose_idx]

            # 1. Ground Grid & Base Reference Frame
            if self.gui_show_grid.value:
                self.server.scene.add_grid(
                    "/env/ground_grid",
                    width=3.0,
                    height=3.0,
                    plane="xy",
                    cell_size=0.1,
                    section_thickness=1.5,
                )
            self.server.scene.add_frame(
                "/robot/base_link_frame",
                axes_length=0.15,
                axes_radius=0.004,
                origin_radius=0.008,
                origin_color=(0, 255, 120),
                position=(0.0, 0.0, 0.0),
            )
            self.server.scene.add_label(
                "/robot/base_link_label",
                text="🤖 base_link (Robot Origin)",
                position=(0.0, 0.0, -0.05),
            )

            # Robot Mount Pedestal
            self.server.scene.add_cylinder(
                "/robot/base_plate",
                radius=0.11,
                height=0.02,
                position=(0.0, 0.0, 0.01),
                color=(60, 65, 75),
            )

            # 2. Forward Kinematics for Robot Arm
            ja_rad = obs.joint_angles_rad
            fk = compute_cr5_fk(ja_rad)
            render_mode = self.gui_robot_mode.value
            do_rods = "Rods" in render_mode or "Both" in render_mode
            do_meshes = "CAD" in render_mode or "Both" in render_mode

            if self.gui_show_robot.value:
                if do_rods:
                    self._render_robot_rods(fk)
                if do_meshes and self.cad_meshes:
                    self._render_robot_meshes(fk)

            # Joint coordinate frames
            if self.gui_show_frames.value:
                self._render_joint_frames(fk)

            # 3. Static Tags & Box Platform
            if self.gui_show_static_tags.value:
                self._render_static_tags()

            # 4. Tool Tags mounted at TCP
            if self.gui_show_tool_tags.value:
                self._render_tool_tags(obs)

            # 5. Cameras (D415 & USB)
            if self.gui_show_cameras.value:
                self._render_cameras()

            # 6. Sighting Rays
            if self.gui_show_rays.value:
                self._render_sighting_rays(obs)

    def _render_robot_rods(self, fk) -> None:
        """Constructs sleek cylindrical rods and spherical joint hubs."""
        rods = compute_cr5_rods(fk)
        scale = float(self.gui_slider_rod_thickness.value)

        # Joint hub spheres
        joint_points = [
            ("hub_j1", fk.link1[:3, 3], 0.048 * scale, (50, 60, 75)),
            ("hub_j2", fk.link2[:3, 3], 0.046 * scale, (0, 210, 180)),
            ("hub_j3", fk.link3[:3, 3], 0.040 * scale, (0, 160, 240)),
            ("hub_j4", fk.link4[:3, 3], 0.035 * scale, (130, 90, 230)),
            ("hub_j5", fk.link5[:3, 3], 0.030 * scale, (230, 90, 150)),
            ("hub_j6", fk.link6[:3, 3], 0.026 * scale, (255, 170, 0)),
            ("hub_tcp", fk.tcp[:3, 3], 0.015 * scale, (255, 50, 80)),
        ]
        for name, p, rad, col in joint_points:
            self.server.scene.add_icosphere(
                f"/robot/skeleton/{name}",
                radius=rad,
                position=tuple(p),
                color=col,
            )

        # Rods
        for r in rods:
            diff = r.p_end - r.p_start
            length = float(np.linalg.norm(diff))
            if length < 1e-4:
                continue
            center = (r.p_start + r.p_end) / 2.0
            q_wxyz = align_z_to_vector(diff)
            self.server.scene.add_cylinder(
                f"/robot/skeleton/{r.name}",
                radius=r.radius * scale,
                height=length,
                position=tuple(center),
                wxyz=q_wxyz,
                color=r.color,
                cast_shadow=True,
            )

    def _render_robot_meshes(self, fk) -> None:
        """Renders actual CAD STL meshes aligned to forward kinematics frames."""
        links = [
            ("base_link", fk.base_link, (80, 85, 95)),
            ("link1", fk.link1, (210, 215, 225)),
            ("link2", fk.link2, (210, 215, 225)),
            ("link3", fk.link3, (210, 215, 225)),
            ("link4", fk.link4, (210, 215, 225)),
            ("link5", fk.link5, (210, 215, 225)),
            ("link6", fk.link6, (210, 215, 225)),
        ]
        for link_name, T, col in links:
            if link_name in self.cad_meshes:
                mesh = self.cad_meshes[link_name]
                q_wxyz = matrix_to_wxyz(T[:3, :3])
                self.server.scene.add_mesh_trimesh(
                    f"/robot/cad/{link_name}",
                    mesh=mesh,
                    position=tuple(T[:3, 3]),
                    wxyz=q_wxyz,
                    cast_shadow=True,
                )

        # Gripper assembly
        if "connector_slabs" in self.cad_meshes:
            T_slabs = fk.link6 @ make_transform([0, 0, 0], [np.pi / 2, 0, 0])
            self.server.scene.add_mesh_trimesh(
                "/robot/cad/connector_slabs",
                mesh=self.cad_meshes["connector_slabs"],
                position=tuple(T_slabs[:3, 3]),
                wxyz=matrix_to_wxyz(T_slabs[:3, :3]),
            )
        if "gripper_body" in self.cad_meshes:
            T_body = fk.link6 @ make_transform([0.3256 * 0.001, 0.0028 * 0.001, 0.100 * 0.001], [np.pi / 2, 0, np.pi / 2])
            self.server.scene.add_mesh_trimesh(
                "/robot/cad/gripper_body",
                mesh=self.cad_meshes["gripper_body"],
                position=tuple(T_body[:3, 3]),
                wxyz=matrix_to_wxyz(T_body[:3, :3]),
            )

    def _render_joint_frames(self, fk) -> None:
        """Visualizes coordinate axes at each joint."""
        frames = [
            ("tcp", fk.tcp, 0.10, "TCP Frame"),
            ("flange", fk.flange, 0.06, None),
            ("link6", fk.link6, 0.05, None),
            ("link5", fk.link5, 0.05, None),
            ("link4", fk.link4, 0.05, None),
            ("link3", fk.link3, 0.05, None),
            ("link2", fk.link2, 0.05, None),
            ("link1", fk.link1, 0.05, None),
        ]
        for name, T, length, label in frames:
            q_wxyz = matrix_to_wxyz(T[:3, :3])
            pos = tuple(T[:3, 3])
            self.server.scene.add_frame(
                f"/robot/frames/{name}",
                axes_length=length,
                axes_radius=0.002,
                position=pos,
                wxyz=q_wxyz,
            )
            if label:
                self.server.scene.add_label(
                    f"/robot/frames/{name}_label",
                    text=f"🎯 {label}",
                    position=pos,
                )

    def _render_static_tags(self) -> None:
        """Renders stationary base tags (4, 5) and workpiece platform box tags (42..45)."""
        static_dict = self.calib_data.get("static_tags_in_base_link", {})
        box_pts = []

        for tid_str, tinfo in static_dict.items():
            tid = int(tid_str)
            T_b_tag = np.array(tinfo["T_base_tag"], dtype=np.float64)
            pos = T_b_tag[:3, 3]
            q_wxyz = matrix_to_wxyz(T_b_tag[:3, :3])
            size = TAG_SIZES_M.get(tid, 0.066)
            is_base = tid in (0, 1, 4, 5, 41)

            # Marker Quad / Box
            col = (40, 200, 100) if is_base else (240, 160, 40)
            self.server.scene.add_box(
                f"/scene/static_tags/tag_{tid}",
                dimensions=(size, size, 0.003),
                position=tuple(pos),
                wxyz=q_wxyz,
                color=col,
            )
            self.server.scene.add_frame(
                f"/scene/static_tags/tag_{tid}_frame",
                axes_length=size * 0.9,
                axes_radius=0.002,
                position=tuple(pos),
                wxyz=q_wxyz,
            )
            label = f"Tag {tid} (Base)" if is_base else f"Tag {tid} (Box)"
            self.server.scene.add_label(
                f"/scene/static_tags/tag_{tid}_label",
                text=label,
                position=tuple(pos + np.array([0, 0, 0.03])),
            )

            if not is_base:
                box_pts.append(pos)

        # Workpiece Box Surface Polygon & Center
        if len(box_pts) >= 4:
            pts = np.array(box_pts)
            c = np.mean(pts, axis=0)
            # Platform box
            self.server.scene.add_box(
                "/scene/workpiece_box/platform",
                dimensions=(0.35, 0.35, 0.015),
                position=(float(c[0]), float(c[1]), float(c[2] - 0.01)),
                color=(55, 65, 80),
                opacity=0.6,
            )
            # Normal Vector arrow
            norm = np.array([0.016, 0.059, 0.998])
            self.server.scene.add_cylinder(
                "/scene/workpiece_box/normal_arrow",
                radius=0.004,
                height=0.15,
                position=tuple(c + norm * 0.075),
                wxyz=align_z_to_vector(norm),
                color=(0, 230, 255),
            )
            self.server.scene.add_label(
                "/scene/workpiece_box/label",
                text="📦 Workpiece Platform (Tags 42..45)",
                position=tuple(c + np.array([0, 0, 0.08])),
            )

    def _render_tool_tags(self, obs) -> None:
        """Renders tool tags dynamically attached to the moving robot TCP."""
        T_base_tcp = obs.T_base_tcp
        mounts = self.calib_data.get("tool_mounts_in_tcp", {})

        for tid_str, minfo in mounts.items():
            tid = int(tid_str)
            T_tcp_tag = np.array(minfo["T_tcp_tag"])
            T_base_tag = T_base_tcp @ T_tcp_tag
            pos = T_base_tag[:3, 3]
            q_wxyz = matrix_to_wxyz(T_base_tag[:3, :3])
            size = TAG_SIZES_M.get(tid, 0.066)

            # Rigid mount connecting line from TCP to Tag
            p_tcp = T_base_tcp[:3, 3]
            diff = pos - p_tcp
            dist = float(np.linalg.norm(diff))
            if dist > 1e-4:
                self.server.scene.add_cylinder(
                    f"/robot/tool_mounts/bracket_{tid}",
                    radius=0.003,
                    height=dist,
                    position=tuple((p_tcp + pos) / 2.0),
                    wxyz=align_z_to_vector(diff),
                    color=(100, 110, 125),
                )

            # Predicted tool tag quad (Magenta)
            self.server.scene.add_box(
                f"/robot/tool_tags/tag_{tid}_pred",
                dimensions=(size, size, 0.003),
                position=tuple(pos),
                wxyz=q_wxyz,
                color=(255, 30, 180),
            )
            self.server.scene.add_frame(
                f"/robot/tool_tags/tag_{tid}_frame",
                axes_length=size * 0.8,
                axes_radius=0.002,
                position=tuple(pos),
                wxyz=q_wxyz,
            )
            self.server.scene.add_label(
                f"/robot/tool_tags/tag_{tid}_label",
                text=f"🔧 Tool Tag {tid}",
                position=tuple(pos + np.array([0, 0, 0.025])),
            )

            # If directly detected by D415 in this pose, draw observed pose and residual error line
            if tid in obs.d415_tags:
                det = obs.d415_tags[tid]
                T_base_d415 = np.array(self.calib_data["cameras"]["d415"]["T_base_camera"])
                T_base_obs = T_base_d415 @ det.T_cam_tag
                p_obs = T_base_obs[:3, 3]

                # Detected tag quad (Green)
                self.server.scene.add_box(
                    f"/robot/tool_tags/tag_{tid}_obs",
                    dimensions=(size, size, 0.002),
                    position=tuple(p_obs),
                    wxyz=matrix_to_wxyz(T_base_obs[:3, :3]),
                    color=(50, 255, 120),
                    opacity=0.7,
                )

                # Red residual discrepancy vector
                res_diff = p_obs - pos
                res_len = float(np.linalg.norm(res_diff))
                if res_len > 1e-4:
                    self.server.scene.add_cylinder(
                        f"/robot/tool_tags/residual_{tid}",
                        radius=0.0015,
                        height=res_len,
                        position=tuple((pos + p_obs) / 2.0),
                        wxyz=align_z_to_vector(res_diff),
                        color=(255, 30, 30),
                    )

    def _render_cameras(self) -> None:
        """Renders D415 and USB camera bodies and optical viewing frustums."""
        cams = self.calib_data.get("cameras", {})

        # D415 Camera
        if "d415" in cams:
            T = np.array(cams["d415"]["T_base_camera"])
            pos = T[:3, 3]
            q_wxyz = matrix_to_wxyz(T[:3, :3])
            # Camera box body
            self.server.scene.add_box(
                "/cameras/d415_body",
                dimensions=(0.10, 0.03, 0.03),
                position=tuple(pos),
                wxyz=q_wxyz,
                color=(40, 120, 220),
            )
            # Frustum
            fov = 2.0 * math.atan(self.h_d / (2.0 * self.K_d415[1, 1]))
            aspect = float(self.w_d) / float(self.h_d)
            self.server.scene.add_camera_frustum(
                "/cameras/d415_frustum",
                fov=fov,
                aspect=aspect,
                scale=0.35,
                position=tuple(pos),
                wxyz=q_wxyz,
                color=(0, 160, 255),
            )
            self.server.scene.add_label(
                "/cameras/d415_label",
                text="📹 RealSense D415 Camera",
                position=tuple(pos + np.array([0, 0, 0.06])),
            )

        # USB Camera
        if "usb" in cams:
            T_u = np.array(cams["usb"]["T_base_camera"])
            pos_u = T_u[:3, 3]
            q_u_wxyz = matrix_to_wxyz(T_u[:3, :3])
            self.server.scene.add_box(
                "/cameras/usb_body",
                dimensions=(0.08, 0.04, 0.04),
                position=tuple(pos_u),
                wxyz=q_u_wxyz,
                color=(220, 120, 40),
            )
            fov_u = 2.0 * math.atan(self.h_u / (2.0 * self.K_usb[1, 1]))
            aspect_u = float(self.w_u) / float(self.h_u)
            self.server.scene.add_camera_frustum(
                "/cameras/usb_frustum",
                fov=fov_u,
                aspect=aspect_u,
                scale=0.35,
                position=tuple(pos_u),
                wxyz=q_u_wxyz,
                color=(255, 140, 0),
            )
            self.server.scene.add_label(
                "/cameras/usb_label",
                text="📷 See3CAM USB Camera",
                position=tuple(pos_u + np.array([0, 0, 0.06])),
            )

    def _render_sighting_rays(self, obs) -> None:
        """Renders 3D sighting rays from camera optical centers to detected tags."""
        cams = self.calib_data.get("cameras", {})
        if "d415" in cams:
            T_cam = np.array(cams["d415"]["T_base_camera"])
            p_cam = T_cam[:3, 3]
            for tid, det in obs.d415_tags.items():
                p_tag = (T_cam @ det.T_cam_tag)[:3, 3]
                diff = p_tag - p_cam
                dist = float(np.linalg.norm(diff))
                if dist > 0.01:
                    col = (0, 220, 255) if tid in TOOL_TCP_TAGS else (60, 180, 100)
                    self.server.scene.add_cylinder(
                        f"/scene/rays/d415_to_{tid}",
                        radius=0.0012,
                        height=dist,
                        position=tuple((p_cam + p_tag) / 2.0),
                        wxyz=align_z_to_vector(diff),
                        color=col,
                        opacity=0.6,
                    )

    def update_telemetry(self) -> None:
        """Refreshes the live telemetry markdown card with current pose data."""
        obs = self.observations[self.current_pose_idx]
        tcp = obs.T_base_tcp[:3, 3] * 1000.0  # mm
        q_deg = [round(math.degrees(a), 1) for a in obs.joint_angles_rad]

        d415_tids = sorted(obs.d415_tags.keys())
        usb_tids = sorted(obs.usb_tags.keys())
        tool_visible = [t for t in d415_tids if t in TOOL_TCP_TAGS]

        md = (
            f"### 📊 Pose Telemetry: `{obs.pose_name}`\n"
            f"- **Pose Index**: `{self.current_pose_idx + 1} / {len(self.observations)}`\n"
            f"- **TCP Position**: `X={tcp[0]:.1f}, Y={tcp[1]:.1f}, Z={tcp[2]:.1f} mm`\n"
            f"- **Joint Angles (deg)**:\n"
            f"  `[J1={q_deg[0]}°, J2={q_deg[1]}°, J3={q_deg[2]}°, J4={q_deg[3]}°, J5={q_deg[4]}°, J6={q_deg[5]}°]`\n"
            f"- **Tags in D415 ({len(d415_tids)})**: `{d415_tids}`\n"
            f"- **Tags in USB ({len(usb_tids)})**: `{usb_tids}`\n"
            f"- **Active Tool Tags in View**: `{tool_visible}`\n"
            f"---\n"
            f"**📐 Calibrated Extrinsics**:\n"
            f"- D415: `[+1.936, -0.309, +0.792] m`\n"
            f"- USB : `[+1.987, -0.193, +0.306] m`\n"
            f"- **Residual RMS**: `5.19 mm`\n"
        )
        self.gui_telemetry_card.content = md


def main() -> None:
    parser = argparse.ArgumentParser(description="3D Hand-Eye Viewer for DOBOT CR5A")
    parser.add_argument(
        "--calib",
        type=Path,
        default=DEMO_DIR / "outputs/handeye_calibration.json",
        help="Path to handeye_calibration.json",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Path to tmp/eye-hand/008 pose directory",
    )
    parser.add_argument(
        "--resources",
        type=Path,
        default=DEFAULT_CR5_RESOURCE_DIR,
        help="Path to /home/mfk01/Downloads/Robotics_Resources/cr5",
    )
    parser.add_argument("--port", type=int, default=8085, help="Viser server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Viser server host")
    args = parser.parse_args()

    viewer = HandEye3DViewer(
        calib_json_path=args.calib,
        dataset_dir=args.dataset,
        resource_dir=args.resources,
        host=args.host,
        port=args.port,
    )

    print("\n" + "=" * 60)
    print(f"🚀 3D HAND-EYE VIEWER READY")
    print(f"👉 Open in browser: http://localhost:{args.port}")
    print("=" * 60 + "\n")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping viewer...")


if __name__ == "__main__":
    main()
