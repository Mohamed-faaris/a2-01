#!/usr/bin/env python3
"""Interactive 3D WebGL Viewer for Two-Camera Planes & Cube Tracking Results.

Features:
- Powered by Viser (Three.js WebGL in browser with orbit, pan, zoom, dark theme).
- Loads and live-watches ``live_tracking_results.json`` from any session/experiment.
- Automatically renders in Base Tag 0 Reference Frame:
  * Base Tag 0 origin (134 mm) + RGB world coordinate frame + ground grid.
  * Plane A (Tags 40, 41) baseline connection line, endpoints, midpoint.
  * Plane B (Tags 42..45) platform surface polygon, corner tags, 3D normal vector.
  * 6-Faced Cube (Tags 20..25) 3D rigid box (41 mm) at centroid + face quads.
  * Custom 'H' frame origin and rotated axes (+2cm X, +5cm Z).
  * Camera 0 and Camera 1 3D frustums and optical centers in the Base frame.
- Interactive GUI:
  * Dropdown to switch between all discovered sessions in ``outputs/``.
  * "Live Auto-Watch" toggle (refreshes automatically as tracking updates).
  * "Text Mode" dropdown: ``max`` / ``lite`` / ``none`` label density in 3D.
  * "🎥 View Shortcuts" buttons: Home / Top / Front / Side / Cube, plus
    "Cam 0 POV" and "Cam 1 POV" to look through the tracked camera frames.
  * "📸 Save Snapshot (JSON + PNG)" button: exports both the JSON state and a
    high-res client 3D render image with a single click.
  * Live metric telemetry cards (Cube centroid, Plane A distance, Plane B normal, H frame).

Usage:
    uv run src/demos/2-cam-planes-cube-tracker/viewer3d.py
    uv run src/demos/2-cam-planes-cube-tracker/viewer3d.py --input outputs/live_tracking_results.json
    uv run src/demos/2-cam-planes-cube-tracker/viewer3d.py --port 8080
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import imageio.v3 as iio
import numpy as np
from scipy.spatial.transform import Rotation
import viser

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("tracker-viewer3d")


def matrix_to_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to (w, x, y, z) quaternion for Viser."""
    try:
        q = Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_quat()  # [x, y, z, w]
        return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
    except Exception:
        return (1.0, 0.0, 0.0, 0.0)


def scan_available_results_files(root: Path) -> list[Path]:
    """Scan outputs directory for all available live_tracking_results.json files."""
    files = []
    if not root.is_dir():
        return files
    # Check root directly
    root_json = root / "live_tracking_results.json"
    if root_json.is_file():
        files.append(root_json)
    # Search subdirectories (sessions and experiments)
    for p in sorted(root.rglob("live_tracking_results.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        if p not in files:
            files.append(p)
    return files


class Tracking3DViewer:
    """Interactive 3D Viser Server for live tracking results."""

    def __init__(self, initial_json_path: Path | None = None, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.outputs_dir = REPO_ROOT / "outputs"
        self.active_json_path = initial_json_path or self._find_best_initial_json()
        self.last_mtime: float = 0.0
        self.current_data: dict[str, Any] = {}
        self.lock = threading.Lock()
        # Cached tracked camera viewpoints: cam_id -> (eye_m, look_at_m, up)
        self.cam_views: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        # Start Viser server
        log.info(f"Starting Viser 3D Web Server on http://localhost:{self.port} ...")
        self.server = viser.ViserServer(host=self.host, port=self.port)

        # Set up GUI controls
        self._setup_gui()

        # Initial render
        self.reload_data(force=True)

        # Start live watcher thread
        self.running = True
        self.watcher_thread = threading.Thread(target=self._live_watch_loop, daemon=True)
        self.watcher_thread.start()

    def _find_best_initial_json(self) -> Path:
        available = scan_available_results_files(self.outputs_dir)
        if available:
            return available[0]
        return self.outputs_dir / "live_tracking_results.json"

    def _setup_gui(self):
        # 1. Header Banner
        self.server.gui.add_markdown(
            "### 🛰️ **3D Tracking Results Viewer**\n"
            "*Unified Base Tag 0 Frame Visualization*"
        )

        # 2. File Selection & Watcher
        available_files = scan_available_results_files(self.outputs_dir)
        dropdown_options = [str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p) for p in available_files]
        if not dropdown_options:
            dropdown_options = ["outputs/live_tracking_results.json"]

        curr_rel = str(self.active_json_path.relative_to(REPO_ROOT)) if self.active_json_path and self.active_json_path.is_relative_to(REPO_ROOT) else str(self.active_json_path)
        if curr_rel not in dropdown_options:
            dropdown_options.insert(0, curr_rel)

        self.file_dropdown = self.server.gui.add_dropdown(
            "Selected File",
            options=dropdown_options,
            initial_value=curr_rel
        )

        @self.file_dropdown.on_update
        def _(val):
            new_path = REPO_ROOT / self.file_dropdown.value
            if new_path.is_file():
                self.active_json_path = new_path
                self.reload_data(force=True)

        self.btn_rescan = self.server.gui.add_button("🔄 Re-scan Outputs Folder")
        @self.btn_rescan.on_click
        def _(_):
            new_opts = [str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)
                        for p in scan_available_results_files(self.outputs_dir)]
            if new_opts:
                self.file_dropdown.options = new_opts
                self.server.get_clients()

        self.cb_live_watch = self.server.gui.add_checkbox("Live Auto-Reload", initial_value=True)

        # Text labels density: max (all), lite (key labels only), none (hidden)
        self.dd_text_mode = self.server.gui.add_dropdown(
            "Text Mode",
            options=["max", "lite", "none"],
            initial_value="max",
        )

        @self.dd_text_mode.on_update
        def _(_):
            self.render_scene_3d(self.current_data)

        # View shortcuts: snap the browser camera to preset / tracked camera views
        self.server.gui.add_divider()
        self.server.gui.add_markdown("**🎥 View Shortcuts**")
        for label, fn in (
            ("🏠 Home", self.view_home),
            ("🔼 Top", self.view_top),
            ("➡️ Front", self.view_front),
            ("↔️ Side", self.view_side),
            ("💠 Cube", self.view_cube),
            ("📷 Cam 0 POV", self.view_camera_0),
            ("📷 Cam 1 POV", self.view_camera_1),
        ):
            btn_view = self.server.gui.add_button(label)

            @btn_view.on_click
            def _(event, _fn=fn):
                _fn(event.client)

        # 3. Snapshot Action Button
        self.server.gui.add_divider()
        self.btn_snapshot = self.server.gui.add_button("📸 Save Snapshot (JSON + PNG)")
        @self.btn_snapshot.on_click
        def _(event):
            self.save_snapshot_json_and_image(event.client)

        # 4. Telemetry Readout Cards
        self.server.gui.add_divider()
        self.telemetry_markdown = self.server.gui.add_markdown(
            "**Waiting for tracking telemetry...**"
        )

    def _live_watch_loop(self):
        """Continuously monitor active JSON file mtime for changes."""
        while self.running:
            try:
                if self.cb_live_watch.value and self.active_json_path and self.active_json_path.is_file():
                    mtime = self.active_json_path.stat().st_mtime
                    if mtime > self.last_mtime:
                        self.reload_data()
            except Exception as e:
                pass
            time.sleep(0.15)

    def reload_data(self, force: bool = False):
        if not self.active_json_path or not self.active_json_path.is_file():
            return

        try:
            mtime = self.active_json_path.stat().st_mtime
            if not force and mtime <= self.last_mtime:
                return

            with open(self.active_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            with self.lock:
                self.last_mtime = mtime
                self.current_data = data

            # Update 3D scene elements
            self.render_scene_3d(data)

            # Update Telemetry GUI card
            self.update_telemetry_card(data)

        except Exception as exc:
            log.warning(f"Error parsing tracking results: {exc}")

    def _add_label(self, name: str, text: str, position: Any, minor: bool = False) -> None:
        """Add a 3D scene text label, respecting the current ``Text Mode``.

        * ``max``  -> all labels (key + minor detail labels).
        * ``lite`` -> only key/summary labels (``minor=True`` labels hidden).
        * ``none`` -> no labels at all.
        """
        mode = self.dd_text_mode.value
        if mode == "none":
            return
        if mode == "lite" and minor:
            return
        self.server.scene.add_label(name, text=text, position=position)

    def render_scene_3d(self, data: dict[str, Any]):
        """Render all 3D features in the Base Tag 0 reference frame."""
        self.server.scene.reset()

        # 1. Base Tag 0 Ground Grid & Reference Frame
        self.server.scene.add_grid(
            "/base/ground_grid",
            width=1.2,
            height=1.2,
            plane="xy",
            cell_size=0.05,
            section_thickness=1.5,
        )

        # Base Tag 0 Marker Box (134 mm x 134 mm x 2 mm)
        self.server.scene.add_box(
            "/base/tag_0_marker",
            dimensions=(0.134, 0.134, 0.002),
            position=(0.0, 0.0, 0.001),
            color=(40, 180, 80),
            opacity=0.9,
        )
        self.server.scene.add_frame(
            "/base/origin_axes",
            axes_length=0.15,
            axes_radius=0.004,
            origin_radius=0.008,
            origin_color=(0, 255, 120),
            position=(0.0, 0.0, 0.0),
        )
        self._add_label(
            "/base/label",
            text="BASE ORIGIN (Tag 0, 134mm)",
            position=(0.0, -0.08, 0.02),
        )

        # 2. Plane A: Line / Pair (Tags 40, 41)
        pa = data.get("plane_a")
        if pa and "tag_40_mm" in pa and "tag_41_mm" in pa:
            p40 = np.asarray(pa["tag_40_mm"], dtype=np.float64) * 0.001
            p41 = np.asarray(pa["tag_41_mm"], dtype=np.float64) * 0.001
            mid = (p40 + p41) / 2.0
            dist_mm = float(pa.get("baseline_dist_mm", np.linalg.norm(p41 - p40) * 1000.0))

            # Segment
            self.server.scene.add_line_segments(
                "/plane_a/baseline",
                points=np.array([[p40, p41]]),
                colors=(0, 220, 255),
                thickness=0.005,
            )
            # Endpoints
            self.server.scene.add_box("/plane_a/tag_40", dimensions=(0.066, 0.066, 0.002), position=p40, color=(0, 220, 255))
            self.server.scene.add_box("/plane_a/tag_41", dimensions=(0.066, 0.066, 0.002), position=p41, color=(0, 220, 255))
            self._add_label("/plane_a/label_40", text=f"Tag 40\n[{p40[0]*1e3:+.0f}, {p40[1]*1e3:+.0f}, {p40[2]*1e3:+.0f}]mm", position=p40 + np.array([0, 0, 0.02]), minor=True)
            self._add_label("/plane_a/label_41", text=f"Tag 41\n[{p41[0]*1e3:+.0f}, {p41[1]*1e3:+.0f}, {p41[2]*1e3:+.0f}]mm", position=p41 + np.array([0, 0, 0.02]), minor=True)
            self._add_label("/plane_a/label_mid", text=f"Baseline: {dist_mm:.1f} mm", position=mid + np.array([0, 0, 0.03]))

        # 3. Plane B: Platform Quad (Tags 42..45)
        pb = data.get("plane_b")
        if pb and "center_mm" in pb and "normal" in pb:
            c_b = np.asarray(pb["center_mm"], dtype=np.float64) * 0.001
            normal = np.asarray(pb["normal"], dtype=np.float64)
            normal = normal / max(1e-6, np.linalg.norm(normal))
            rmse_mm = float(pb.get("coplanarity_rmse_mm", 0.0))

            # Normal Arrow
            self.server.scene.add_arrows(
                "/plane_b/normal_vector",
                points=np.array([[c_b, c_b + normal * 0.12]]),
                colors=(255, 140, 0),
                shaft_radius=0.004,
                head_radius=0.010,
                head_length=0.025,
            )

            # Corners and polygon
            centers_dict = pb.get("centers_mm") or pb.get("corners_mm") or {}
            corner_pts = []
            for tid_str, pt_mm in centers_dict.items():
                pt = np.asarray(pt_mm, dtype=np.float64) * 0.001
                corner_pts.append(pt)
                self.server.scene.add_box(
                    f"/plane_b/tag_{tid_str}",
                    dimensions=(0.066, 0.066, 0.002),
                    position=pt,
                    color=(255, 140, 0),
                    opacity=0.85,
                )
                self._add_label(f"/plane_b/label_{tid_str}", text=f"Tag {tid_str}", position=pt + np.array([0, 0, 0.02]), minor=True)

            # Quad outline
            if len(corner_pts) >= 3:
                # Build simple fan mesh for platform surface
                v_arr = np.array(corner_pts, dtype=np.float32)
                faces = []
                for i in range(1, len(corner_pts) - 1):
                    faces.append([0, i, i + 1])
                self.server.scene.add_mesh_simple(
                    "/plane_b/surface",
                    vertices=v_arr,
                    faces=np.array(faces, dtype=np.int32),
                    color=(255, 160, 20),
                    opacity=0.45,
                    side="double",
                )

            self._add_label(
                "/plane_b/center_label",
                text=f"PLANE B Normal: [{normal[0]:+.2f}, {normal[1]:+.2f}, {normal[2]:+.2f}]\nCoplanar RMSE: {rmse_mm:.1f}mm",
                position=c_b + np.array([0, 0, 0.04]),
            )

        # 4. 6-Faced AprilTag Cube (Tags 20..25)
        cube = data.get("cube_centroid")
        if cube and "xyz_mm" in cube:
            c_cube = np.asarray(cube["xyz_mm"], dtype=np.float64) * 0.001
            spread_mm = float(cube.get("residual_spread_mm", 0.0))
            faces = cube.get("visible_faces", [])

            # 41 mm rigid cube
            self.server.scene.add_box(
                "/cube/body",
                dimensions=(0.041, 0.041, 0.041),
                position=c_cube,
                color=(240, 50, 240),
                opacity=0.85,
            )
            self.server.scene.add_frame(
                "/cube/centroid_frame",
                axes_length=0.06,
                axes_radius=0.003,
                origin_radius=0.006,
                origin_color=(255, 0, 255),
                position=c_cube,
            )
            self._add_label(
                "/cube/label",
                text=f"CUBE CENTROID (41mm)\n[{c_cube[0]*1e3:+.1f}, {c_cube[1]*1e3:+.1f}, {c_cube[2]*1e3:+.1f}] mm\nFaces: {faces} (±{spread_mm:.1f}mm)",
                position=c_cube + np.array([0, 0, 0.035]),
            )

        # Cube detected face polygons if recorded
        cube_faces = data.get("cube_faces_mm") or {}
        for fid_str, corners_mm in cube_faces.items():
            if len(corners_mm) == 4:
                v_face = np.asarray(corners_mm, dtype=np.float32) * 0.001
                f_idx = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
                self.server.scene.add_mesh_simple(
                    f"/cube/face_{fid_str}",
                    vertices=v_face,
                    faces=f_idx,
                    color=(200, 30, 200),
                    opacity=0.7,
                    side="double",
                )

        # 5. Custom 'H' Frame (Rotated +2cm X, +5cm Z)
        h_frame = data.get("custom_h_frame")
        if h_frame and "origin_base_m" in h_frame and "R_custom_base" in h_frame:
            o_h = np.asarray(h_frame["origin_base_m"], dtype=np.float64)
            R_h = np.asarray(h_frame["R_custom_base"], dtype=np.float64)
            q_h = matrix_to_wxyz(R_h)

            self.server.scene.add_frame(
                "/custom_h_frame/axes",
                axes_length=0.10,
                axes_radius=0.004,
                origin_radius=0.007,
                origin_color=(255, 220, 0),
                position=o_h,
                wxyz=q_h,
            )
            cube_in_h = h_frame.get("cube_centroid_in_h_mm")
            h_text = f"CUSTOM 'H' FRAME (+2cm X, +5cm Z)\nOrigin: [{o_h[0]*1e3:+.1f}, {o_h[1]*1e3:+.1f}, {o_h[2]*1e3:+.1f}] mm"
            if cube_in_h:
                h_text += f"\nCube in H: [{cube_in_h[0]:+.1f}, {cube_in_h[1]:+.1f}, {cube_in_h[2]:+.1f}] mm"
            self._add_label(
                "/custom_h_frame/label",
                text=h_text,
                position=o_h + np.array([0, 0, 0.03]),
            )

        # 6. Camera Frustums in Base Frame
        self.cam_views.clear()
        cam_ext = data.get("cam_extrinsics") or {}
        for cid_str, ext in cam_ext.items():
            if "R_c_base" in ext and "t_c_base_m" in ext:
                R_c_b = np.asarray(ext["R_c_base"], dtype=np.float64)
                t_c_b = np.asarray(ext["t_c_base_m"], dtype=np.float64)
                # Optical center in Base: -R_c_b.T @ t_c_b
                cam_pos = -R_c_b.T @ t_c_b
                # Camera orientation in Base: R_c_b.T
                q_cam = matrix_to_wxyz(R_c_b.T)

                col = (0, 180, 255) if cid_str == "0" else (255, 100, 180)
                self.server.scene.add_camera_frustum(
                    f"/cameras/cam_{cid_str}",
                    fov=60.0,
                    aspect=1280.0 / 720.0,
                    scale=0.09,
                    color=col,
                    position=cam_pos,
                    wxyz=q_cam,
                )
                self._add_label(
                    f"/cameras/label_cam_{cid_str}",
                    text=f"Camera {cid_str}\n[{cam_pos[0]*1e3:+.0f}, {cam_pos[1]*1e3:+.0f}, {cam_pos[2]*1e3:+.0f}]mm",
                    position=cam_pos + np.array([0, 0, 0.02]),
                    minor=True,
                )

                # Cache a "look through this camera" viewpoint for the view
                # shortcut buttons. OpenCV convention: +Z forward, -Y up.
                R_base_cam = R_c_b.T
                look_dir = R_base_cam @ np.array([0.0, 0.0, 1.0])
                up_dir = R_base_cam @ np.array([0.0, -1.0, 0.0])
                self.cam_views[cid_str] = (
                    cam_pos.astype(np.float64),
                    (cam_pos + look_dir * 0.4).astype(np.float64),
                    up_dir.astype(np.float64),
                )

    # ------------------------------------------------------------------
    # View shortcuts: snap the browser camera to useful viewpoints.
    # ------------------------------------------------------------------
    def _fly_to_view(self, client: viser.ClientHandle | None, eye, target, up=(0.0, 0.0, 1.0)):
        """Move a client's viewport camera to the given eye/target/up pose."""
        if client is None:
            return
        eye = np.asarray(eye, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        up = np.asarray(up, dtype=np.float64)
        if np.linalg.norm(target - eye) < 1e-6:
            target = eye + np.array([0.0, 0.0, 0.1])
        try:
            with client.atomic():
                client.camera.position = eye
                client.camera.look_at = target
                client.camera.up_direction = up
        except Exception as exc:
            log.warning(f"Could not move camera: {exc}")

    def view_home(self, client: viser.ClientHandle | None = None):
        self._fly_to_view(client, (0.55, -0.55, 0.42), (0.0, 0.0, 0.05))

    def view_top(self, client: viser.ClientHandle | None = None):
        self._fly_to_view(client, (0.0, 0.0, 0.95), (0.0, 0.0, 0.0), up=(0.0, -1.0, 0.0))

    def view_front(self, client: viser.ClientHandle | None = None):
        self._fly_to_view(client, (0.0, -0.75, 0.18), (0.0, 0.0, 0.06))

    def view_side(self, client: viser.ClientHandle | None = None):
        self._fly_to_view(client, (0.75, 0.0, 0.18), (0.0, 0.0, 0.06))

    def view_cube(self, client: viser.ClientHandle | None = None):
        cube = (self.current_data or {}).get("cube_centroid") or {}
        if "xyz_mm" not in cube:
            self.view_home(client)
            return
        c = np.asarray(cube["xyz_mm"], dtype=np.float64) * 0.001
        self._fly_to_view(client, c + np.array([0.28, -0.28, 0.20]), c)

    def _view_tracked_camera(self, client: viser.ClientHandle | None, cid_str: str):
        """Fly the browser camera to the tracked pose of a physical camera."""
        view = self.cam_views.get(cid_str)
        if view is None:
            log.warning(f"No cached pose for Camera {cid_str}; using home view.")
            self.view_home(client)
            return
        eye, target, up = view
        # Pull back slightly so the scene is in front of the camera plane.
        dir_vec = target - eye
        self._fly_to_view(client, eye - dir_vec * 0.35, target, up=up)

    def view_camera_0(self, client: viser.ClientHandle | None = None):
        self._view_tracked_camera(client, "0")

    def view_camera_1(self, client: viser.ClientHandle | None = None):
        self._view_tracked_camera(client, "1")

    def update_telemetry_card(self, data: dict[str, Any]):
        """Render markdown card on GUI panel with live metric values."""
        ts = data.get("timestamp", time.time())
        iso_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

        b0 = "🟢 LOCKED" if data.get("cam0_has_base") else "⚪ SEARCHING"
        b1 = "🟢 LOCKED" if data.get("cam1_has_base") else "⚪ SEARCHING"

        lines = [
            f"**Last Update:** `{iso_str}`\n",
            f"| Camera | Base Tag 0 Status |",
            f"| :--- | :--- |",
            f"| **Camera 0** | {b0} |",
            f"| **Camera 1** | {b1} |\n",
        ]

        cube = data.get("cube_centroid")
        if cube:
            c = cube["xyz_mm"]
            lines += [
                f"**🎲 6-Faced Cube (Tags 20..25, 41mm):**",
                f"- Centroid $(X,Y,Z)$: `[{c[0]:+6.1f}, {c[1]:+6.1f}, {c[2]:+6.1f}] mm`",
                f"- Visible Faces: `{cube.get('visible_faces')}` | Spread: `±{cube.get('residual_spread_mm', 0):.1f} mm`\n",
            ]
        else:
            lines.append("**🎲 Cube Centroid:** Searching...\n")

        pa = data.get("plane_a")
        if pa:
            lines += [
                f"**📏 Plane A (Tags 40, 41):**",
                f"- Baseline Distance: `{pa.get('baseline_dist_mm', 0):.1f} mm`",
                f"- Midpoint $(X,Y,Z)$: `[{pa['center_mm'][0]:+6.1f}, {pa['center_mm'][1]:+6.1f}, {pa['center_mm'][2]:+6.1f}] mm`\n",
            ]

        pb = data.get("plane_b")
        if pb:
            n = pb.get("normal", [0, 0, 1])
            lines += [
                f"**📐 Plane B (Tags 42..45):**",
                f"- Normal: `[{n[0]:+5.2f}, {n[1]:+5.2f}, {n[2]:+5.2f}]`",
                f"- Center: `[{pb['center_mm'][0]:+6.1f}, {pb['center_mm'][1]:+6.1f}, {pb['center_mm'][2]:+6.1f}] mm`",
                f"- Coplanarity RMSE: `{pb.get('coplanarity_rmse_mm', 0):.2f} mm`\n",
            ]

        h = data.get("custom_h_frame")
        if h:
            o = h["origin_base_mm"]
            lines += [
                f"**⚡ Custom Baseline 'H' Frame:**",
                f"- Origin: `[{o[0]:+6.1f}, {o[1]:+6.1f}, {o[2]:+6.1f}] mm`",
            ]
            if "cube_centroid_in_h_mm" in h:
                ch = h["cube_centroid_in_h_mm"]
                lines.append(f"- Cube in H: `[{ch[0]:+6.1f}, {ch[1]:+6.1f}, {ch[2]:+6.1f}] mm`")

        self.telemetry_markdown.content = "\n".join(lines)

    def save_snapshot_json_and_image(self, client: viser.ClientHandle | None = None):
        """Export both JSON telemetry snapshot and 3D client render image."""
        stamp = time.strftime("%Y%m%d_%H%M%S")
        target_dir = self.active_json_path.parent if self.active_json_path else (self.outputs_dir / "snapshots")
        target_dir.mkdir(parents=True, exist_ok=True)

        json_out = target_dir / f"snapshot_{stamp}.json"
        img_out = target_dir / f"snapshot_{stamp}.png"

        # 1. Save JSON Snapshot
        with self.lock:
            snap_data = dict(self.current_data)
        snap_data["snapshot_export_time"] = datetime.now().isoformat()

        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(snap_data, f, indent=2)
        log.info(f"Saved snapshot JSON: {json_out}")

        # 2. Capture and Save 3D Render Image
        img_saved = False
        target_client = client
        if target_client is None:
            clients = self.server.get_clients()
            if clients:
                target_client = next(iter(clients.values()))

        if target_client is not None:
            try:
                # Capture WebGL render from client
                render_rgb = target_client.get_render(height=720, width=1280)
                iio.imwrite(img_out, render_rgb)
                img_saved = True
                log.info(f"Saved snapshot 3D image: {img_out}")
            except Exception as exc:
                log.warning(f"Could not grab client render: {exc}")

        msg = f"Saved snapshot JSON: {json_out.name}"
        if img_saved:
            msg += f" + 3D Render Image: {img_out.name}"

        if target_client is not None:
            target_client.add_notification(
                title="📸 Snapshot Saved!",
                body=f"Exported to {target_dir.name}/\n{json_out.name}\n{img_out.name if img_saved else ''}",
                color="green"
            )


def main():
    parser = argparse.ArgumentParser(description="Interactive 3D WebGL Viewer for Two-Camera Planes & Cube Tracking Results")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to live_tracking_results.json")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Viser HTTP port (default 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Viser server bind host (default 0.0.0.0)")
    args = parser.parse_args()

    in_path = Path(args.input) if args.input else None
    viewer = Tracking3DViewer(initial_json_path=in_path, host=args.host, port=args.port)

    log.info(f"3D Viewer ready! Open http://localhost:{args.port} in your browser to inspect.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        log.info("Shutting down 3D viewer.")
        viewer.running = False
        viewer.server.stop()


if __name__ == "__main__":
    main()
