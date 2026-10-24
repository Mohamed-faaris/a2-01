#!/usr/bin/env python3
"""Interactive 3D WebGL Viewer for Multi-Camera Planes & Cube Tracking Results.

Features:
- Powered by Viser (Three.js WebGL in browser with orbit, pan, zoom, dark theme).
- Loads and live-watches ``live_tracking_results.json`` from any session/experiment.
- Automatically renders in Base Tag 0 / base_link Reference Frame:
  * Base Tag 0 origin (134 mm) + RGB world coordinate frame + ground grid.
  * Plane A (Tags 40, 41) baseline connection line, endpoints, midpoint.
  * Plane B (Tags 42..45) platform surface polygon, corner tags, 3D normal vector.
  * 6-Faced Cube (Tags 20..25) 3D rigid box (41 mm) at centroid + face quads.
  * Custom 'H' frame origin and rotated axes (+2cm X, +5cm Z).
  * 3D camera frustums and optical centers for all N cameras in Base frame.
- Interactive GUI:
  * Dropdown to switch between all discovered sessions in ``outputs/``.
  * "Live Auto-Watch" toggle (refreshes automatically as tracking updates).
  * "Text Mode" dropdown: ``max`` / ``lite`` / ``none`` label density in 3D.
  * "🎥 View Shortcuts" buttons: Home / Top / Front / Side / Cube, plus
    dynamic camera POV buttons for each tracked camera.
  * "📸 Save Snapshot (JSON + PNG)" button.
  * Multi-Camera live metric telemetry card.

Usage:
    python3 src/demos/multi-cam-planes-cube-tracker/viewer3d.py
    python3 src/demos/multi-cam-planes-cube-tracker/viewer3d.py --input outputs/live_tracking_results.json
    python3 src/demos/multi-cam-planes-cube-tracker/viewer3d.py --port 8080
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
log = logging.getLogger("multi-cam-viewer3d")


def matrix_to_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to (w, x, y, z) quaternion for Viser."""
    try:
        q = Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_quat()
        return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
    except Exception:
        return (1.0, 0.0, 0.0, 0.0)


def scan_available_results_files(outputs_root: Path) -> list[Path]:
    """Scans outputs/ for all live_tracking_results.json files."""
    found = []
    if not outputs_root.is_dir():
        return found

    for p in outputs_root.rglob("live_tracking_results.json"):
        if p.is_file():
            found.append(p)

    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found


class MultiCamViewer3DApp:
    """Viser-based 3D telemetry viewer for Multi-Camera Planes & Cube tracker."""

    def __init__(self, initial_json_path: Path | None = None, port: int = 8080):
        self.port = port
        self.outputs_dir = REPO_ROOT / "outputs"
        self.active_json_path = initial_json_path
        self.lock = threading.Lock()
        self.current_data: dict[str, Any] = {}
        self.last_mtime: float = 0.0
        self.running = True

        # View shortcut camera cache: camera_id -> (eye, look_at, up)
        self.cam_views: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._cam_pov_buttons: dict[str, Any] = {}

        # 1. Start Viser Server
        self.server = viser.ViserServer(port=self.port)
        self.server.configure_theme(
            titlebar_content=None,
            control_layout="floating",
            dark_mode=True,
            show_logo=False,
            brand_color=(0, 200, 255),
        )

        # 2. File Selection & Watcher
        available_files = scan_available_results_files(self.outputs_dir)
        dropdown_options = [str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p) for p in available_files]
        if not dropdown_options:
            dropdown_options = ["outputs/live_tracking_results.json"]

        curr_rel = str(self.active_json_path.relative_to(REPO_ROOT)) if self.active_json_path and self.active_json_path.is_relative_to(REPO_ROOT) else (str(self.active_json_path) if self.active_json_path else dropdown_options[0])
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

        self.dd_text_mode = self.server.gui.add_dropdown(
            "Text Mode",
            options=["max", "lite", "none"],
            initial_value="max",
        )

        @self.dd_text_mode.on_update
        def _(_):
            self.render_scene_3d(self.current_data)

        # Standard preset view shortcuts
        self.server.gui.add_divider()
        self.server.gui.add_markdown("**🎥 View Shortcuts**")
        for label, fn in (
            ("🏠 Home", self.view_home),
            ("🔼 Top", self.view_top),
            ("➡️ Front", self.view_front),
            ("↔️ Side", self.view_side),
            ("💠 Cube", self.view_cube),
        ):
            btn_view = self.server.gui.add_button(label)
            @btn_view.on_click
            def _(event, _fn=fn):
                _fn(event.client)

        self.gui_pov_folder = self.server.gui.add_folder("📷 Camera POVs")

        # Snapshot Action Button
        self.server.gui.add_divider()
        self.btn_snapshot = self.server.gui.add_button("📸 Save Snapshot (JSON + PNG)")
        @self.btn_snapshot.on_click
        def _(event):
            self.save_snapshot_json_and_image(event.client)

        # Telemetry Readout Cards
        self.server.gui.add_divider()
        self.telemetry_markdown = self.server.gui.add_markdown(
            "**Waiting for tracking telemetry...**"
        )

        # Initial Load
        if self.active_json_path and self.active_json_path.is_file():
            self.reload_data(force=True)
        elif dropdown_options:
            cand = REPO_ROOT / dropdown_options[0]
            if cand.is_file():
                self.active_json_path = cand
                self.reload_data(force=True)

        self.watch_thread = threading.Thread(target=self._file_watcher_loop, daemon=True)
        self.watch_thread.start()

    def _file_watcher_loop(self):
        while self.running:
            try:
                if self.cb_live_watch.value and self.active_json_path and self.active_json_path.is_file():
                    mtime = self.active_json_path.stat().st_mtime
                    if mtime > self.last_mtime:
                        self.reload_data()
            except Exception:
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

            self.render_scene_3d(data)
            self.update_telemetry_card(data)

        except Exception as exc:
            log.warning(f"Error parsing tracking results: {exc}")

    def _add_label(self, name: str, text: str, position: Any, minor: bool = False) -> None:
        mode = self.dd_text_mode.value
        if mode == "none":
            return
        if mode == "lite" and minor:
            return
        self.server.scene.add_label(name, text=text, position=position)

    def render_scene_3d(self, data: dict[str, Any]):
        """Render all 3D features in the Base Tag 0 / base_link reference frame."""
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

        origin_b = np.array([0.0, 0.0, 0.0])
        self.server.scene.add_frame(
            "/base/axes",
            axes_length=0.12,
            axes_radius=0.005,
            origin_radius=0.008,
            origin_color=(0, 255, 100),
            position=origin_b,
        )
        self._add_label("/base/origin_label", text="base_link [0, 0, 0]", position=origin_b + np.array([0, 0, 0.02]))

        tag0_info = data.get("tag0")
        if tag0_info and "xyz_mm" in tag0_info:
            p0 = np.asarray(tag0_info["xyz_mm"], dtype=np.float64) * 0.001
            self.server.scene.add_box("/base/tag_0", dimensions=(0.134, 0.134, 0.002), position=p0, color=(0, 255, 100))
            self._add_label("/base/label_tag_0", text=f"Tag 0\n[{p0[0]*1e3:+.0f}, {p0[1]*1e3:+.0f}, {p0[2]*1e3:+.0f}]mm", position=p0 + np.array([0, 0, 0.02]), minor=True)

        # 2. Plane A: Line / Pair (Tags 40, 41)
        pa = data.get("plane_a")
        if pa and "tag_40_mm" in pa and "tag_41_mm" in pa:
            p40 = np.asarray(pa["tag_40_mm"], dtype=np.float64) * 0.001
            p41 = np.asarray(pa["tag_41_mm"], dtype=np.float64) * 0.001
            mid = (p40 + p41) / 2.0
            dist_mm = float(pa.get("baseline_dist_mm", np.linalg.norm(p41 - p40) * 1000.0))

            self.server.scene.add_spline_catmull_rom(
                "/plane_a/baseline",
                positions=np.array([p40, mid, p41]),
                color=(0, 220, 255),
                line_width=3.0,
            )
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

            self.server.scene.add_arrows(
                "/plane_b/normal_vector",
                points=np.array([[c_b, c_b + normal * 0.12]]),
                colors=(255, 140, 0),
                shaft_radius=0.004,
                head_radius=0.010,
                head_length=0.025,
            )

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

            if len(corner_pts) >= 3:
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

        # 4. 6-Faced Cube (Tags 20..25)
        cube = data.get("cube_centroid")
        if cube and "xyz_mm" in cube:
            c_pos = np.asarray(cube["xyz_mm"], dtype=np.float64) * 0.001
            spread_mm = float(cube.get("residual_spread_mm", 0.0))
            faces_str = ",".join(map(str, cube.get("visible_faces", [])))

            self.server.scene.add_box(
                "/cube/body",
                dimensions=(0.041, 0.041, 0.041),
                position=c_pos,
                color=(220, 20, 220),
                opacity=0.85,
            )
            self._add_label(
                "/cube/label",
                text=f"CUBE Centroid: [{c_pos[0]*1e3:+.0f}, {c_pos[1]*1e3:+.0f}, {c_pos[2]*1e3:+.0f}] mm\nFaces: [{faces_str}] Spread: ±{spread_mm:.1f}mm",
                position=c_pos + np.array([0, 0, 0.035]),
            )

        cube_faces = data.get("cube_faces_mm") or {}
        for tid_str, corners_mm in cube_faces.items():
            pts = np.asarray(corners_mm, dtype=np.float32) * 0.001
            if len(pts) == 4:
                f_idx = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
                self.server.scene.add_mesh_simple(
                    f"/cube/face_{tid_str}",
                    vertices=pts,
                    faces=f_idx,
                    color=(200, 30, 200),
                    opacity=0.7,
                    side="double",
                )

        # 5. Custom 'H' Frame
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

        # 6. Camera Frustums for all N cameras
        self.cam_views.clear()
        cam_ext = data.get("cam_extrinsics") or {}

        palette = [
            (0, 180, 255),
            (255, 100, 180),
            (80, 230, 100),
            (255, 170, 40),
            (180, 100, 255),
            (60, 230, 240),
        ]

        for cid_str, ext in cam_ext.items():
            if "R_c_base" in ext and "t_c_base_m" in ext:
                R_c_b = np.asarray(ext["R_c_base"], dtype=np.float64)
                t_c_b = np.asarray(ext["t_c_base_m"], dtype=np.float64)
                cam_pos = -R_c_b.T @ t_c_b
                q_cam = matrix_to_wxyz(R_c_b.T)

                col = palette[int(cid_str) % len(palette)]
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

                R_base_cam = R_c_b.T
                look_dir = R_base_cam @ np.array([0.0, 0.0, 1.0])
                up_dir = R_base_cam @ np.array([0.0, -1.0, 0.0])
                self.cam_views[str(cid_str)] = (
                    cam_pos.astype(np.float64),
                    (cam_pos + look_dir * 0.4).astype(np.float64),
                    up_dir.astype(np.float64),
                )

        # Update POV buttons
        self._update_pov_buttons()

    def _update_pov_buttons(self):
        for cid_str in sorted(self.cam_views.keys(), key=lambda s: int(s) if s.isdigit() else 99):
            if cid_str not in self._cam_pov_buttons:
                btn = self.gui_pov_folder.add_button(f"📷 Cam {cid_str} POV")
                def make_handler(cid):
                    def _h(event):
                        self.view_camera(event.client, cid)
                    return _h
                btn.on_click(make_handler(cid_str))
                self._cam_pov_buttons[cid_str] = btn

    def view_home(self, client: viser.ClientHandle | None = None):
        target_clients = [client] if client is not None else list(self.server.get_clients().values())
        for c in target_clients:
            c.camera.position = np.array([0.7, -0.9, 0.8])
            c.camera.look_at = np.array([0.0, 0.2, 0.1])
            c.camera.up_direction = np.array([0.0, 0.0, 1.0])

    def view_top(self, client: viser.ClientHandle | None = None):
        target_clients = [client] if client is not None else list(self.server.get_clients().values())
        for c in target_clients:
            c.camera.position = np.array([0.0, 0.2, 1.4])
            c.camera.look_at = np.array([0.0, 0.2, 0.0])
            c.camera.up_direction = np.array([0.0, 1.0, 0.0])

    def view_front(self, client: viser.ClientHandle | None = None):
        target_clients = [client] if client is not None else list(self.server.get_clients().values())
        for c in target_clients:
            c.camera.position = np.array([0.0, -1.2, 0.3])
            c.camera.look_at = np.array([0.0, 0.2, 0.1])
            c.camera.up_direction = np.array([0.0, 0.0, 1.0])

    def view_side(self, client: viser.ClientHandle | None = None):
        target_clients = [client] if client is not None else list(self.server.get_clients().values())
        for c in target_clients:
            c.camera.position = np.array([1.2, 0.2, 0.3])
            c.camera.look_at = np.array([0.0, 0.2, 0.1])
            c.camera.up_direction = np.array([0.0, 0.0, 1.0])

    def view_cube(self, client: viser.ClientHandle | None = None):
        cube = self.current_data.get("cube_centroid")
        if not cube or "xyz_mm" not in cube:
            return
        c_pos = np.asarray(cube["xyz_mm"], dtype=np.float64) * 0.001
        target_clients = [client] if client is not None else list(self.server.get_clients().values())
        for c in target_clients:
            c.camera.position = c_pos + np.array([0.15, -0.22, 0.18])
            c.camera.look_at = c_pos
            c.camera.up_direction = np.array([0.0, 0.0, 1.0])

    def view_camera(self, client: viser.ClientHandle | None, cid_str: str):
        if cid_str not in self.cam_views:
            return
        cam_pos, look_at, up_dir = self.cam_views[cid_str]
        target_clients = [client] if client is not None else list(self.server.get_clients().values())
        for c in target_clients:
            c.camera.position = cam_pos
            c.camera.look_at = look_at
            c.camera.up_direction = up_dir

    def update_telemetry_card(self, data: dict[str, Any]):
        cam_ext = data.get("cam_extrinsics") or {}
        num_cams = max(len(cam_ext), 1)

        lines = [
            f"### 📊 Tracking Telemetry Snapshot",
            f"**File:** `{self.active_json_path.name if self.active_json_path else 'None'}`\n",
            f"| Camera | Status | Method |",
            f"| :--- | :--- | :--- |",
        ]

        for cid in sorted([int(k) for k in cam_ext.keys()]):
            ext = cam_ext.get(cid) or cam_ext.get(str(cid))
            meth = ext.get("method", "locked")
            status = "🟢 LOCKED"
            lines.append(f"| **Camera {cid}** | {status} | `{meth}` |")

        if not cam_ext:
            lines.append(f"| *No cameras locked* | 🔴 SEARCHING | none |")
        lines.append("\n")

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
        stamp = time.strftime("%Y%m%d_%H%M%S")
        target_dir = self.active_json_path.parent if self.active_json_path else (self.outputs_dir / "snapshots")
        target_dir.mkdir(parents=True, exist_ok=True)

        json_out = target_dir / f"snapshot_{stamp}.json"
        img_out = target_dir / f"snapshot_{stamp}.png"

        with self.lock:
            snap_data = dict(self.current_data)
        snap_data["snapshot_export_time"] = datetime.now().isoformat()

        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(snap_data, f, indent=2)

        target_client = client or next(iter(self.server.get_clients().values()), None)
        img_saved = False
        if target_client is not None:
            try:
                render_rgb = target_client.get_render(height=720, width=1280)
                iio.imwrite(img_out, render_rgb)
                img_saved = True
            except Exception as exc:
                log.warning(f"Could not grab client render: {exc}")

        msg = f"Saved snapshot JSON: {json_out.name}"
        if img_saved:
            msg += f" + 3D Render Image: {img_out.name}"
        if target_client is not None:
            target_client.add_notification(title="📸 Snapshot Saved!", body=msg, loading=False)


def main():
    parser = argparse.ArgumentParser(description="Multi-Camera Planes & Cube 3D Viewer")
    parser.add_argument("--input", type=str, default=None, help="Path to live_tracking_results.json")
    parser.add_argument("--port", type=int, default=8080, help="Viser WebGL port (default: 8080)")
    args = parser.parse_args()

    in_path = Path(args.input) if args.input else None
    app = MultiCamViewer3DApp(initial_json_path=in_path, port=args.port)
    print(f"\n🌐 [3D VIEWER ONLINE] Open http://localhost:{args.port} in your browser.")
    print("Press Ctrl+C to terminate.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nViewer stopped.")


if __name__ == "__main__":
    main()
