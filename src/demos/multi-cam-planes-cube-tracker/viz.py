#!/usr/bin/env python3
"""Visualization layer for the Multi-Camera Planes & Cube Tracker.

- Re-projects the fused Base-Frame geometry (the same numeric arrays shown in
  the HUD and exported JSON) back into each camera view and into a software
  3D scene panel, so every [X, Y, Z] array can be verified visually across all cameras.
- Dynamically tiles N camera feeds into a fixed 640x720 column.
- Renders the telemetry HUD dashboard with multi-camera status cards.
- Logs each frame's coordinates to a time-series CSV and saves periodic
  annotated-frame snapshots (photo collection) via :class:`SessionLogger`.

Independent of ``tracker``: takes plain arrays / results dicts.
"""

from __future__ import annotations

import csv
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

log = logging.getLogger("multi-cam-planes-cube-tracker")

# BGR colors shared by camera overlays and the 3D scene panel
COL_X = (0, 0, 255)          # Base +X axis: Red
COL_Y = (0, 255, 0)          # Base +Y axis: Green
COL_Z = (255, 0, 0)          # Base +Z axis: Blue
COL_PLANE_A = (255, 200, 0)  # Plane A / Tags 40,41: Cyan
COL_PLANE_B = (0, 140, 255)  # Plane B / Tags 42..45: Orange
COL_CUBE = (220, 0, 220)     # Cube / Tags 20..25: Magenta
COL_HFRAME = (0, 255, 255)   # Custom H frame: Yellow
COL_CAM = (80, 200, 255)     # Camera frustums
COL_GRID = (58, 62, 72)      # Ground grid

# Palette for distinct camera frustums and badges in multi-camera mode
CAM_PALETTE = [
    (80, 200, 255),   # Cam 0: Cyan/Sky blue
    (255, 120, 200),  # Cam 1: Pink/Magenta
    (100, 255, 120),  # Cam 2: Light green
    (255, 180, 60),   # Cam 3: Amber/Orange
    (200, 140, 255),  # Cam 4: Lavender/Purple
    (80, 240, 240),   # Cam 5: Aqua
    (240, 200, 80),   # Cam 6: Golden yellow
    (140, 200, 255),  # Cam 7: Ice blue
]


def get_camera_color(cam_id: int) -> tuple[int, int, int]:
    """Return a consistent color for camera ID."""
    return CAM_PALETTE[cam_id % len(CAM_PALETTE)]


def safe_pt(uv: np.ndarray | Sequence[float] | None, max_coord: float = 30000.0) -> tuple[int, int] | None:
    """Safely converts (u, v) array or sequence to integer pixel tuple.

    Returns (int, int) or None if point contains NaN/Inf, is out of bounds, or invalid.
    Guarantees OpenCV drawing functions will never fail with type/overflow errors.
    """
    if uv is None:
        return None
    try:
        arr = np.asarray(uv, dtype=np.float64).ravel()
        if arr.size < 2 or not np.isfinite(arr[:2]).all():
            return None
        x, y = float(arr[0]), float(arr[1])
        if abs(x) > max_coord or abs(y) > max_coord:
            return None
        return (int(round(x)), int(round(y)))
    except Exception:
        return None


def project_base_point(
    p_base_m: np.ndarray,
    R_c_base: np.ndarray,
    t_c_base: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray | None:
    """Project a Base-Frame metric point (metres) into camera pixels.

    Returns (u, v) in pixels, or None when behind camera or projection is invalid.
    """
    try:
        p_base = np.asarray(p_base_m, dtype=np.float64).ravel()
        if p_base.size != 3 or not np.isfinite(p_base).all():
            return None
        p_cam = R_c_base @ p_base + t_c_base
        if not np.isfinite(p_cam).all() or p_cam[2] <= 0.01:
            return None
        px, _ = cv2.projectPoints(
            p_cam.reshape(1, 3), np.zeros(3), np.zeros(3), K, dist_coeffs
        )
        uv = px.reshape(2)
        if not np.isfinite(uv).all() or abs(uv[0]) > 25000 or abs(uv[1]) > 25000:
            return None
        return uv
    except Exception:
        return None


def fmt_xyz_mm(p_mm: Sequence[float]) -> str:
    """Compact '[+12.3,-45.6,+78.9]' readout used by 3D overlay labels."""
    return f"[{p_mm[0]:+6.1f},{p_mm[1]:+6.1f},{p_mm[2]:+6.1f}]"


def draw_shadow_text(img: np.ndarray, org: tuple[int, int], text: str, color,
                     scale: float = 0.38, thickness: int = 1) -> None:
    """Outlined text so labels stay readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def convex_hull_polygon(pts_2d: np.ndarray) -> np.ndarray | None:
    """Compute 2D convex hull of 2D points for platform polygon fill."""
    if pts_2d.shape[0] < 3:
        return None
    try:
        pts = pts_2d.astype(np.float32).reshape(-1, 1, 2)
        hull = cv2.convexHull(pts)
        return hull.reshape(-1, 2)
    except Exception:
        return None


class SessionLogger:
    """Time-series history log + periodic annotated-frame photo collection.

    Writes each frame's metric coordinates to ``session_dir/history.csv`` and
    saves the combined annotated UI frame every ``photo_interval_s`` seconds
    into ``session_dir/frames/``.
    """

    CSV_COLUMNS = [
        "frame", "wall_time_iso", "elapsed_s",
        "cams_locked",
        "cube_x_mm", "cube_y_mm", "cube_z_mm", "cube_spread_mm", "cube_faces",
        "plane_a_x_mm", "plane_a_y_mm", "plane_a_z_mm", "plane_a_baseline_mm",
        "plane_b_x_mm", "plane_b_y_mm", "plane_b_z_mm", "plane_b_rmse_mm", "plane_b_tags",
        "h_origin_x_mm", "h_origin_y_mm", "h_origin_z_mm",
        "h_cube_x_mm", "h_cube_y_mm", "h_cube_z_mm",
    ]

    def __init__(self, session_root: Path, history_interval_s: float = 0.1,
                 photo_interval_s: float = 0.0, use_timestamp_subfolder: bool = True):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        root = Path(session_root)
        if use_timestamp_subfolder:
            self.session_dir = root / f"session_{stamp}"
        else:
            self.session_dir = root
        self.frames_dir = self.session_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.session_dir / "history.csv"
        self._fh = open(self.history_path, "w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(self.CSV_COLUMNS)
        self.history_interval_s = max(0.0, float(history_interval_s))
        self.photo_interval_s = max(0.0, float(photo_interval_s))
        self._frame_idx = 0
        self._last_history_t = -1e9
        self._last_photo_t = -1e9
        self._start_wall = time.time()
        self.photos_saved = 0
        self._closed = False
        log.info(f"Session time-series log: {self.history_path}")

    @staticmethod
    def _f(v) -> str:
        return "" if v is None else f"{float(v):.3f}"

    @classmethod
    def _row(cls, idx: int, now: float, elapsed: float, results: dict) -> list:
        cube = results.get("cube_centroid") or {}
        pa = results.get("plane_a") or {}
        pb = results.get("plane_b") or {}
        h = results.get("custom_h_frame") or {}
        cube_xyz = cube.get("xyz_mm") or [None, None, None]
        pa_c = pa.get("center_mm") or [None, None, None]
        pb_c = pb.get("center_mm") or [None, None, None]
        h_o = h.get("origin_base_mm") or [None, None, None]
        h_c = h.get("cube_centroid_in_h_mm") or [None, None, None]

        # Gather locked camera IDs
        exts = results.get("cam_extrinsics") or {}
        cams_locked = ";".join(str(cid) for cid in sorted(exts.keys()))

        return [
            idx,
            datetime.fromtimestamp(now).isoformat(timespec="milliseconds"),
            f"{elapsed:.3f}",
            cams_locked,
            cls._f(cube_xyz[0]), cls._f(cube_xyz[1]), cls._f(cube_xyz[2]),
            cls._f(cube.get("residual_spread_mm")),
            ";".join(map(str, cube.get("visible_faces") or [])),
            cls._f(pa_c[0]), cls._f(pa_c[1]), cls._f(pa_c[2]),
            cls._f(pa.get("baseline_dist_mm")),
            cls._f(pb_c[0]), cls._f(pb_c[1]), cls._f(pb_c[2]),
            cls._f(pb.get("coplanarity_rmse_mm")),
            ";".join(map(str, pb.get("detected_tags") or [])),
            cls._f(h_o[0]), cls._f(h_o[1]), cls._f(h_o[2]),
            cls._f(h_c[0]), cls._f(h_c[1]), cls._f(h_c[2]),
        ]

    def record(self, results: dict, ui_frame: np.ndarray | None = None) -> None:
        """Append one time-series row and (on interval) save one photo."""
        if self._closed:
            return
        self._frame_idx += 1
        now = time.time()
        elapsed = now - self._start_wall

        if elapsed - self._last_history_t >= self.history_interval_s:
            self._last_history_t = elapsed
            self._writer.writerow(self._row(self._frame_idx, now, elapsed, results))
            self._fh.flush()

        if (ui_frame is not None and self.photo_interval_s > 0.0
                and elapsed - self._last_photo_t >= self.photo_interval_s):
            self._last_photo_t = elapsed
            self.save_photo(ui_frame, f"frame_{self._frame_idx:06d}_t{elapsed:08.1f}s.png")

    def save_photo(self, ui_frame: np.ndarray, name: str) -> None:
        path = self.frames_dir / name
        try:
            cv2.imwrite(str(path), ui_frame)
            self.photos_saved += 1
            log.info(f"Saved session photo: {path}")
        except Exception as exc:
            log.warning(f"Could not save session photo {path}: {exc}")

    def close(self, final_frame: np.ndarray | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        if final_frame is not None and self.photo_interval_s > 0.0:
            self.save_photo(final_frame, "frame_final.png")
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass
        log.info(f"Session log closed: {self.session_dir} "
                 f"({self._frame_idx} frames, {self.photos_saved} photos)")


def layout_camera_views(vis_views: list[np.ndarray], width: int = 640, height: int = 720) -> np.ndarray:
    """Neatly tile N camera views into a fixed (width x height) column canvas.

    - 1 camera: 640x360 centered vertically with letterboxing.
    - 2 cameras: two 640x360 views vertically stacked (standard 2-cam view).
    - 3 or 4 cameras: 2x2 grid (320x360 each).
    - >4 cameras: dynamic 2-column or 3-column grid scaled to fit.
    """
    n = len(vis_views)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (18, 20, 24)

    if n == 0:
        cv2.putText(canvas, "No Cameras Configured", (width // 2 - 120, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 140), 1, cv2.LINE_AA)
        return canvas

    if n == 1:
        # Centered 640x360
        th = cv2.resize(vis_views[0], (width, height // 2))
        y_off = (height - height // 2) // 2
        canvas[y_off:y_off + height // 2, 0:width] = th
        return canvas

    if n == 2:
        # Stacked 640x360 + 640x360
        th0 = cv2.resize(vis_views[0], (width, height // 2))
        th1 = cv2.resize(vis_views[1], (width, height // 2))
        return np.vstack([th0, th1])

    if n in (3, 4):
        # 2x2 grid: each cell is (width // 2, height // 2) = (320, 360)
        cell_w = width // 2
        cell_h = height // 2
        for i in range(4):
            r = i // 2
            c = i % 2
            y0 = r * cell_h
            x0 = c * cell_w
            if i < n:
                th = cv2.resize(vis_views[i], (cell_w, cell_h))
                canvas[y0:y0 + cell_h, x0:x0 + cell_w] = th
            else:
                # Placeholder for unused 4th slot in 3-cam mode
                cv2.rectangle(canvas, (x0 + 4, y0 + 4), (x0 + cell_w - 4, y0 + cell_h - 4), (28, 32, 40), -1)
                cv2.putText(canvas, "Cam Slot Empty", (x0 + 30, y0 + cell_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 85, 100), 1, cv2.LINE_AA)
        return canvas

    # For n > 4: dynamic grid with cols = 2 or 3
    cols = 2 if n <= 6 else 3
    rows = math.ceil(n / cols)
    cell_w = width // cols
    cell_h = height // rows

    for i, v in enumerate(vis_views):
        r = i // cols
        c = i % cols
        y0 = r * cell_h
        x0 = c * cell_w
        th = cv2.resize(v, (cell_w, cell_h))
        canvas[y0:y0 + cell_h, x0:x0 + cell_w] = th

    return canvas


def annotate_camera_view(frame: np.ndarray, dets: dict, cam_label: str,
                         results: dict | None = None, cam_id: int = 0,
                         K: np.ndarray | None = None,
                         dist: np.ndarray | None = None) -> np.ndarray:
    """Annotate tag polygons, IDs and the re-projected 3D overlay."""
    vis = frame.copy()

    # Draw detected AprilTags in this frame
    for tid, d in dets.items():
        pts = d.corners_px.astype(np.int32)
        if tid == 0:
            color = (0, 255, 0)       # Base Tag 0: Green
            label = f"BASE [Tag {tid}]"
        elif tid in (40, 41):
            color = (255, 200, 0)     # Plane A (40, 41): Cyan
            label = f"Plane A [Tag {tid}]"
        elif tid in range(42, 46):
            color = (0, 140, 255)     # Plane B (42..45): Orange
            label = f"Plane B [Tag {tid}]"
        elif tid in range(20, 26):
            color = (220, 0, 220)     # Cube (20..25): Magenta
            label = f"Cube Face [Tag {tid}]"
        else:
            color = (200, 200, 200)
            label = f"Tag {tid}"

        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
        c = d.center_px.astype(np.int32)
        cv2.circle(vis, tuple(c), 4, color, -1)
        cv2.putText(vis, label, (pts[0][0], max(16, pts[0][1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Header badge with camera specific color bar
    badge_col = get_camera_color(cam_id)
    cv2.rectangle(vis, (10, 10), (240, 38), (30, 30, 30), -1)
    cv2.line(vis, (10, 10), (10, 38), badge_col, 4)
    cv2.putText(vis, f"🎥 {cam_label}", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    if results and K is not None and dist is not None:
        draw_base_frame_overlay(vis, results, cam_id, K, dist)
    return vis


def draw_base_frame_overlay(vis: np.ndarray, results: dict, cam_id: int,
                            K: np.ndarray, dist: np.ndarray) -> None:
    """Re-project the fused metric results into one camera view (in-place)."""
    ext = (results.get("cam_extrinsics") or {}).get(cam_id)
    if not ext:
        return

    R_c_base = np.array(ext["R_c_base"], dtype=np.float64)
    t_c_base = np.array(ext["t_c_base_m"], dtype=np.float64)
    img_h, img_w = vis.shape[:2]

    def proj(p_m: np.ndarray) -> np.ndarray | None:
        return project_base_point(np.asarray(p_m, dtype=np.float64),
                                  R_c_base, t_c_base, K, dist)

    def mm(v) -> np.ndarray:
        return np.asarray(v, dtype=np.float64) * 0.001

    def label_px(uv, text, color, dx=10, dy=-8, scale=0.36):
        pt = safe_pt(uv)
        if pt is None:
            return
        x, y = pt
        if not (-240 <= x <= img_w + 240 and -240 <= y <= img_h + 240):
            return
        draw_shadow_text(vis, (x + dx, max(16, y + dy)), text, color, scale=scale)

    # 1. Base axes at the base_link origin [0, 0, 0] (X red, Y green, Z blue)
    origin_m = np.zeros(3)
    o_pt = safe_pt(proj(origin_m))
    if o_pt is not None:
        for d_m, col, name in [((0.10, 0.0, 0.0), COL_X, "X"),
                               ((0.0, 0.10, 0.0), COL_Y, "Y"),
                               ((0.0, 0.0, 0.10), COL_Z, "Z")]:
            e_pt = safe_pt(proj(np.array(d_m, dtype=np.float64)))
            if e_pt is None:
                continue
            cv2.arrowedLine(vis, o_pt, e_pt, col, 2, cv2.LINE_AA, tipLength=0.2)
            label_px(e_pt, name, col, dx=4, dy=-4, scale=0.55)
        label_px(o_pt, "base_link [0,0,0]", (0, 255, 120), dy=14, scale=0.34)

    # 1b. Tag 0 reference marker in base_link
    t0_info = results.get("tag0")
    if t0_info:
        p0_m = mm(t0_info["xyz_mm"])
        pt0 = safe_pt(proj(p0_m))
        if pt0 is not None:
            cv2.circle(vis, pt0, 5, (0, 255, 100), -1)
            label_px(pt0, f"Tag 0 {fmt_xyz_mm(t0_info['xyz_mm'])}", (0, 255, 100), dy=-10, scale=0.34)

    # 2. Plane A (Tags 40, 41): segment + endpoints with coordinate readout
    pa = results.get("plane_a")
    if pa:
        p40, p41 = mm(pa["tag_40_mm"]), mm(pa["tag_41_mm"])
        pt40, pt41 = safe_pt(proj(p40)), safe_pt(proj(p41))
        if pt40 is not None and pt41 is not None:
            cv2.line(vis, pt40, pt41, COL_PLANE_A, 2, cv2.LINE_AA)
        label_px(pt40, f"A40 {fmt_xyz_mm(pa['tag_40_mm'])}", COL_PLANE_A, dx=12)
        label_px(pt41, f"A41 {fmt_xyz_mm(pa['tag_41_mm'])}", COL_PLANE_A, dx=12)
        for pt in (pt40, pt41):
            if pt is not None:
                cv2.circle(vis, pt, 6, COL_PLANE_A, 2)

    # 3. Plane B (Tags 42..45): convex hull outline, centers and normal arrow
    pb = results.get("plane_b")
    if pb:
        proj_pts = []
        for tid, c_mm in (pb.get("centers_mm") or {}).items():
            u = proj(mm(c_mm))
            if u is not None:
                proj_pts.append(u)
                pt = safe_pt(u)
                if pt is not None:
                    cv2.circle(vis, pt, 5, COL_PLANE_B, 2)
                    label_px(u, f"B{tid} {fmt_xyz_mm(c_mm)}", COL_PLANE_B, dx=8, dy=12)

        if len(proj_pts) >= 3:
            hull = convex_hull_polygon(np.array(proj_pts))
            if hull is not None:
                overlay = vis.copy()
                cv2.fillPoly(overlay, [hull.astype(np.int32)], (0, 100, 200))
                cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
                cv2.polylines(vis, [hull.astype(np.int32)], isClosed=True,
                              color=COL_PLANE_B, thickness=2, lineType=cv2.LINE_AA)

        c_base = mm(pb["center_mm"])
        n_base = np.array(pb["normal"], dtype=np.float64)
        c_pt, n_pt = safe_pt(proj(c_base)), safe_pt(proj(c_base + n_base * 0.12))
        if c_pt is not None and n_pt is not None:
            cv2.arrowedLine(vis, c_pt, n_pt, (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.25)
            label_px(n_pt, f"PL-B n=[{n_base[0]:+.2f},{n_base[1]:+.2f},{n_base[2]:+.2f}]",
                     (0, 255, 255), dx=6, dy=-6)

    # 4. Cube Centroid (Tags 20..25)
    cube = results.get("cube_centroid")
    if cube:
        c_m = mm(cube["xyz_mm"])
        c_pt = safe_pt(proj(c_m))
        if c_pt is not None:
            cv2.circle(vis, c_pt, 8, COL_CUBE, -1)
            cv2.circle(vis, c_pt, 12, (255, 255, 255), 1)
            txt = f"CUBE {fmt_xyz_mm(cube['xyz_mm'])} ±{cube.get('residual_spread_mm', 0):.1f}mm"
            label_px(c_pt, txt, COL_CUBE, dx=14, dy=4, scale=0.40)

    # 4b. Re-projected Cube Faces
    for tid, corners_mm in (results.get("cube_faces_mm") or {}).items():
        cuv = [proj(mm(c)) for c in corners_mm]
        if any(u is None for u in cuv):
            continue
        poly = np.array([u for u in cuv if u is not None], dtype=np.int32)
        cv2.polylines(vis, [poly], isClosed=True, color=COL_CUBE, thickness=1, lineType=cv2.LINE_AA)

    # 5. Custom 'H' Frame
    h_info = results.get("custom_h_frame")
    if h_info:
        o_m = np.array(h_info["origin_base_m"], dtype=np.float64)
        R_h = np.array(h_info["R_custom_base"], dtype=np.float64)
        o_pt = safe_pt(proj(o_m))
        if o_pt is not None:
            for i, (col, nm) in enumerate([((0, 255, 255), "Hx"),
                                           ((255, 255, 0), "Hy"),
                                           ((240, 240, 240), "Hz")]):
                e_pt = safe_pt(proj(o_m + R_h[:, i] * 0.08))
                if e_pt is not None:
                    cv2.arrowedLine(vis, o_pt, e_pt, col, 1, cv2.LINE_AA, tipLength=0.2)
                    label_px(e_pt, nm, col, dx=4, dy=-4, scale=0.4)
            cv2.circle(vis, o_pt, 5, COL_HFRAME, -1)
            label_px(o_pt, f"H {fmt_xyz_mm(h_info['origin_base_mm'])}", COL_HFRAME, dy=14, scale=0.34)


def render_scene3d_panel(results: dict,
                         calibs: dict[int, Any] | Sequence[Any],
                         width: int = 640, height: int = 720) -> np.ndarray:
    """Software-rendered 3D perspective view of the Base Frame scene with all cameras."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (18, 20, 24)
    cv2.rectangle(panel, (0, 0), (width, 50), (28, 32, 40), -1)
    cv2.putText(panel, "3D Scene View (Base Frame)", (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

    # Normalize calibs lookup
    calib_map = {}
    if isinstance(calibs, dict):
        calib_map = calibs
    elif isinstance(calibs, (list, tuple)):
        for idx, c in enumerate(calibs):
            calib_map[idx] = c

    # Collect scene points (metres) for auto-framing
    pts: list[np.ndarray] = [np.zeros(3)]
    pa = results.get("plane_a")
    if pa:
        pts += [np.asarray(pa["tag_40_mm"]) * 0.001, np.asarray(pa["tag_41_mm"]) * 0.001]
    pb = results.get("plane_b")
    if pb:
        pts += [np.asarray(c) * 0.001 for c in (pb.get("centers_mm") or {}).values()]
    cube = results.get("cube_centroid")
    if cube:
        pts.append(np.asarray(cube["xyz_mm"]) * 0.001)
    for corners in (results.get("cube_faces_mm") or {}).values():
        pts += [np.asarray(c) * 0.001 for c in corners]
    exts = results.get("cam_extrinsics") or {}
    pts += [np.asarray(e["position_base_mm"]) * 0.001 for e in exts.values()]

    P = np.array(pts, dtype=np.float64)
    lo, hi = P.min(axis=0), P.max(axis=0)
    center = (lo + hi) / 2.0
    radius = max(0.12, float(np.max(np.linalg.norm(P - center, axis=1))))

    # Virtual pinhole camera (look-at, up = +Z)
    fov_deg = 52.0
    eye = center + np.array([0.75, -1.15, 0.85]) * (radius / math.tan(math.radians(fov_deg / 2.0)) * 1.3)
    fwd = center - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    upv = np.cross(right, fwd)
    f_px = (width / 2.0) / math.tan(math.radians(fov_deg / 2.0))

    def proj(p_m) -> np.ndarray | None:
        d = np.asarray(p_m, dtype=np.float64) - eye
        zc = float(fwd @ d)
        if zc <= 1e-4:
            return None
        return np.array([f_px * float(right @ d) / zc + width / 2.0,
                         -f_px * float(upv @ d) / zc + height / 2.0])

    def seg(p_a, p_b, color, thickness=1):
        pt_a, pt_b = safe_pt(proj(p_a)), safe_pt(proj(p_b))
        if pt_a is None or pt_b is None:
            return
        cv2.line(panel, pt_a, pt_b, color, thickness, cv2.LINE_AA)

    def dot(p_m, color, radius_px=4):
        pt = safe_pt(proj(p_m))
        if pt is not None:
            cv2.circle(panel, pt, radius_px, color, -1)

    def text_at(p_m, text, color, dx=8, dy=-8, scale=0.36):
        pt = safe_pt(proj(p_m))
        if pt is not None:
            draw_shadow_text(panel, (pt[0] + dx, max(16, pt[1] + dy)), text, color, scale=scale)

    # Ground grid in Z=0 plane
    g_step = 0.05
    g_extent = max(0.20, math.ceil(radius * 1.2 / g_step) * g_step)
    xs = np.arange(-g_extent, g_extent + 1e-4, g_step)
    for x in xs:
        seg(np.array([x, -g_extent, 0.0]), np.array([x, g_extent, 0.0]), COL_GRID)
    for y in xs:
        seg(np.array([-g_extent, y, 0.0]), np.array([g_extent, y, 0.0]), COL_GRID)

    # Base axes at origin
    axis_len = 0.10
    seg(np.zeros(3), np.array([axis_len, 0.0, 0.0]), COL_X, 2)
    seg(np.zeros(3), np.array([0.0, axis_len, 0.0]), COL_Y, 2)
    seg(np.zeros(3), np.array([0.0, 0.0, axis_len]), COL_Z, 2)
    text_at(np.array([axis_len, 0.0, 0.0]), "X", COL_X, scale=0.45)
    text_at(np.array([0.0, axis_len, 0.0]), "Y", COL_Y, scale=0.45)
    text_at(np.array([0.0, 0.0, axis_len]), "Z", COL_Z, scale=0.45)
    text_at(np.zeros(3), "base_link", (0, 255, 120), dy=14, scale=0.34)

    # Tag 0 reference marker
    t0_info = results.get("tag0")
    if t0_info:
        p0_m = np.asarray(t0_info["xyz_mm"]) * 0.001
        dot(p0_m, (0, 255, 100), 5)
        text_at(p0_m, f"Tag 0 {fmt_xyz_mm(t0_info['xyz_mm'])}", (0, 255, 100), dy=-10, scale=0.34)

    # Plane A
    if pa:
        p40 = np.asarray(pa["tag_40_mm"]) * 0.001
        p41 = np.asarray(pa["tag_41_mm"]) * 0.001
        seg(p40, p41, COL_PLANE_A, 2)
        for p_m, tid in ((p40, 40), (p41, 41)):
            dot(p_m, COL_PLANE_A, 5)
            text_at(p_m, f"A{tid} {fmt_xyz_mm(np.asarray(p_m) * 1000.0)}", COL_PLANE_A)

    # Plane B
    if pb:
        proj_pts = []
        for tid, c_mm in (pb.get("centers_mm") or {}).items():
            p_m = np.asarray(c_mm) * 0.001
            u = proj(p_m)
            if u is not None:
                proj_pts.append(u)
            dot(p_m, COL_PLANE_B, 4)
            text_at(p_m, f"B{tid}", COL_PLANE_B, scale=0.32)
        hull = convex_hull_polygon(np.array(proj_pts)) if len(proj_pts) >= 3 else None
        if hull is not None:
            cv2.fillPoly(panel, [hull.astype(np.int32)], (45, 70, 105))
            cv2.polylines(panel, [hull.astype(np.int32)], True, COL_PLANE_B, 2, cv2.LINE_AA)
        c_m = np.asarray(pb["center_mm"]) * 0.001
        n_m = np.asarray(pb["normal"], dtype=np.float64)
        seg(c_m, c_m + n_m * 0.08, (255, 255, 255), 2)
        text_at(c_m, f"PL-B {fmt_xyz_mm(pb['center_mm'])}", (255, 255, 255), dy=16)

    # Cube
    for tid, corners_mm in (results.get("cube_faces_mm") or {}).items():
        cuv = [proj(np.asarray(c) * 0.001) for c in corners_mm]
        if any(u is None for u in cuv):
            continue
        poly = np.array([u for u in cuv if u is not None], dtype=np.int32)
        cv2.polylines(panel, [poly], True, COL_CUBE, 1, cv2.LINE_AA)
    if cube:
        c_m = np.asarray(cube["xyz_mm"]) * 0.001
        dot(c_m, COL_CUBE, 6)
        text_at(c_m, f"CUBE {fmt_xyz_mm(cube['xyz_mm'])}", COL_CUBE, dy=18)

    # Custom H frame
    h_info = results.get("custom_h_frame")
    if h_info:
        o_m = np.asarray(h_info["origin_base_m"], dtype=np.float64)
        R_h = np.asarray(h_info["R_custom_base"], dtype=np.float64)
        for i, (col, nm) in enumerate([((0, 255, 255), "Hx"),
                                       ((255, 255, 0), "Hy"),
                                       ((240, 240, 240), "Hz")]):
            seg(o_m, o_m + R_h[:, i] * 0.06, col, 1)
        dot(o_m, COL_HFRAME, 4)
        text_at(o_m, f"H {fmt_xyz_mm(h_info['origin_base_mm'])}", COL_HFRAME, dy=14, scale=0.32)

    # Camera frustums for all cameras
    for c_id, e in exts.items():
        R = np.asarray(e["R_c_base"], dtype=np.float64)
        t = np.asarray(e["t_c_base_m"], dtype=np.float64)
        c_m = -R.T @ t
        cam_col = get_camera_color(int(c_id))

        calib = calib_map.get(int(c_id))
        w_px, h_px = calib.image_size if calib else (1280, 720)
        K = calib.camera_matrix if calib else np.array([[900, 0, 640], [0, 900, 360], [0, 0, 1]])
        Kinv = np.linalg.inv(K)
        depth = radius * 0.45
        ends = []
        for uv in np.array([[0.0, 0.0, 1.0], [w_px, 0.0, 1.0],
                            [w_px, h_px, 1.0], [0.0, h_px, 1.0]]):
            d_c = Kinv @ uv
            d_c = d_c / d_c[2]
            ends.append(c_m + (R.T @ d_c) * depth)
        for ep in ends:
            seg(c_m, ep, cam_col)
        for i in range(4):
            seg(ends[i], ends[(i + 1) % 4], cam_col)
        text_at(c_m, f"CAM{c_id}", cam_col, dy=-12, scale=0.5)

    if not exts:
        cv2.putText(panel, "NO BASE TAG LOCK - waiting for Tag 0", (16, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 110, 160), 1, cv2.LINE_AA)

    # Legend
    cv2.rectangle(panel, (10, height - 72), (width - 10, height - 12), (25, 28, 34), -1)
    draw_shadow_text(panel, (18, height - 52),
                     "X=red Y=grn Z=blu | PlaneA cyan | PlaneB orange",
                     (200, 205, 215), scale=0.35)
    draw_shadow_text(panel, (18, height - 32),
                     "Cube magenta | H frame yellow | Multi-Cam Frustums",
                     (200, 205, 215), scale=0.35)
    return panel


def render_hud_dashboard(results: dict, custom_h_frame_active: bool = False,
                         num_cameras: int = 2,
                         width: int = 640, height: int = 720) -> np.ndarray:
    """Render a dark-themed telemetry dashboard with multi-camera metric coordinates."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (18, 20, 24)

    # Header
    cv2.rectangle(panel, (0, 0), (width, 50), (28, 32, 40), -1)
    cv2.putText(panel, "Multi-Camera 3D Tracker Telemetry", (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 2, cv2.LINE_AA)

    y = 72

    # Section 1: Multi-Camera Status Table
    cam_exts = results.get("cam_extrinsics") or {}
    cam_box_h = 42 + 20 * math.ceil(max(num_cameras, len(cam_exts), 1) / 2)
    cv2.rectangle(panel, (14, y - 18), (width - 14, y + cam_box_h), (25, 30, 38), -1)

    cv2.putText(panel, "COORDINATE ORIGIN: base_link [0, 0, 0] mm", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 120), 1, cv2.LINE_AA)
    map_sz = results.get("world_map_size", 0)
    if map_sz > 0:
        cv2.putText(panel, f"World Map: {map_sz} tags", (width - 170, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)
    y += 22

    active_cams = max(num_cameras, max([int(k) for k in cam_exts.keys()] + [0]) + 1)
    for cid in range(active_cams):
        col_idx = cid % 2
        row_idx = cid // 2
        line_y = y + row_idx * 20
        col_x = 24 if col_idx == 0 else (width // 2 + 10)

        ext_info = cam_exts.get(cid) or cam_exts.get(str(cid))
        has_t0 = bool(results.get(f"cam{cid}_has_base", False))
        if ext_info:
            meth = ext_info.get("method", "locked")
            status = "LOCKED [Multi-PnP]" if meth == "multi_tag_pnp" else "LOCKED [Tag0]"
            col = (0, 255, 100)
        elif has_t0:
            status = "LOCKED [Tag0]"
            col = (0, 255, 100)
        else:
            status = "SEARCHING..."
            col = (80, 80, 220)

        cam_badge_col = get_camera_color(cid)
        cv2.circle(panel, (col_x + 2, line_y - 4), 3, cam_badge_col, -1)
        cv2.putText(panel, f"Cam {cid}: {status}", (col_x + 10, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)

    y += cam_box_h + 10

    # Section 2: Cube 3D Centroid (Tags 20..25)
    cube = results.get("cube_centroid")
    cv2.rectangle(panel, (14, y - 18), (width - 14, y + 78), (25, 30, 38), -1)
    cv2.putText(panel, "6-FACED APRILTAG CUBE (Tags 20..25, 41 mm)", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 255), 1, cv2.LINE_AA)
    y += 22
    if cube:
        x, yp, z = cube["xyz_mm"]
        faces = ",".join(map(str, cube["visible_faces"]))
        cv2.putText(panel, f"Centroid (X,Y,Z): [{x:+6.1f}, {yp:+6.1f}, {z:+6.1f}] mm", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        y += 20
        cv2.putText(panel, f"Visible Faces: [{faces}] | Obs: {cube['num_obs']} | Spread: ±{cube['residual_spread_mm']:.1f} mm",
                    (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "Status: Searching for Cube Faces (20..25)...", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 150), 1, cv2.LINE_AA)
    y += 65

    # Section 3: Plane A (Tags 40, 41)
    pa = results.get("plane_a")
    cv2.rectangle(panel, (14, y - 18), (width - 14, y + 78), (25, 30, 38), -1)
    cv2.putText(panel, "PLANE A: LINE / PAIR (Tags 40, 41, 66 mm)", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)
    y += 22
    if pa:
        cv2.putText(panel, f"Baseline Distance: {pa['baseline_dist_mm']:.1f} mm (Z-Height: {pa['z_height_mm']:+.1f} mm)",
                    (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        y += 20
        c = pa["center_mm"]
        cv2.putText(panel, f"Pair Midpoint (X,Y,Z): [{c[0]:+6.1f}, {c[1]:+6.1f}, {c[2]:+6.1f}] mm",
                    (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "Status: Searching for Tags 40 & 41...", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 150), 1, cv2.LINE_AA)
    y += 65

    # Section 4: Plane B (Tags 42..45)
    pb = results.get("plane_b")
    cv2.rectangle(panel, (14, y - 18), (width - 14, y + 96), (25, 30, 38), -1)
    cv2.putText(panel, "PLANE B: PLATFORM QUAD (Tags 42..45, 66 mm)", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 150, 255), 1, cv2.LINE_AA)
    y += 22
    if pb:
        norm = pb["normal"]
        tags = ",".join(map(str, pb["detected_tags"]))
        cv2.putText(panel, f"Normal: [{norm[0]:+5.2f}, {norm[1]:+5.2f}, {norm[2]:+5.2f}] | Coplanar RMSE: {pb['coplanarity_rmse_mm']:.2f} mm",
                    (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        y += 20
        c = pb["center_mm"]
        cv2.putText(panel, f"Center (X,Y,Z): [{c[0]:+6.1f}, {c[1]:+6.1f}, {c[2]:+6.1f}] mm (Tags: {tags})",
                    (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "Status: Searching for Tags 42, 43, 44, 45...", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 150), 1, cv2.LINE_AA)
    y += 82

    # Section 5: base_link Reference Frame
    bl = results.get("base_link") or results.get("custom_h_frame")
    status_bl = "ACTIVE" if bl and bl.get("active") else "READY (Press 'H' to Activate)"
    status_col = (0, 255, 120) if bl and bl.get("active") else (140, 180, 220)
    cv2.rectangle(panel, (14, y - 18), (width - 14, y + 84), (25, 30, 38), -1)
    cv2.putText(panel, f"BASE_LINK REFERENCE FRAME: {status_bl}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_col, 1, cv2.LINE_AA)
    y += 20
    if bl:
        ox, oy, oz = bl.get("origin_in_tag0_base_mm") or bl.get("origin_base_mm", [0, 0, 0])
        cv2.putText(panel, f"Origin in Tag 0 frame: [{ox:+6.1f}, {oy:+6.1f}, {oz:+6.1f}] mm (+2cm X, +5cm Z)",
                    (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
        y += 18
        if "tag0" in results:
            t0 = results["tag0"]["xyz_mm"]
            cv2.putText(panel, f"Tag 0 in base_link: [{t0[0]:+6.1f}, {t0[1]:+6.1f}, {t0[2]:+6.1f}] mm",
                        (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 200, 100), 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, "Spec: X=Top(40->41), Y=Right, Z=Cam -> RotY(90) -> Z=UP",
                        (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 200, 160), 1, cv2.LINE_AA)
        y += 18
        cv2.putText(panel, "All coordinates in HUD and JSON are relative to base_link [0, 0, 0]",
                    (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 200, 160), 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "Status: Requires Tags 40 & 41 baseline to be visible",
                    (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 150), 1, cv2.LINE_AA)

    # Key controls footer
    cv2.line(panel, (14, height - 36), (width - 14, height - 36), (60, 65, 75), 1)
    cv2.putText(panel, "Keys: 'H'=Custom Frame | 'V'=3D Scene | 'S'=Export JSON | 'Q'=Quit",
                (18, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 150, 170), 1, cv2.LINE_AA)

    return panel
