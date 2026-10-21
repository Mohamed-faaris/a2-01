#!/usr/bin/env python3
"""Visualization layer for the Two-Camera Planes & Cube Tracker.

- Re-projects the fused Base-Frame geometry (the same numeric arrays shown in
  the HUD and exported JSON) back into the camera views and into a software
  3D scene panel, so every [X, Y, Z] array can be verified visually.
- Renders the telemetry HUD dashboard.
- Logs each frame's coordinates to a time-series CSV and saves periodic
  annotated-frame snapshots (photo collection) via :class:`SessionLogger`.

No pipeline/tracker imports here: every renderer takes plain arrays/results
dicts, so this module stays independent of ``tracker``.
"""

from __future__ import annotations

import csv
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

log = logging.getLogger("2cam-planes-cube-tracker")

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
    """Compact '[+12.3,-45.6,+78.9]' readout used by every 3D overlay label."""
    return f"[{p_mm[0]:+6.1f},{p_mm[1]:+6.1f},{p_mm[2]:+6.1f}]"


def draw_shadow_text(img: np.ndarray, org: tuple[int, int], text: str, color,
                     scale: float = 0.38, thickness: int = 1) -> None:
    """Outlined text so labels stay readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def convex_hull_polygon(points_px: np.ndarray) -> np.ndarray | None:
    """Closed 2D convex hull of projected points (for plane outlines)."""
    if len(points_px) < 3:
        return None
    return cv2.convexHull(np.asarray(points_px, dtype=np.float32))


class SessionLogger:
    """Time-series history log + periodic annotated-frame photo collection.

    Writes each frame's metric coordinates to ``session_dir/history.csv`` and
    saves the combined annotated UI frame (cams + 3D overlay + HUD) every
    ``photo_interval_s`` seconds into ``session_dir/frames/``, so a whole
    tracking session can be plotted and browsed offline.
    """

    CSV_COLUMNS = [
        "frame", "wall_time_iso", "elapsed_s",
        "cam0_has_base", "cam1_has_base",
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
        return [
            idx,
            datetime.fromtimestamp(now).isoformat(timespec="milliseconds"),
            f"{elapsed:.3f}",
            int(bool(results.get("cam0_has_base"))),
            int(bool(results.get("cam1_has_base"))),
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


def annotate_camera_view(frame: np.ndarray, dets: dict, cam_label: str,
                         results: dict | None = None, cam_id: int = 0,
                         K: np.ndarray | None = None,
                         dist: np.ndarray | None = None) -> np.ndarray:
    """Annotate tag polygons, IDs and the re-projected 3D overlay.

    ``dets`` maps tag_id -> object with ``corners_px`` (4x2) and ``center_px``.
    """
    vis = frame.copy()

    for tid, d in dets.items():
        pts = d.corners_px.astype(np.int32)
        # Choose color code by entity type
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

    # Header badge
    cv2.rectangle(vis, (10, 10), (220, 38), (30, 30, 30), -1)
    cv2.putText(vis, f"🎥 {cam_label}", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    if results and K is not None and dist is not None:
        draw_base_frame_overlay(vis, results, cam_id, K, dist)
    return vis


def draw_base_frame_overlay(vis: np.ndarray, results: dict, cam_id: int,
                            K: np.ndarray, dist: np.ndarray) -> None:
    """Re-project the fused metric results into one camera view (in-place).

    Everything drawn here is generated from the *same* numeric arrays that
    the HUD and the exported JSON report carry (Base-Frame millimetres), so
    any misalignment against the raw 2D tag detections exposes a bad array.
    """
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

    def label_m(p_m, text, color, **kw):
        label_px(proj(p_m), text, color, **kw)

    # 1. Base-frame axes at the Tag 0 origin (X red, Y green, Z blue)
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
            pt = safe_pt(u)
            if pt is None:
                continue
            proj_pts.append(pt)
            cv2.circle(vis, pt, 5, COL_PLANE_B, -1)
            label_px(pt, f"B{tid} {fmt_xyz_mm(c_mm)}", COL_PLANE_B)
        hull = convex_hull_polygon(np.array(proj_pts)) if len(proj_pts) >= 3 else None
        if hull is not None:
            cv2.polylines(vis, [hull.astype(np.int32)], True, COL_PLANE_B, 2, cv2.LINE_AA)
        c_m = mm(pb["center_mm"])
        n_m = np.asarray(pb["normal"], dtype=np.float64)
        c_pt, n_pt = safe_pt(proj(c_m)), safe_pt(proj(c_m + n_m * 0.08))
        if c_pt is not None and n_pt is not None:
            cv2.arrowedLine(vis, c_pt, n_pt, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.2)
        label_m(c_m, f"PL-B {fmt_xyz_mm(pb['center_mm'])}", (255, 255, 255), dy=16)

    # 4. Cube: fused face quads + centroid crosshair
    for tid, corners_mm in (results.get("cube_faces_mm") or {}).items():
        cuv = [safe_pt(proj(mm(c))) for c in corners_mm]
        if any(pt is None for pt in cuv):
            continue
        poly = np.array(cuv, dtype=np.int32)
        cv2.polylines(vis, [poly], True, COL_CUBE, 1, cv2.LINE_AA)
        label_px(poly.mean(axis=0), f"F{tid}", COL_CUBE, dx=2, dy=4, scale=0.32)
    cube = results.get("cube_centroid")
    if cube:
        c_m = mm(cube["xyz_mm"])
        c_pt = safe_pt(proj(c_m))
        if c_pt is not None:
            cv2.drawMarker(vis, c_pt, COL_CUBE,
                           cv2.MARKER_TILTED_CROSS, 16, 2)
        label_m(c_m, f"CUBE {fmt_xyz_mm(cube['xyz_mm'])}", COL_CUBE, dy=22)

    # 5. Custom H frame axes (thin) when available
    h_info = results.get("custom_h_frame")
    if h_info:
        o_m = np.asarray(h_info["origin_base_m"], dtype=np.float64)
        R_h = np.asarray(h_info["R_custom_base"], dtype=np.float64)
        o_pt = safe_pt(proj(o_m))
        if o_pt is not None:
            for i, (col, nm) in enumerate([((0, 255, 255), "Hx"),
                                           ((255, 255, 0), "Hy"),
                                           ((240, 240, 240), "Hz")]):
                e_pt = safe_pt(proj(o_m + R_h[:, i] * 0.06))
                if e_pt is None:
                    continue
                cv2.arrowedLine(vis, o_pt, e_pt, col, 1, cv2.LINE_AA, tipLength=0.25)
                label_px(e_pt, nm, col, dx=2, dy=2, scale=0.32)
        label_m(o_m, f"H {fmt_xyz_mm(h_info['origin_base_mm'])}", COL_HFRAME, dy=28)

    # Legend strip
    cv2.rectangle(vis, (10, img_h - 60), (330, img_h - 12), (25, 28, 34), -1)
    draw_shadow_text(vis, (18, img_h - 40), "3D OVERLAY (Base Frame, mm)", (200, 205, 215), scale=0.35)
    draw_shadow_text(vis, (18, img_h - 22), "X=red Y=grn Z=blu A=cyn B=org Cb=mag H=yel",
                     (200, 205, 215), scale=0.33)


def render_scene3d_panel(results: dict, calib0, calib1,
                         width: int = 640, height: int = 720) -> np.ndarray:
    """Software-rendered 3D perspective view of the Base Frame scene."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (18, 20, 24)
    cv2.rectangle(panel, (0, 0), (width, 50), (28, 32, 40), -1)
    cv2.putText(panel, "3D Scene View (Base Frame)", (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

    # ---- Collect scene points (metres) for auto-framing -----------------
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

    # ---- Virtual pinhole camera (look-at, up = +Z) -----------------------
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

    def dot(p_m, color, r_px=5):
        pt = safe_pt(proj(p_m))
        if pt is not None:
            cv2.circle(panel, pt, r_px, color, -1)

    def text_at(p_m, text, color, dx=8, dy=-8, scale=0.36):
        pt = safe_pt(proj(p_m))
        if pt is None:
            return
        draw_shadow_text(panel, (pt[0] + dx, max(16, pt[1] + dy)), text, color, scale=scale)

    # ---- Ground grid on z = 0 -------------------------------------------
    span = float(max(hi[0] - lo[0], hi[1] - lo[1], 0.2)) + 0.14
    step = 0.05 if span < 0.45 else 0.1
    n_lines = int(span / step)
    x0 = center[0] - span / 2.0
    y0 = center[1] - span / 2.0
    for i in range(n_lines + 1):
        seg(np.array([x0 + i * step, y0, 0.0]), np.array([x0 + i * step, y0 + span, 0.0]), COL_GRID)
        seg(np.array([x0, y0 + i * step, 0.0]), np.array([x0 + span, y0 + i * step, 0.0]), COL_GRID)

    # ---- Base axes at the Tag 0 origin ----------------------------------
    for d_m, col, name in [((0.10, 0.0, 0.0), COL_X, "X"),
                           ((0.0, 0.10, 0.0), COL_Y, "Y"),
                           ((0.0, 0.0, 0.10), COL_Z, "Z")]:
        e_m = np.array(d_m, dtype=np.float64)
        ue = proj(e_m)
        seg(np.zeros(3), e_m, col, 2)
        if ue is not None:
            draw_shadow_text(panel, (int(ue[0]) + 4, int(ue[1]) - 4), name, col, scale=0.55)

    # ---- Plane A ---------------------------------------------------------
    if pa:
        p40 = np.asarray(pa["tag_40_mm"]) * 0.001
        p41 = np.asarray(pa["tag_41_mm"]) * 0.001
        seg(p40, p41, COL_PLANE_A, 2)
        for p_m, tid in ((p40, 40), (p41, 41)):
            dot(p_m, COL_PLANE_A, 5)
            text_at(p_m, f"A{tid} {fmt_xyz_mm(np.asarray(p_m) * 1000.0)}", COL_PLANE_A)

    # ---- Plane B ---------------------------------------------------------
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

    # ---- Cube -----------------------------------------------------------
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

    # ---- Custom H frame --------------------------------------------------
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

    # ---- Camera frustums -------------------------------------------------
    for c_id, e in exts.items():
        R = np.asarray(e["R_c_base"], dtype=np.float64)
        t = np.asarray(e["t_c_base_m"], dtype=np.float64)
        c_m = -R.T @ t
        calib = calib0 if c_id == 0 else calib1
        w_px, h_px = calib.image_size
        Kinv = np.linalg.inv(calib.camera_matrix)
        depth = radius * 0.45
        ends = []
        for uv in np.array([[0.0, 0.0, 1.0], [w_px, 0.0, 1.0],
                            [w_px, h_px, 1.0], [0.0, h_px, 1.0]]):
            d_c = Kinv @ uv
            d_c = d_c / d_c[2]
            ends.append(c_m + (R.T @ d_c) * depth)
        for ep in ends:
            seg(c_m, ep, COL_CAM)
        for i in range(4):
            seg(ends[i], ends[(i + 1) % 4], COL_CAM)
        text_at(c_m, f"CAM{c_id}", COL_CAM, dy=-12, scale=0.5)

    if not exts:
        cv2.putText(panel, "NO BASE TAG LOCK - waiting for Tag 0", (16, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 110, 160), 1, cv2.LINE_AA)

    # ---- Legend ----------------------------------------------------------
    cv2.rectangle(panel, (10, height - 72), (width - 10, height - 12), (25, 28, 34), -1)
    draw_shadow_text(panel, (18, height - 52),
                     "X=red Y=grn Z=blu | PlaneA cyan | PlaneB orange",
                     (200, 205, 215), scale=0.35)
    draw_shadow_text(panel, (18, height - 32),
                     "Cube magenta | H frame yellow | Cam frustum blue",
                     (200, 205, 215), scale=0.35)
    return panel


def render_hud_dashboard(results: dict, custom_h_frame_active: bool = False,
                         width: int = 640, height: int = 720) -> np.ndarray:
    """Render a dark-themed telemetry dashboard with all metric coordinates."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (18, 20, 24)  # Sleek dark background

    # Header
    cv2.rectangle(panel, (0, 0), (width, 50), (28, 32, 40), -1)
    cv2.putText(panel, "3D Multi-Feature Tracker Telemetry", (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

    y = 75

    # Section 1: Base Reference (Tag 0)
    cv2.rectangle(panel, (14, y - 18), (width - 14, y + 40), (25, 30, 38), -1)
    b0_status = "LOCKED" if results.get("cam0_has_base") else "SEARCHING"
    b1_status = "LOCKED" if results.get("cam1_has_base") else "SEARCHING"
    b0_col = (0, 255, 100) if results.get("cam0_has_base") else (80, 80, 220)
    b1_col = (0, 255, 100) if results.get("cam1_has_base") else (80, 80, 220)

    cv2.putText(panel, "BASE REFERENCE FRAME (Tag 0, 134 mm)", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(panel, f"Cam 0: {b0_status}", (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, b0_col, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Cam 1: {b1_status}", (180, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, b1_col, 1, cv2.LINE_AA)
    y += 45

    # Section 2: Cube 3D Centroid (Tags 20..25)
    cube = results.get("cube_centroid")
    cv2.rectangle(panel, (14, y - 18), (width - 14, y + 80), (25, 30, 38), -1)
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
    cv2.rectangle(panel, (14, y - 18), (width - 14, y + 80), (25, 30, 38), -1)
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
    cv2.rectangle(panel, (14, y - 18), (width - 14, y + 100), (25, 30, 38), -1)
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
    y += 85

    # Section 5: Custom Baseline Reference Frame (Key 'H')
    h_info = results.get("custom_h_frame")
    status_h = "ACTIVE" if custom_h_frame_active else "READY (Press 'H' to Activate)"
    status_col = (0, 255, 120) if custom_h_frame_active else (140, 180, 220)
    cv2.rectangle(panel, (14, y - 18), (width - 14, y + 90), (25, 30, 38), -1)
    cv2.putText(panel, f"CUSTOM BASELINE FRAME [KEY 'H']: {status_h}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_col, 1, cv2.LINE_AA)
    y += 20
    if h_info:
        ox, oy, oz = h_info["origin_base_mm"]
        cv2.putText(panel, f"Origin (+2cm X, +5cm Z): [{ox:+6.1f}, {oy:+6.1f}, {oz:+6.1f}] mm",
                    (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
        y += 18
        if "cube_centroid_in_h_mm" in h_info:
            cx, cy, cz = h_info["cube_centroid_in_h_mm"]
            cv2.putText(panel, f"Cube in Custom Frame: [{cx:+6.1f}, {cy:+6.1f}, {cz:+6.1f}] mm",
                        (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 100, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, "Cube in Custom Frame: Waiting for cube detection...",
                        (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 160), 1, cv2.LINE_AA)
        y += 18
        cv2.putText(panel, "Spec: X=Top(40->41), Y=Right, Z=Cam -> RotY(90) -> Z=UP",
                    (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 200, 160), 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "Status: Requires Tags 40 & 41 baseline to be visible",
                    (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 150), 1, cv2.LINE_AA)
    y += 75

    # Key controls footer
    cv2.line(panel, (14, height - 36), (width - 14, height - 36), (60, 65, 75), 1)
    cv2.putText(panel, "Keys: 'H'=Custom Frame | 'V'=3D Scene | 'S'=Export JSON | 'Q'=Quit",
                (18, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 150, 170), 1, cv2.LINE_AA)

    return panel
