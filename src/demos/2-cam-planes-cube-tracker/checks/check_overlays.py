#!/usr/bin/env python3
"""Offline verification of the 3D overlay projection math + scene panel + session logger.

Uses a synthetic, well-posed tag-0 pose so the check is independent of the
detector/PnP quality on any particular render.

Run from anywhere:
    PYTHONPATH=src .venv/bin/python src/demos/2-cam-planes-cube-tracker/checks/check_overlays.py
"""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

DEMO_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = DEMO_DIR.parent.parent.parent
for _p in (str(REPO_ROOT / "src"), str(DEMO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tracker import (  # noqa: E402
    CameraCalibration,
    DetectedTagPose,
    TwoCameraPlanesCubePipeline,
    detect_and_estimate_poses,
    project_tag_to_base,
    solve_camera_extrinsics_from_base,
)
from viz import (  # noqa: E402
    COL_CUBE,
    COL_PLANE_A,
    COL_PLANE_B,
    SessionLogger,
    project_base_point,
)

BG = cv2.imread(str(REPO_ROOT / "experiment/20_cube_centroid_and_target_tag/001/inputs/cam0_render.png"))
assert BG is not None, "missing experiment render"

# True intrinsics of the synthetic render (experiment/20 .../ground_truth.json)
K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
D = np.zeros(5)


def calib(name: str) -> CameraCalibration:
    return CameraCalibration(name=name, camera_matrix=K, dist_coeffs=D, image_size=(1280, 720))


pipe = TwoCameraPlanesCubePipeline(calib0=calib("cam0"), calib1=calib("cam1"))

# --- 1. Synthetic well-posed tag-0 pose (0.55 m ahead, tilted) --------------
rx, ry = np.radians(20.0), np.radians(-15.0)
Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
R_c_t = Rx @ Ry
t_c_t = np.array([0.02, -0.01, 0.55])
hh = pipe.registry.get_size(0) / 2.0
obj = np.array([[-hh, -hh, 0], [hh, -hh, 0], [hh, hh, 0], [-hh, hh, 0]], dtype=np.float64)
corners_px, _ = cv2.projectPoints(obj, cv2.Rodrigues(R_c_t)[0], t_c_t, K, D)
center_px, _ = cv2.projectPoints(np.zeros(3), cv2.Rodrigues(R_c_t)[0], t_c_t, K, D)
det0 = DetectedTagPose(0, corners_px.reshape(-1, 2), center_px.reshape(2), R_c_t, t_c_t, 30.0)

R_c_base, t_c_base = solve_camera_extrinsics_from_base(det0)

# Base origin must land exactly on the projected tag center; corners must round-trip
uv_o = project_base_point(np.zeros(3), R_c_base, t_c_base, K, D)
e_center = float(np.linalg.norm(uv_o - center_px.reshape(2)))
corners_cam = t_c_t + (R_c_t @ obj.T).T                  # tag corners in camera frame
corners_base = (R_c_base.T @ (corners_cam - t_c_base).T).T  # ... and in base frame
corner_errs = []
for p_b, px_gt in zip(corners_base, det0.corners_px):
    uv = project_base_point(p_b, R_c_base, t_c_base, K, D)
    corner_errs.append(float(np.linalg.norm(uv - px_gt)))
print(f"[1] synthetic round-trip: origin {e_center:.2e} px, corners max {max(corner_errs):.2e} px")
assert e_center < 1e-6 and max(corner_errs) < 1e-6, "projection math mismatch"

# --- 2. Results dict with injected Plane A / Plane B / Cube (Base mm) -------
# Features are placed between the camera and the base tag (camera sits at -Z
# in the base frame for this pose) so they land inside the image.
results = {
    "timestamp": 0.0, "cam0_has_base": True, "cam1_has_base": False,
    "cam0_tags": [0], "cam1_tags": [],
    "cam_extrinsics": {0: {"R_c_base": R_c_base.tolist(), "t_c_base_m": t_c_base.tolist(),
                           "position_base_mm": ((-R_c_base.T @ t_c_base) * 1000).tolist()}},
    "plane_a": {"tag_40_mm": [-60.0, 20.0, -120.0], "tag_41_mm": [60.0, 20.0, -120.0],
                "baseline_dist_mm": 120.0, "center_mm": [0.0, 20.0, -120.0], "z_height_mm": -120.0},
    "plane_b": {"detected_tags": [42, 43, 44], "normal": [0.0, 0.0, 1.0],
                "center_mm": [0.0, 73.0, -180.0], "z_height_mm": -180.0, "coplanarity_rmse_mm": 1.2,
                "centers_mm": {42: [-70.0, 60.0, -180.0], 43: [70.0, 60.0, -180.0],
                               44: [0.0, 100.0, -180.0]}},
    "cube_centroid": {"xyz_mm": [0.0, 0.0, -150.0], "visible_faces": [20],
                      "num_obs": 1, "residual_spread_mm": 0.5},
}
h2 = 20.5
results["cube_faces_mm"] = {20: [[-h2, -h2, -130.0], [h2, -h2, -130.0],
                                 [h2, h2, -130.0], [-h2, h2, -130.0]]}
results["custom_h_frame"] = {"origin_base_mm": [0.0, 20.0, -120.0],
                             "origin_base_m": [0.0, 0.02, -0.12],
                             "R_custom_base": np.eye(3).tolist()}

# --- 3. Drawn overlay pixels must appear at the projected positions ---------
gray0 = cv2.cvtColor(BG, cv2.COLOR_BGR2GRAY)
dets0 = detect_and_estimate_poses(gray0, K, D, pipe.detector0, pipe.registry, 1.0)
v0 = pipe.annotate_camera_view(BG.copy(), dets0, "cam0", results, 0)


def near_color(img, uv, bgr, radius=14, tol=40):
    x, y = int(round(uv[0])), int(round(uv[1]))
    win = img[max(0, y - radius):y + radius, max(0, x - radius):x + radius].astype(int)
    return bool(np.any(np.all(np.abs(win - np.array(bgr)) <= tol, axis=2)))


checks = [
    ("planeA-40", results["plane_a"]["tag_40_mm"], COL_PLANE_A),
    ("planeA-41", results["plane_a"]["tag_41_mm"], COL_PLANE_A),
    ("planeB-43", results["plane_b"]["centers_mm"][43], COL_PLANE_B),
    ("cube-face", results["cube_faces_mm"][20][0], COL_CUBE),
    ("cube-centroid", results["cube_centroid"]["xyz_mm"], COL_CUBE),
]
for name, p_mm, col in checks:
    uv = project_base_point(np.array(p_mm) * 0.001, R_c_base, t_c_base, K, D)
    in_frame = 0 <= uv[0] < 1280 and 0 <= uv[1] < 720
    ok = in_frame and near_color(v0, uv, col)
    print(f"[3] {name:14s} @ px {np.round(uv, 1)} in-frame={in_frame} color-present={ok}")
    assert ok, f"{name} overlay pixel missing at projected position"

# --- 4. 3D scene panel + composite UI --------------------------------------
panel = pipe.render_scene3d_panel(results, 640, 720)
bg_mask = np.all(np.abs(panel.astype(int) - np.array([18, 20, 24])) <= 2, axis=2)
feature_px = int((~bg_mask).sum()) - 640 * 50
print(f"[4] scene panel feature pixels: {feature_px}")
assert feature_px > 4000, "scene panel looks empty"

hud = pipe.render_hud_dashboard(results, 640, 720)
cam_column = np.vstack([cv2.resize(v0, (640, 360)), cv2.resize(BG, (640, 360))])
ui = np.hstack([cam_column, panel, hud])
print(f"[5] composite UI shape: {ui.shape}")
assert ui.shape == (720, 1920, 3)
cv2.imwrite(str(REPO_ROOT / "tmp/overlay_check.png"), ui)

# --- 5. Session logger: history rows + photo collection --------------------
sess_root = REPO_ROOT / "tmp/session_check"
shutil.rmtree(sess_root, ignore_errors=True)
sess = SessionLogger(sess_root, history_interval_s=0.0, photo_interval_s=0.5)
for _ in range(3):
    sess.record(results, ui)
sess.close(final_frame=ui)
hist = sorted(sess_root.glob("session_*/history.csv"))[0]
import csv  # noqa: E402

rows = list(csv.DictReader(open(hist)))
photos = sorted(sess_root.glob("session_*/frames/*.png"))
print(f"[6] history rows={len(rows)}, photos={len(photos)}, "
      f"cube_xyz row=[{rows[-1]['cube_x_mm']}, {rows[-1]['cube_y_mm']}, {rows[-1]['cube_z_mm']}]")
assert len(rows) == 3 and len(photos) == 2
assert rows[-1]["cube_x_mm"] == "0.000" and rows[-1]["plane_b_rmse_mm"] == "1.200"

# --- 6. Smoke test: real detector still runs end-to-end on the render -------
smoke = pipe.process_frames(BG, BG.copy())
print(f"[7] process_frames smoke: tags={smoke['cam0_tags']}, extrinsics={'cam_extrinsics' in smoke}")
print("ALL CHECKS PASSED")
