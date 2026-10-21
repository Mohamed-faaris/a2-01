#!/usr/bin/env python3
"""Offline verification for the Single-Camera tracker.

Covers:
  1. Projection round-trip math (Base Frame <-> pixels).
  2. Physics-constrained geometry units:
     - shared plane normal from coplanar markers (fit_platform_plane_constrained)
     - cube face snapping parallel/perpendicular to the plane (snap_face_normal_to_axes)
     - constrained cube centroid + on-plane gap + outlier rejection
  3. Drawn overlay pixels appear at the projected positions.
  4. 3D scene panel + detections panel + composite UI (1920x720).
  5. Session logger: outputs/<name>/<time>/ history + photos.
  6. Real detector end-to-end smoke on a synthetic render.

Run from anywhere:
    PYTHONPATH=src .venv/bin/python src/demos/1-cam-planes-cube-tracker/checks/check_overlays.py
"""
import csv
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
    SingleCameraPlanesCubePipeline,
    detect_and_estimate_poses,
    estimate_cube_centroid_constrained,
    fit_platform_plane_constrained,
    project_tag_to_base,
    snap_face_normal_to_axes,
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


pipe = SingleCameraPlanesCubePipeline(calib=calib("cam0"))


def unit(v):
    return v / np.linalg.norm(v)


def rot_from_z(z):
    """Any right-handed rotation whose local +Z maps onto z."""
    z = unit(np.asarray(z, dtype=float))
    ref = np.array([0.0, 1.0, 0.0]) if abs(np.dot(z, [0.0, 1.0, 0.0])) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = unit(np.cross(ref, z))
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


# ===========================================================================
# 1. Synthetic well-posed tag-0 pose (0.55 m ahead, tilted)
# ===========================================================================
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

# ===========================================================================
# 2. Physics-constrained geometry units
# ===========================================================================
rng = np.random.default_rng(7)

# --- 2a. shared plane normal from 4 coplanar markers ----------------------
n_gt = unit(np.array([0.08, 0.05, 1.0]))
d_gt = 0.06
ref = np.array([0.0, 1.0, 0.0]) if abs(np.dot(n_gt, [0.0, 1.0, 0.0])) < 0.9 else np.array([1.0, 0.0, 0.0])
u_gt = unit(np.cross(ref, n_gt))
v_gt = np.cross(n_gt, u_gt)
plane_pts = [d_gt * n_gt + a * u_gt + b * v_gt
             for a, b in [(-0.12, -0.08), (0.12, -0.08), (0.12, 0.08), (-0.12, 0.08)]]
tag_data = []
for i, c_gt in enumerate(plane_pts):
    n_obs = unit(n_gt + 0.01 * rng.standard_normal(3))          # ~1 deg normal noise
    c_obs = c_gt + 0.001 * rng.standard_normal(3)               # 1 mm center noise
    tag_data.append((42 + i, c_obs, n_obs))

fit = fit_platform_plane_constrained(tag_data)
ang_n = float(np.degrees(np.arccos(np.clip(np.dot(fit["normal"], n_gt), -1, 1))))
errs = [float(np.dot(fit["normal"], c) - fit["offset_m"]) for c in plane_pts]
print(f"[2a] shared plane: normal err {ang_n:.3f} deg, flatness RMSE {fit['rmse_mm']:.2f} mm, "
      f"inliers {fit['inlier_tids']}")
assert ang_n < 1.0, "shared plane normal wrong"
assert fit["rmse_mm"] < 5.0 and len(fit["inlier_tids"]) == 4, "plane fit quality"
assert max(abs(np.dot(fit["normal"], c) - fit["offset_m"])
           for c in fit["centers_proj"].values()) < 1e-9, \
    "projected centers not coplanar"
gt_off = max(abs(np.dot(fit["normal"], c) - fit["offset_m"]) for c in plane_pts)
print(f"[2a] GT plane points vs fit plane: max {gt_off * 1000:.2f} mm (noise-limited)")
assert gt_off < 0.003, "fitted plane deviates from GT points"

# --- 2b. face normal snapping ---------------------------------------------
n_pl = fit["normal"]
s_top, a_top, k_top = snap_face_normal_to_axes(n_pl, n_pl)
s_bot, a_bot, k_bot = snap_face_normal_to_axes(-n_pl, n_pl)
ref = np.array([0.0, 1.0, 0.0]) if abs(np.dot(n_pl, [0.0, 1.0, 0.0])) < 0.9 else np.array([1.0, 0.0, 0.0])
side_dir = unit(np.cross(n_pl, ref))   # exactly perpendicular to the plane normal
s_side, a_side, k_side = snap_face_normal_to_axes(side_dir, n_pl)
tilted = unit(np.cos(np.radians(70)) * n_pl + np.sin(np.radians(70)) * side_dir)  # 70 deg from n
s_tilt, a_tilt, k_tilt = snap_face_normal_to_axes(tilted, n_pl)
print(f"[2b] snapping: top {k_top}/{a_top:.2f}d, bottom {k_bot}/{a_bot:.2f}d, "
      f"side {k_side}/{a_side:.2f}d, tilted70 {k_tilt}/{a_tilt:.2f}d")
assert (k_top, k_bot, k_side) == ("top", "bottom", "side")
assert a_top < 1e-9 and a_bot < 1e-9 and a_side < 1e-9
assert k_tilt == "side" and abs(a_tilt - 20.0) < 1e-6   # 70 deg -> 20 deg off the side axis

# --- 2c. constrained cube centroid + gap + outlier rejection ---------------
side_m = 0.041
c_gt = d_gt * n_pl + 0.03 * u_gt - 0.02 * v_gt + (side_m / 2.0) * n_pl  # resting on plane
R_id, t_zero = np.eye(3), np.zeros(3)


def obs_from_pose(center_base, normal_base):
    """(tid, R_c_t, t_c_t, R_c_base, t_c_base) with base == 'camera' frame."""
    return rot_from_z(normal_base), np.asarray(center_base, dtype=float)


obs = []
# top face (20): normal ~ n with 1 deg noise
n_top_obs = unit(n_pl + 0.017 * rng.standard_normal(3))
obs.append((20, *obs_from_pose(c_gt + (side_m / 2.0) * n_pl + 0.0005 * rng.standard_normal(3), n_top_obs)))
# side face A (24): normal ~ u with noise
n_side_obs = unit(side_dir + 0.017 * rng.standard_normal(3))
obs.append((24, *obs_from_pose(c_gt + (side_m / 2.0) * side_dir + 0.0005 * rng.standard_normal(3), n_side_obs)))
# side face B (23): orthogonal side
other_side = unit(np.cross(n_pl, side_dir))
n_side2_obs = unit(other_side + 0.017 * rng.standard_normal(3))
obs.append((23, *obs_from_pose(c_gt + (side_m / 2.0) * other_side + 0.0005 * rng.standard_normal(3), n_side2_obs)))
# outlier (22): normal 50 deg off BOTH allowed axes -> must be rejected
bad_normal = unit(np.cos(np.radians(50)) * n_pl + np.sin(np.radians(50)) * side_dir)
obs.append((22, *obs_from_pose(c_gt + np.array([0.0, 0.0, 0.2]), bad_normal)))
# pack with identity extrinsics
obs = [(tid, Rc, tc, R_id, t_zero) for tid, Rc, tc in obs]

cube = estimate_cube_centroid_constrained(obs, side_m, n_pl, d_gt)
xyz_err = float(np.linalg.norm(np.array(cube["xyz_mm"]) / 1000.0 - c_gt))
print(f"[2c] cube: xyz err {xyz_err * 1000:.2f} mm, gap {cube['on_plane_gap_mm']:+.2f} mm, "
      f"faces {cube['visible_faces']} (outlier 22 rejected: {22 not in cube['visible_faces']}), "
      f"max_snap {cube['max_snap_deg']:.1f} deg, yaw {cube['yaw_deg']:.1f} deg")
assert cube is not None and 22 not in cube["visible_faces"], "outlier face not rejected"
assert cube["visible_faces"] == [20, 23, 24], "constrained faces wrong"
assert xyz_err < 0.003, "cube centroid worse than 3 mm"
assert abs(cube["on_plane_gap_mm"]) < 3.0, "resting cube should have ~0 gap"
assert cube["face_snaps"][20]["kind"] == "top", "top face not classified top"
assert cube["yaw_deg"] is not None, "yaw missing"

# ===========================================================================
# 3. Results dict with injected Plane A / Plane B / Cube (Base mm)
# ===========================================================================
h2 = side_m / 2.0 * 1000.0
results = {
    "timestamp": 0.0, "cam_has_base": True,
    "cam_tags": [0, 42, 43, 44, 45, 20],
    "detections": [
        {"id": 0, "margin": 30.0, "entity": "BASE"},
        {"id": 42, "margin": 140.0, "entity": "Plane B"},
        {"id": 20, "margin": 95.0, "entity": "Cube"},
    ],
    "cam_extrinsics": {0: {"R_c_base": R_c_base.tolist(), "t_c_base_m": t_c_base.tolist(),
                           "position_base_mm": ((-R_c_base.T @ t_c_base) * 1000).tolist()}},
    "plane_a": {"tag_40_mm": [-60.0, 20.0, -120.0], "tag_41_mm": [60.0, 20.0, -120.0],
                "baseline_dist_mm": 120.0, "center_mm": [0.0, 20.0, -120.0], "z_height_mm": -120.0},
    "plane_b": {"detected_tags": [42, 43, 44], "inlier_tags": [42, 43, 44],
                "normal": [0.0, 0.0, 1.0], "shared_normal": True, "offset_m": 0.180,
                "center_mm": [0.0, 73.0, -180.0], "z_height_mm": -180.0,
                "coplanarity_rmse_mm": 1.2,
                "centers_mm": {42: [-70.0, 60.0, -180.0], 43: [70.0, 60.0, -180.0],
                               44: [0.0, 100.0, -180.0]}},
    "cube_centroid": {"xyz_mm": [0.0, 0.0, -150.0], "xyz_contact_mm": [0.0, 0.0, -149.4],
                      "on_plane_gap_mm": 0.6, "max_snap_deg": 4.2, "yaw_deg": 3.1,
                      "visible_faces": [20], "num_obs": 1, "inlier_count": 1,
                      "residual_spread_mm": 0.5,
                      "face_snaps": {20: {"kind": "top", "snap_deg": 4.2}}},
}
results["cube_faces_mm"] = {20: [[-h2, -h2, -130.0], [h2, -h2, -130.0],
                                 [h2, h2, -130.0], [-h2, h2, -130.0]]}
results["custom_h_frame"] = {"origin_base_mm": [0.0, 20.0, -120.0],
                             "origin_base_m": [0.0, 0.02, -0.12],
                             "R_custom_base": np.eye(3).tolist()}

# ===========================================================================
# 4. Drawn overlay pixels must appear at the projected positions
# ===========================================================================
gray0 = cv2.cvtColor(BG, cv2.COLOR_BGR2GRAY)
dets0 = detect_and_estimate_poses(gray0, K, D, pipe.detector, pipe.registry, 1.0)
v0 = pipe.annotate_camera_view(BG.copy(), dets0, "cam0", results)


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

# ===========================================================================
# 5. 3D scene panel + detections panel + composite UI
# ===========================================================================
panel = pipe.render_scene3d_panel(results, 640, 720)
bg_mask = np.all(np.abs(panel.astype(int) - np.array([18, 20, 24])) <= 2, axis=2)
feature_px = int((~bg_mask).sum()) - 640 * 50
print(f"[4] scene panel feature pixels: {feature_px}")
assert feature_px > 4000, "scene panel looks empty"

dets_panel = pipe.render_detections_panel(results, 640, 360)
hud = pipe.render_hud_dashboard(results, 640, 720)
cam_column = np.vstack([cv2.resize(v0, (640, 360)), dets_panel])
ui = np.hstack([cam_column, panel, hud])
print(f"[5] composite UI shape: {ui.shape}")
assert ui.shape == (720, 1920, 3), ui.shape
Path(REPO_ROOT / "tmp").mkdir(exist_ok=True)
cv2.imwrite(str(REPO_ROOT / "tmp/overlay_check_1cam.png"), ui)

# ===========================================================================
# 6. Session logger: outputs/<name>/<time>/ history rows + photos
# ===========================================================================
sess_root = REPO_ROOT / "tmp/session_check_1cam"
shutil.rmtree(sess_root, ignore_errors=True)
sess = SessionLogger(sess_root, history_interval_s=0.0, photo_interval_s=0.5)
for _ in range(3):
    sess.record(results, ui)
sess.close(final_frame=ui)
hist = sorted(sess_root.glob("*/history.csv"))[0]      # <root>/<time>/history.csv
rows = list(csv.DictReader(open(hist)))
photos = sorted(sess_root.glob("*/frames/*.png"))
assert hist.parent.name.replace("_", "").isdigit(), f"time folder expected, got {hist.parent.name}"
print(f"[6] session dir {hist.parent.relative_to(sess_root)} (time-named), "
      f"rows={len(rows)}, photos={len(photos)}, "
      f"cube_xyz row=[{rows[-1]['cube_x_mm']}, {rows[-1]['cube_y_mm']}, {rows[-1]['cube_z_mm']}]")
assert len(rows) == 3 and len(photos) == 2
assert rows[-1]["cube_x_mm"] == "0.000" and rows[-1]["plane_b_rmse_mm"] == "1.200"
assert rows[-1]["cam_has_base"] == "1"

# ===========================================================================
# 7. Real detector still runs end-to-end on the render
# ===========================================================================
smoke = pipe.process_frame(BG)
print(f"[7] process_frame smoke: tags={smoke['cam_tags']}, "
      f"base={smoke['cam_has_base']}, extrinsics={'cam_extrinsics' in smoke}, "
      f"plane_b={'shared_normal' if (smoke.get('plane_b') or {}).get('shared_normal') else 'n/a'}, "
      f"cube={'constrained' if smoke.get('cube_centroid') else 'n/a'}")
assert "cam_has_base" in smoke and ("cam_extrinsics" in smoke) == (0 in smoke["cam_tags"])
if smoke.get("cube_centroid"):
    assert "on_plane_gap_mm" in smoke["cube_centroid"], "cube result missing on-plane gap"
if smoke.get("plane_b"):
    assert smoke["plane_b"]["shared_normal"] is True, "plane_b missing shared normal"

# ===========================================================================
# 8. Regression: process_frame assembly with plane+cube detections injected.
#    The exp20 render has no tags 42..45/20..25, so the plane_b/cube dict
#    comps only ever ran on a live camera (where the tuple-unpack crash
#    happened). Pin the whole assembly offline with exact synthetic poses.
# ===========================================================================
import tracker as tracker_mod  # noqa: E402


def _det(tid, center_base, R_b_t):
    t_c = R_c_base @ np.asarray(center_base, dtype=float) + t_c_base
    return DetectedTagPose(tid, np.zeros((4, 2)), np.zeros(2),
                           R_c_base @ np.asarray(R_b_t, dtype=float), t_c, 50.0)


R_flat = rot_from_z(np.array([0.0, 0.0, 1.0]))     # tag facing +Z (up)
R_side_x = rot_from_z(np.array([1.0, 0.0, 0.0]))   # face normal +X
R_side_ny = rot_from_z(np.array([0.0, -1.0, 0.0]))  # face normal -Y

side_m = pipe.cube_side_m
plane_z = 0.06
centroid_gt = np.array([0.01, 0.03, plane_z + side_m / 2.0])

fake_dets = {0: det0}
for tid, xy in [(42, (-0.08, -0.06)), (43, (0.08, -0.06)),
                (44, (0.08, 0.06)), (45, (-0.08, 0.06))]:
    fake_dets[tid] = _det(tid, np.array([xy[0], xy[1], plane_z]), R_flat)
fake_dets[40] = _det(40, np.array([-0.05, -0.12, 0.0]), R_flat)
fake_dets[41] = _det(41, np.array([0.05, -0.12, 0.0]), R_flat)
fake_dets[20] = _det(20, centroid_gt + np.array([0.0, 0.0, side_m / 2.0]), R_flat)
fake_dets[24] = _det(24, centroid_gt + np.array([side_m / 2.0, 0.0, 0.0]), R_side_x)
fake_dets[23] = _det(23, centroid_gt - np.array([0.0, side_m / 2.0, 0.0]), R_side_ny)

orig_detect = tracker_mod.detect_and_estimate_poses
tracker_mod.detect_and_estimate_poses = lambda gray, K_, D_, det_, reg_, ar=1.0: dict(fake_dets)
try:
    asm = pipe.process_frame(BG)
finally:
    tracker_mod.detect_and_estimate_poses = orig_detect

pb = asm.get("plane_b")
assert pb and pb["shared_normal"] is True and pb["detected_tags"] == [42, 43, 44, 45], pb
assert abs(pb["offset_m"] - plane_z) < 1e-6, pb["offset_m"]
assert pb["coplanarity_rmse_mm"] < 1e-6, pb["coplanarity_rmse_mm"]
assert sorted(pb["corners_mm"]) == [42, 43, 44, 45]
assert pb["corners_mm"][42] == pb["centers_mm"][42], \
    "corners_mm must carry tag CENTERS (tuple-unpack regression wrote normals instead)"

cb = asm.get("cube_centroid")
assert cb, "constrained cube missing with injected faces"
xyz_err = float(np.linalg.norm(np.array(cb["xyz_mm"]) / 1000.0 - centroid_gt)) * 1000.0
print(f"[8] assembly: plane_b offset {pb['offset_m'] * 1000:.1f} mm flatness "
      f"{pb['coplanarity_rmse_mm']:.4f} mm | cube xyz err {xyz_err:.3f} mm, "
      f"gap {cb['on_plane_gap_mm']:+.3f} mm, faces {cb['visible_faces']}, "
      f"H-frame {'cube_centroid_in_h_mm' in (asm.get('custom_h_frame') or {})}")
assert xyz_err < 0.5, xyz_err
assert abs(cb["on_plane_gap_mm"]) < 0.5, cb["on_plane_gap_mm"]
assert cb["visible_faces"] == [20, 23, 24], cb["visible_faces"]
hf = asm.get("custom_h_frame") or {}
assert "cube_centroid_in_h_mm" in hf, "H-frame missing cube coordinates"
assert asm.get("plane_a"), "plane A missing with injected tags 40/41"

print("ALL CHECKS PASSED")
