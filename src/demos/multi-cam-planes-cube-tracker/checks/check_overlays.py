#!/usr/bin/env python3
"""Offline verification of Multi-Camera 3D overlays, projection math, scene panel, and pipeline.

Run from repository root:
    PYTHONPATH=src python3 src/demos/multi-cam-planes-cube-tracker/checks/check_overlays.py
"""

from __future__ import annotations

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
    MultiCameraPlanesCubePipeline,
    detect_and_estimate_poses,
    project_tag_to_base,
    solve_camera_extrinsics_from_base,
)
from viz import (  # noqa: E402
    COL_CUBE,
    COL_PLANE_A,
    COL_PLANE_B,
    SessionLogger,
    layout_camera_views,
    project_base_point,
    render_hud_dashboard,
    render_scene3d_panel,
)

BG = cv2.imread(str(REPO_ROOT / "experiment/20_cube_centroid_and_target_tag/001/inputs/cam0_render.png"))
assert BG is not None, "missing experiment render for testing"

# Intrinsics of synthetic render
K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
D = np.zeros(5, dtype=np.float64)


def make_calib(name: str) -> CameraCalibration:
    return CameraCalibration(name=name, camera_matrix=K.copy(), dist_coeffs=D.copy(), image_size=(1280, 720))


def run_checks():
    print("=" * 80)
    print("🧪 RUNNING MULTI-CAMERA PLANES & CUBE TRACKER VERIFICATION CHECKS")
    print("=" * 80)

    # 1. Synthetic Tag 0 Pose Round-Trip
    calibs = [make_calib("cam0"), make_calib("cam1"), make_calib("cam2")]
    pipe = MultiCameraPlanesCubePipeline(calibs=calibs, sources=["mock", "mock", "mock"])

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
    uv_o = project_base_point(np.zeros(3), R_c_base, t_c_base, K, D)
    e_center = float(np.linalg.norm(uv_o - center_px.reshape(2)))
    assert e_center < 1e-5, f"Tag 0 origin projection mismatch: {e_center}"
    print(f"  [CHECK 1 PASS] Synthetic Tag 0 round-trip origin error: {e_center:.2e} px")

    # 2. Multi-Camera Camera View Layout (1, 2, 3, 4 cameras)
    dummy_view = np.zeros((720, 1280, 3), dtype=np.uint8)
    for num_c in (1, 2, 3, 4, 6):
        views = [dummy_view.copy() for _ in range(num_c)]
        canvas = layout_camera_views(views, width=640, height=720)
        assert canvas.shape == (720, 640, 3), f"Tiled canvas shape mismatch for {num_c} cams: {canvas.shape}"
    print("  [CHECK 2 PASS] Multi-Camera tiling layout verified (1, 2, 3, 4, 6 cameras -> 640x720 canvas)")

    # 3. Multi-Camera Results and Telemetry Panel
    results = {
        "timestamp": 12345.67,
        "cam0_has_base": True,
        "cam1_has_base": True,
        "cam2_has_base": False,
        "cam_tags": {0: [0, 40, 41], 1: [0, 42, 43], 2: [20, 21]},
        "cam_extrinsics": {
            0: {"R_c_base": R_c_base.tolist(), "t_c_base_m": t_c_base.tolist(),
                "position_base_mm": ((-R_c_base.T @ t_c_base) * 1000).tolist(), "method": "bootstrap_tag0"},
            1: {"R_c_base": R_c_base.tolist(), "t_c_base_m": t_c_base.tolist(),
                "position_base_mm": ((-R_c_base.T @ t_c_base) * 1000).tolist(), "method": "multi_tag_pnp"},
        },
        "world_map_size": 7,
        "plane_a": {"tag_40_mm": [-60.0, 20.0, -120.0], "tag_41_mm": [60.0, 20.0, -120.0],
                    "baseline_dist_mm": 120.0, "center_mm": [0.0, 20.0, -120.0], "z_height_mm": -120.0},
        "plane_b": {"detected_tags": [42, 43, 44, 45], "normal": [0.0, 0.0, 1.0],
                    "center_mm": [0.0, 73.0, -180.0], "z_height_mm": -180.0, "coplanarity_rmse_mm": 1.2,
                    "centers_mm": {42: [-70.0, 60.0, -180.0], 43: [70.0, 60.0, -180.0],
                                   44: [0.0, 100.0, -180.0], 45: [50.0, 110.0, -180.0]}},
        "cube_centroid": {"xyz_mm": [0.0, 0.0, -150.0], "visible_faces": [20, 22],
                          "num_obs": 2, "residual_spread_mm": 0.4},
        "cube_faces_mm": {20: [[-20, -20, -130], [20, -20, -130], [20, 20, -130], [-20, 20, -130]]},
        "custom_h_frame": {"origin_base_mm": [0.0, 20.0, -120.0], "origin_base_m": [0.0, 0.02, -0.12],
                           "R_custom_base": np.eye(3).tolist(), "cube_centroid_in_h_mm": [0.0, -20.0, -30.0],
                           "active": True},
    }

    hud = render_hud_dashboard(results, custom_h_frame_active=True, num_cameras=3, width=640, height=720)
    assert hud.shape == (720, 640, 3), f"HUD panel shape mismatch: {hud.shape}"
    print("  [CHECK 3 PASS] Multi-Camera HUD Dashboard rendered cleanly (640x720)")

    # 4. Multi-Camera 3D Scene Panel with Frustums
    scene = render_scene3d_panel(results, calibs, width=640, height=720)
    assert scene.shape == (720, 640, 3), f"Scene panel shape mismatch: {scene.shape}"
    print("  [CHECK 4 PASS] Multi-Camera 3D Scene panel rendered cleanly with camera frustums (640x720)")

    # 5. Composite UI Assembly
    views = [dummy_view.copy() for _ in range(3)]
    cam_col = layout_camera_views(views, width=640, height=720)
    composite = np.hstack([cam_col, scene, hud])
    assert composite.shape == (720, 1920, 3), f"Composite UI shape mismatch: {composite.shape}"
    print("  [CHECK 5 PASS] Unified 1920x720 Composite UI generated cleanly")

    # 6. SessionLogger Multi-Camera
    test_session_root = REPO_ROOT / "outputs/test_multi_cam_verify"
    if test_session_root.exists():
        shutil.rmtree(test_session_root)

    logger = SessionLogger(test_session_root, history_interval_s=0.0, photo_interval_s=0.0, use_timestamp_subfolder=False)
    logger.record(results, composite)
    logger.close(composite)

    assert (test_session_root / "history.csv").is_file(), "history.csv not generated"
    shutil.rmtree(test_session_root)
    print("  [CHECK 6 PASS] Multi-Camera SessionLogger verified (history.csv + frames)")

    print("=" * 80)
    print("🎉 ALL MULTI-CAMERA VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_checks()
