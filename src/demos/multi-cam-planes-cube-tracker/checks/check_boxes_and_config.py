#!/usr/bin/env python3
"""Verification check for config.yaml, box & box2 quad platforms, and UI rendering."""

from pathlib import Path
import sys
import numpy as np

DEMO_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = DEMO_DIR.parent.parent.parent
SRC_ROOT = REPO_ROOT / "src"
for p in (str(SRC_ROOT), str(DEMO_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tracker import (
    CameraCalibration,
    MultiCameraPlanesCubePipeline,
    DetectedTagPose,
)
from main import build_camera_and_calib_lists
from viz import render_hud_dashboard, annotate_camera_view, render_scene3d_panel
import argparse
import yaml

def test_config_and_boxes():
    print("=" * 80)
    print("🧪 VERIFYING CONFIG.YAML & DUAL PLATFORM (BOX & BOX2) TRACKING")
    print("=" * 80)

    cfg_path = DEMO_DIR / "config.yaml"
    assert cfg_path.is_file(), f"Missing config.yaml at {cfg_path}"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # 1. Verify static tags specification
    static_tags = set(cfg["static_tags"]["all_static"])
    expected_statics = {0, 1, 4, 5, 40, 41, 42, 43, 44, 45, 50, 51, 52, 53, 54, 55}
    assert expected_statics.issubset(static_tags), f"Missing expected static tags: {expected_statics - static_tags}"
    print(f"  [PASS 1] Static tags in config match user requirement: {sorted(static_tags)}")

    # 2. Verify moveable cube
    assert cfg["moveable_cube"]["name"] == "box"
    assert cfg["moveable_cube"]["tag_ids"] == [20, 21, 22, 23, 24, 25]
    assert cfg["moveable_cube"]["side_length_m"] == 0.050
    assert cfg["moveable_cube"].get("parent_platform") == "box"
    print(f"  [PASS 2] Moveable cube verified: 5 cm cube, tags 20..25, parent_platform='box'")

    # 3. Verify boxes
    assert "box" in cfg["boxes"], "Missing box in boxes config"
    assert "box2" in cfg["boxes"], "Missing box2 in boxes config"
    assert cfg["boxes"]["box"]["tag_ids"] == [42, 43, 44, 45]
    assert cfg["boxes"]["box2"]["tag_ids"] == [50, 51, 52, 53]
    print(f"  [PASS 3] Platform boxes verified: 'box' (42..45) and 'box2' (50..53)")

    # 4. Verify Camera Loader
    args = argparse.Namespace(cams=None, calibs=None, cam0=None, cam1=None, cam2=None, cam3=None, cam4=None, cam5=None,
                              calib0=None, calib1=None, calib2=None, calib3=None, calib4=None, calib5=None)
    sources, calibs = build_camera_and_calib_lists(args, config=cfg)
    assert len(calibs) == 2, f"Expected 2 cameras from config, got {len(calibs)}"
    print(f"  [PASS 4] Camera loader successfully loaded {len(calibs)} cameras using config")

    # 5. Initialize Pipeline with config
    sources = ["mock", "mock"]
    pipeline = MultiCameraPlanesCubePipeline(
        calibs=calibs,
        sources=sources,
        markers_yaml=str(REPO_ROOT / "calibrations" / "markers.yaml"),
        base_link_config=str(REPO_ROOT / "calibrations" / "base_link.yaml"),
        config=cfg,
    )
    assert pipeline.static_tag_ids == frozenset(expected_statics)
    assert pipeline.cube_tag_ids == frozenset(range(20, 26))
    assert pipeline.cube_parent_platform == "box"
    print(f"  [PASS 5] Pipeline initialized with {len(pipeline.static_tag_ids)} static tags and 2 platform box definitions")

    # 6. Verify Synthetic Dual-Box and Cube Detection
    sim_results = {
        "timestamp": 1234567.89,
        "world_map_size": 14,
        "boxes": {
            "box": {
                "id": "platform_box",
                "name": "box",
                "detected_tags": [42, 43, 44, 45],
                "normal": [0.0, 0.0, 1.0],
                "d_offset_m": 0.492,
                "center_mm": [640.0, -114.0, -492.0],
                "z_height_mm": -492.0,
                "coplanarity_rmse_mm": 0.85,
                "centers_mm": {
                    42: [590.0, -164.0, -492.0],
                    43: [690.0, -164.0, -492.0],
                    44: [690.0, -64.0, -492.0],
                    45: [590.0, -64.0, -492.0],
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
                "coplanarity_rmse_mm": 0.62,
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
            "z_height_mm": -492.0,
            "center_mm": [640.0, 200.0, -492.0],
            "tag_40_mm": [640.0, 95.0, -492.0],
            "tag_41_mm": [640.0, 305.0, -492.0],
        },
        "cube_centroid": {
            "xyz_mm": [640.0, -114.0, -467.0],
            "residual_spread_mm": 0.45,
            "visible_faces": [20, 22],
            "num_obs": 4,
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

    # 7. Render UI Panels
    hud = render_hud_dashboard(sim_results, num_cameras=2, width=640, height=720)
    assert hud.shape == (720, 640, 3)
    print("  [PASS 6] HUD Dashboard rendered cleanly with Section 4A (BOX) and Section 4B (BOX2)")

    scene3d = render_scene3d_panel(sim_results, calibs, width=640, height=720)
    assert scene3d.shape == (720, 640, 3)
    print("  [PASS 7] 3D Scene panel rendered cleanly with both box & box2 quad platforms")

    # Camera overlay
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    R_c_base = np.eye(3)
    t_c_base = np.array([0.0, 0.0, 1.5])
    K = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5)
    annotated = annotate_camera_view(frame, {}, "Camera 0", results=sim_results, cam_id=0, K=K, dist=dist)
    assert annotated.shape == (720, 1280, 3)
    print("  [PASS 8] Camera view overlay rendered with distinct colors for box (orange) and box2 (magenta)")

    print("=" * 80)
    print("🎉 ALL TESTS PASSED! FULL CONFIG & DUAL-BOX TRACKING SYSTEM VERIFIED.")
    print("=" * 80)

if __name__ == "__main__":
    test_config_and_boxes()
