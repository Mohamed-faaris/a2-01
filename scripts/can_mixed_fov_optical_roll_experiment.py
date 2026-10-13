#!/usr/bin/env python3
"""Experiment: Mixed FOVs and In-Plane Z-Axis (Roll) Camera Rotations.

Features:
  - Heterogeneous Camera FOVs:
      e.g., Cam0 wide (75° to 90°) & Cam1 narrow (45° to 60°).
  - Rotations around the Camera's local optical Z-axis (In-Plane Roll):
      Each camera rotates by arbitrary roll angles (e.g., -45° to +45° or up to 90°)
      around its optical view axis pointing at the workspace.
  - Floating 3D cans placed at varied heights and positions.
  - AprilTag fiducial reference marker on the ground plane.
  - Full multi-method evaluation (DLT & Ray Midpoint across YOLO / Color Contour).

Follows TESTING_PROCEDURE.md sequential ID numbering:
  experiment/10_can_mixed_fov_optical_roll/
    shared/
    reports/
    <scene_id>/
      inputs/
      artifacts/
      outputs/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from pupil_apriltags import Detector
from ultralytics import YOLO

from can_rel_marker_experiment import (
    WIDTH, HEIGHT, CAN_RADIUS_M, CAN_HEIGHT_M,
    ensure_marker_image, get_cv_extrinsics
)
from multi_method_benchmark_30 import (
    triangulate_dlt, triangulate_ray_midpoint,
    detect_cans_yolo, detect_cans_color_contour, evaluate_methodology
)

DEFAULT_YOLO_MODEL = "/home/mfk01/Projects/econ-systems/demos/demo2/yolo26s.pt"
DEFAULT_MARKER_SIZE_M = 0.088
DEFAULT_MARKER_ID = 3


def make_K(hfov_deg: float, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def generate_blender_scene_script_roll(
    artifacts_dir: Path,
    inputs_dir: Path,
    marker_path: Path,
    marker_size: float,
    marker_pos: tuple[float, float, float],
    cans: list[dict],
    cam0_pos: tuple[float, float, float],
    cam0_fov: float,
    cam0_roll_deg: float,
    cam1_pos: tuple[float, float, float],
    cam1_fov: float,
    cam1_roll_deg: float,
) -> str:
    escaped_art = str(artifacts_dir.resolve()).replace("\\", "\\\\").replace("'", "\\'")
    escaped_inp = str(inputs_dir.resolve()).replace("\\", "\\\\").replace("'", "\\'")
    escaped_marker = str(marker_path.resolve()).replace("\\", "\\\\").replace("'", "\\'")
    cans_repr = repr(cans)

    focal0_px = (WIDTH / 2.0) / math.tan(math.radians(cam0_fov / 2.0))
    focal1_px = (WIDTH / 2.0) / math.tan(math.radians(cam1_fov / 2.0))

    return f'''import bpy
import math
import os
import json
import mathutils

art_dir = r'{escaped_art}'
inp_dir = r'{escaped_inp}'
marker_img = r'{escaped_marker}'
os.makedirs(art_dir, exist_ok=True)
os.makedirs(inp_dir, exist_ok=True)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

scene = bpy.context.scene
available_engines = {{item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}}
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in available_engines else "BLENDER_EEVEE"
scene.render.resolution_x = {WIDTH}
scene.render.resolution_y = {HEIGHT}
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.view_settings.view_transform = "Standard"

# Ambient lighting
world = bpy.data.worlds.new("World")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.75, 0.75, 0.75, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
scene.world = world

# Floor
bpy.ops.mesh.primitive_plane_add(size=12.0, location=(0, 0, 0))
floor = bpy.context.active_object
mat_f = bpy.data.materials.new("FloorMat")
mat_f.use_nodes = True
mat_f.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.35, 0.35, 0.35, 1.0)
floor.data.materials.append(mat_f)

# AprilTag Marker (Emission)
image = bpy.data.images.load(marker_img, check_existing=True)
image.colorspace_settings.name = "Non-Color"
mat_m = bpy.data.materials.new("MarkerMat")
mat_m.use_nodes = True
nodes = mat_m.node_tree.nodes
nodes.clear()
tex = nodes.new("ShaderNodeTexImage")
tex.image = image
tex.interpolation = "Closest"
emission = nodes.new("ShaderNodeEmission")
out_node = nodes.new("ShaderNodeOutputMaterial")
mat_m.node_tree.links.new(tex.outputs["Color"], emission.inputs["Color"])
mat_m.node_tree.links.new(emission.outputs["Emission"], out_node.inputs["Surface"])

hm = {marker_size} / 2.0
m_mesh = bpy.data.meshes.new("MarkerMesh")
m_mesh.from_pydata([(-hm, -hm, 0), (hm, -hm, 0), (hm, hm, 0), (-hm, hm, 0)], [], [(0, 1, 2, 3)])
m_mesh.update()
uv = m_mesh.uv_layers.new(name="UVMap")
for loop_idx, uv_c in zip(m_mesh.polygons[0].loop_indices, [(0, 0), (1, 0), (1, 1), (0, 1)]):
    uv.data[loop_idx].uv = uv_c

marker_obj = bpy.data.objects.new("AprilTag_Marker", m_mesh)
scene.collection.objects.link(marker_obj)
marker_obj.location = {marker_pos}
marker_obj.data.materials.append(mat_m)

# 3D Cans
cans = {cans_repr}
for c in cans:
    pos = c["position"]
    radius = c["radius"]
    height = c["height"]
    name = c["name"]
    color = c["color"]
    rot = c.get("rotation_euler", [0, 0, 0])
    
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=radius, depth=height, location=(pos[0], pos[1], pos[2]))
    obj = bpy.context.active_object
    obj.name = name
    obj.rotation_euler = (rot[0], rot[1], rot[2])
    
    mat = bpy.data.materials.new(name + "_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
    bsdf.inputs["Roughness"].default_value = 0.3
    obj.data.materials.append(mat)

# Dual Lights
for lx, ly, lz in [(2.0, -2.0, 4.0), (-2.0, 2.0, 4.0)]:
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_data.energy = 3.5
    light_obj = bpy.data.objects.new("Sun", light_data)
    scene.collection.objects.link(light_obj)
    light_obj.location = (lx, ly, lz)

# Target point: centroid of all items
all_pts = [c["position"] for c in cans] + [{marker_pos}]
avg_x = sum(p[0] for p in all_pts) / len(all_pts)
avg_y = sum(p[1] for p in all_pts) / len(all_pts)
avg_z = sum(p[2] for p in all_pts) / len(all_pts)
target_loc = mathutils.Vector((avg_x, avg_y, avg_z))

def create_rolled_cam(name, loc, focal_px, roll_deg):
    c_data = bpy.data.cameras.new(name)
    c_data.lens = focal_px * 36.0 / {WIDTH}
    c_data.sensor_width = 36.0
    c_data.sensor_fit = "HORIZONTAL"
    c_data.clip_start = 0.01
    c_data.clip_end = 100.0
    c_obj = bpy.data.objects.new(name, c_data)
    scene.collection.objects.link(c_obj)
    c_obj.location = loc
    
    # Base tracking orientation towards workspace
    dir_vec = target_loc - mathutils.Vector(loc)
    q_base = dir_vec.to_track_quat('-Z', 'Y')
    
    # Rotate around optical Z axis (in-plane camera roll)
    q_roll = mathutils.Quaternion((0, 0, 1), math.radians(roll_deg))
    q_final = q_base @ q_roll
    c_obj.rotation_euler = q_final.to_euler()
    return c_obj

cam0 = create_rolled_cam("Camera_0", {cam0_pos}, {focal0_px}, {cam0_roll_deg})
cam1 = create_rolled_cam("Camera_1", {cam1_pos}, {focal1_px}, {cam1_roll_deg})

bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(art_dir, "scene.blend"))

scene.camera = cam0
scene.render.filepath = os.path.join(inp_dir, "cam0_render.png")
bpy.ops.render.render(write_still=True)

scene.camera = cam1
scene.render.filepath = os.path.join(inp_dir, "cam1_render.png")
bpy.ops.render.render(write_still=True)

with open(os.path.join(inp_dir, "ground_truth_cameras.json"), "w") as f:
    json.dump({{
        "cam0_matrix_world": [list(row) for row in cam0.matrix_world],
        "cam1_matrix_world": [list(row) for row in cam1.matrix_world],
        "cam0_position": list(cam0.location),
        "cam1_position": list(cam1.location),
        "cam0_fov_deg": {cam0_fov},
        "cam1_fov_deg": {cam1_fov},
        "cam0_roll_deg": {cam0_roll_deg},
        "cam1_roll_deg": {cam1_roll_deg},
    }}, f, indent=2)
'''


def run_single_scene(
    exp_dir: Path,
    yolo_model: YOLO,
    marker_size: float,
    marker_id: int,
    marker_pos: tuple[float, float, float],
    cans: list[dict],
    cam0_pos: tuple[float, float, float],
    cam0_fov: float,
    cam0_roll_deg: float,
    cam1_pos: tuple[float, float, float],
    cam1_fov: float,
    cam1_roll_deg: float,
    blender_bin: str = "/snap/bin/blender",
) -> dict:
    shared_dir = exp_dir.parent / "shared"
    inputs_dir = exp_dir / "inputs"
    artifacts_dir = exp_dir / "artifacts"
    outputs_dir = exp_dir / "outputs"

    for d in (shared_dir, inputs_dir, artifacts_dir, outputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    marker_path = ensure_marker_image(shared_dir, marker_id)

    corner_world = np.array([
        marker_pos[0] - marker_size / 2.0,
        marker_pos[1] - marker_size / 2.0,
        marker_pos[2]
    ])
    for c in cans:
        p = np.array(c["position"])
        c["gt_dist_to_marker_corner_m"] = float(np.linalg.norm(p - corner_world))

    K0 = make_K(cam0_fov)
    K1 = make_K(cam1_fov)

    gt_data = {
        "marker": {
            "id": marker_id,
            "size_m": marker_size,
            "center_position": list(marker_pos),
            "corner_origin_position": corner_world.tolist(),
        },
        "cans": cans,
        "camera0_position": list(cam0_pos),
        "camera1_position": list(cam1_pos),
        "cam0_intrinsics": {
            "fx": float(K0[0, 0]), "fy": float(K0[1, 1]),
            "cx": float(K0[0, 2]), "cy": float(K0[1, 2]),
            "hfov_deg": cam0_fov, "roll_deg": cam0_roll_deg
        },
        "cam1_intrinsics": {
            "fx": float(K1[0, 0]), "fy": float(K1[1, 1]),
            "cx": float(K1[0, 2]), "cy": float(K1[1, 2]),
            "hfov_deg": cam1_fov, "roll_deg": cam1_roll_deg
        }
    }
    (inputs_dir / "ground_truth.json").write_text(json.dumps(gt_data, indent=2) + "\n", encoding="utf-8")

    # Render Blender Scene
    script_str = generate_blender_scene_script_roll(
        artifacts_dir=artifacts_dir,
        inputs_dir=inputs_dir,
        marker_path=marker_path,
        marker_size=marker_size,
        marker_pos=marker_pos,
        cans=cans,
        cam0_pos=cam0_pos,
        cam0_fov=cam0_fov,
        cam0_roll_deg=cam0_roll_deg,
        cam1_pos=cam1_pos,
        cam1_fov=cam1_fov,
        cam1_roll_deg=cam1_roll_deg,
    )
    (artifacts_dir / "build_scene.py").write_text(script_str, encoding="utf-8")

    cmd = [blender_bin, "--background", "--factory-startup", "--python", str(artifacts_dir / "build_scene.py")]
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Blender failed: {res.stderr}")

    cam_data = json.loads((inputs_dir / "ground_truth_cameras.json").read_text(encoding="utf-8"))
    M0 = np.array(cam_data["cam0_matrix_world"])
    M1 = np.array(cam_data["cam1_matrix_world"])
    R0, t0 = get_cv_extrinsics(M0)
    R1, t1 = get_cv_extrinsics(M1)

    P0 = K0 @ np.hstack((R0, t0.reshape(3, 1)))
    P1 = K1 @ np.hstack((R1, t1.reshape(3, 1)))

    img0 = cv2.imread(str(inputs_dir / "cam0_render.png"))
    img1 = cv2.imread(str(inputs_dir / "cam1_render.png"))

    # Marker pose detection
    detector = Detector(families="tag36h11", nthreads=2, quad_decimate=1.0)
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    cparams0 = (float(K0[0, 0]), float(K0[1, 1]), float(K0[0, 2]), float(K0[1, 2]))
    cparams1 = (float(K1[0, 0]), float(K1[1, 1]), float(K1[0, 2]), float(K1[1, 2]))
    dets0_m = detector.detect(gray0, estimate_tag_pose=True, camera_params=cparams0, tag_size=marker_size)
    dets1_m = detector.detect(gray1, estimate_tag_pose=True, camera_params=cparams1, tag_size=marker_size)

    det0 = next((d for d in dets0_m if d.tag_id == marker_id), None)
    det1 = next((d for d in dets1_m if d.tag_id == marker_id), None)

    if det0 is not None:
        t_c0_m = np.asarray(det0.pose_t, dtype=float).reshape(3)
        marker_center_world_est = R0.T @ (t_c0_m - t0)
    elif det1 is not None:
        t_c1_m = np.asarray(det1.pose_t, dtype=float).reshape(3)
        marker_center_world_est = R1.T @ (t_c1_m - t1)
    else:
        raise RuntimeError("Marker not detected in either rolled camera view")

    corner_offset_world = np.array([-marker_size / 2.0, -marker_size / 2.0, 0.0])
    marker_corner_world_est = marker_center_world_est + corner_offset_world

    # Run detection methodologies
    yolo_dets0 = detect_cans_yolo(img0, yolo_model, conf=0.005)
    yolo_dets1 = detect_cans_yolo(img1, yolo_model, conf=0.005)
    cntr_dets0 = detect_cans_color_contour(img0)
    cntr_dets1 = detect_cans_color_contour(img1)

    # Multi-K Ray Midpoint Triangulation (handles different FOVs)
    def tri_dlt(p0, p1):
        return triangulate_dlt(p0, p1, P0, P1)

    def tri_mid(p0, p1):
        C0 = -R0.T @ t0.reshape(3)
        C1 = -R1.T @ t1.reshape(3)
        ray0_c = np.linalg.inv(K0) @ np.array([p0[0], p0[1], 1.0])
        ray1_c = np.linalg.inv(K1) @ np.array([p1[0], p1[1], 1.0])
        d0 = R0.T @ (ray0_c / np.linalg.norm(ray0_c))
        d1 = R1.T @ (ray1_c / np.linalg.norm(ray1_c))
        w0 = C0 - C1
        a, b, c = float(np.dot(d0, d0)), float(np.dot(d0, d1)), float(np.dot(d1, d1))
        d, e = float(np.dot(d0, w0)), float(np.dot(d1, w0))
        denom = a * c - b * b
        if abs(denom) < 1e-9:
            return (C0 + C1) / 2.0, 0.0
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
        P_a, P_b = C0 + s * d0, C1 + t * d1
        return (P_a + P_b) / 2.0, float(np.linalg.norm(P_a - P_b))

    # Evaluate across methodologies
    rep_ydlt = evaluate_methodology("yolo_dlt", yolo_dets0, yolo_dets1, tri_dlt, cans, marker_corner_world_est, img0, img1, outputs_dir / "yolo_dlt")
    rep_ymid = evaluate_methodology("yolo_midpoint", yolo_dets0, yolo_dets1, tri_mid, cans, marker_corner_world_est, img0, img1, outputs_dir / "yolo_midpoint")
    rep_cdlt = evaluate_methodology("contour_dlt", cntr_dets0, cntr_dets1, tri_dlt, cans, marker_corner_world_est, img0, img1, outputs_dir / "contour_dlt")
    rep_cmid = evaluate_methodology("contour_midpoint", cntr_dets0, cntr_dets1, tri_mid, cans, marker_corner_world_est, img0, img1, outputs_dir / "contour_midpoint")

    return {
        "yolo_dlt": rep_ydlt,
        "yolo_midpoint": rep_ymid,
        "contour_dlt": rep_cdlt,
        "contour_midpoint": rep_cmid,
        "cam0_fov": cam0_fov,
        "cam1_fov": cam1_fov,
        "cam0_roll_deg": cam0_roll_deg,
        "cam1_roll_deg": cam1_roll_deg,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Mixed FOVs with Optical Z-Axis Roll Rotations Experiment")
    parser.add_argument("--base-dir", type=Path, default=Path("experiment/10_can_mixed_fov_optical_roll"))
    parser.add_argument("--yolo-model", type=Path, default=Path(DEFAULT_YOLO_MODEL))
    parser.add_argument("--num-scenes", type=int, default=5)
    args = parser.parse_args()

    args.base_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = args.base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO model from {args.yolo_model}...")
    yolo_model = YOLO(str(args.yolo_model))

    summary_rows = []
    np.random.seed(2026)

    print(f"=== Starting Mixed FOVs + Optical Roll Experiments ({args.num_scenes} scenes) ===")

    for i in range(1, args.num_scenes + 1):
        scene_name = f"{i:03d}"
        scene_dir = args.base_dir / scene_name

        # Floating cans at 3D heights
        cans = [
            {
                "name": "can_red",
                "color": [0.85, 0.08, 0.08],
                "position": [float(np.random.uniform(-0.25, -0.05)), float(np.random.uniform(0.04, 0.20)), float(np.random.uniform(0.08, 0.28))],
                "rotation_euler": [float(np.radians(np.random.uniform(-20, 20))), float(np.radians(np.random.uniform(-20, 20))), 0.0],
                "radius": CAN_RADIUS_M,
                "height": CAN_HEIGHT_M,
            },
            {
                "name": "can_green",
                "color": [0.08, 0.85, 0.08],
                "position": [float(np.random.uniform(-0.06, 0.08)), float(np.random.uniform(0.10, 0.26)), float(np.random.uniform(0.12, 0.35))],
                "rotation_euler": [float(np.radians(np.random.uniform(-20, 20))), float(np.radians(np.random.uniform(-20, 20))), 0.0],
                "radius": CAN_RADIUS_M,
                "height": CAN_HEIGHT_M,
            },
            {
                "name": "can_blue",
                "color": [0.08, 0.20, 0.88],
                "position": [float(np.random.uniform(0.10, 0.26)), float(np.random.uniform(0.03, 0.18)), float(np.random.uniform(0.05, 0.22))],
                "rotation_euler": [float(np.radians(np.random.uniform(-20, 20))), float(np.radians(np.random.uniform(-20, 20))), 0.0],
                "radius": CAN_RADIUS_M,
                "height": CAN_HEIGHT_M,
            },
        ]

        # Different FOVs
        fov0 = float(np.random.choice([75.0, 80.0, 85.0, 90.0]))  # Wide FOV
        fov1 = float(np.random.choice([45.0, 50.0, 55.0, 60.0]))  # Narrow/Normal FOV

        # In-Plane Roll Rotation around optical Z-axis
        roll0 = float(np.random.uniform(-40.0, 40.0))
        roll1 = float(np.random.uniform(-40.0, 40.0))

        # Camera positions (opposing / wide stereo baselines)
        cam0_pos = (float(np.random.uniform(-0.60, -0.30)), float(np.random.uniform(-1.05, -0.80)), float(np.random.uniform(0.40, 0.65)))
        cam1_pos = (float(np.random.uniform(0.30, 0.60)), float(np.random.uniform(-1.05, -0.80)), float(np.random.uniform(0.40, 0.65)))

        print(f"Rendering & Evaluating Scene {scene_name} (FOV: {fov0:.0f}° vs {fov1:.0f}°, Roll: {roll0:+.1f}° vs {roll1:+.1f}°)...")
        results = run_single_scene(
            exp_dir=scene_dir,
            yolo_model=yolo_model,
            marker_size=DEFAULT_MARKER_SIZE_M,
            marker_id=DEFAULT_MARKER_ID,
            marker_pos=(0.0, -0.05, 0.001),
            cans=cans,
            cam0_pos=cam0_pos,
            cam0_fov=fov0,
            cam0_roll_deg=roll0,
            cam1_pos=cam1_pos,
            cam1_fov=fov1,
            cam1_roll_deg=roll1,
        )

        for m_name in ["yolo_dlt", "yolo_midpoint", "contour_dlt", "contour_midpoint"]:
            rep = results[m_name]
            err = rep.get("mean_corner_distance_error_cm")
            summary_rows.append({
                "scene_id": scene_name,
                "methodology": m_name,
                "cam0_fov": fov0,
                "cam1_fov": fov1,
                "cam0_roll_deg": round(roll0, 1),
                "cam1_roll_deg": round(roll1, 1),
                "mean_error_cm": err if err is not None else 999.0,
                "passed": rep.get("status") == "PASS",
                "num_cans_evaluated": len(rep.get("detections", [])),
            })

    # Save summary
    csv_path = reports_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    (reports_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8")

    print(f"\n[DONE] Finished {args.num_scenes} scenes.")
    print(f"Results saved to: {reports_dir}")


if __name__ == "__main__":
    main()
