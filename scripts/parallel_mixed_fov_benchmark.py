#!/usr/bin/env python3
"""Parallel benchmark of multi-camera scenes mixing DIFFERENT Camera FOVs!

Features:
  - Heterogeneous stereo cameras: Cam0 and Cam1 have DIFFERENT FOVs:
      e.g. Cam0 wide (75° - 90° FOV) & Cam1 telephoto/narrow (45° - 55° FOV)
  - Cans floating in mid-air at varied heights and 3D tilts.
  - Cameras placed at varied opposing/wide angles.
  - AprilTag reference marker on the floor.
  - Fully parallelized batch execution across scenes.
  - Evaluates both YOLO + DLT and Contour + DLT triangulation.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from pupil_apriltags import Detector
from ultralytics import YOLO

from multi_method_benchmark_30 import (
    WIDTH, HEIGHT, CAN_RADIUS_M, CAN_HEIGHT_M,
    ensure_marker_image, get_cv_extrinsics,
    triangulate_dlt, triangulate_ray_midpoint,
    detect_cans_yolo, detect_cans_color_contour, evaluate_methodology
)


def make_K(hfov_deg: float, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def generate_blender_scene_script_mixed_fov(
    artifacts_dir: Path,
    inputs_dir: Path,
    marker_path: Path,
    marker_size: float,
    marker_pos: tuple[float, float, float],
    cans: list[dict],
    cam0_pos: tuple[float, float, float],
    cam0_fov: float,
    cam1_pos: tuple[float, float, float],
    cam1_fov: float,
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

# Floor plane
bpy.ops.mesh.primitive_plane_add(size=12.0, location=(0, 0, 0))
floor = bpy.context.active_object
mat_f = bpy.data.materials.new("FloorMat")
mat_f.use_nodes = True
mat_f.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.35, 0.35, 0.35, 1.0)
floor.data.materials.append(mat_f)

# AprilTag Marker Material
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

# Marker Mesh on floor
hm = {marker_size} / 2.0
m_mesh = bpy.data.meshes.new("MarkerMesh")
m_mesh.from_pydata([(-hm, -hm, 0), (hm, -hm, 0), (hm, hm, 0), (-hm, hm, 0)], [], [(0, 1, 2, 3)])
m_mesh.update()
uv = m_mesh.uv_layers.new(name="UVMap")
for loop_idx, uv_c in zip(m_mesh.polygons[0].loop_indices, [(0, 0), (1, 0), (1, 1), (0, 1)]):
    uv.data[loop_idx].uv = uv_c

marker_obj = bpy.data.objects.new("AprilTag_Marker", m_mesh)
scene.collection.objects.link(marker_obj)
marker_obj.location = ({marker_pos[0]}, {marker_pos[1]}, {marker_pos[2]})
marker_obj.data.materials.append(mat_m)

# Floating Cans in air
cans = {cans_repr}
for c in cans:
    pos = c["position"]
    rot = c.get("rotation", [0, 0, 0])
    radius = c["radius"]
    height = c["height"]
    name = c["name"]
    color = c["color"]
    
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

# Multiple Lights
for lx, ly, lz in [(2, -2, 4), (-2, 2, 4)]:
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_data.energy = 3.5
    light_obj = bpy.data.objects.new("Sun", light_data)
    scene.collection.objects.link(light_obj)
    light_obj.location = (lx, ly, lz)

all_pts = [c["position"] for c in cans] + [{marker_pos}]
avg_x = sum(p[0] for p in all_pts) / len(all_pts)
avg_y = sum(p[1] for p in all_pts) / len(all_pts)
avg_z = sum(p[2] for p in all_pts) / len(all_pts)
target_loc = mathutils.Vector((avg_x, avg_y, avg_z))

# Camera 0 with FOV 0 ({cam0_fov} deg)
c0_data = bpy.data.cameras.new("Cam0")
c0_data.lens = {focal0_px} * 36.0 / {WIDTH}
c0_data.sensor_width = 36.0
c0_data.sensor_fit = "HORIZONTAL"
cam0 = bpy.data.objects.new("Cam0", c0_data)
scene.collection.objects.link(cam0)
cam0.location = {cam0_pos}
dir0 = target_loc - mathutils.Vector({cam0_pos})
cam0.rotation_euler = dir0.to_track_quat('-Z', 'Y').to_euler()

# Camera 1 with FOV 1 ({cam1_fov} deg)
c1_data = bpy.data.cameras.new("Cam1")
c1_data.lens = {focal1_px} * 36.0 / {WIDTH}
c1_data.sensor_width = 36.0
c1_data.sensor_fit = "HORIZONTAL"
cam1 = bpy.data.objects.new("Cam1", c1_data)
scene.collection.objects.link(cam1)
cam1.location = {cam1_pos}
dir1 = target_loc - mathutils.Vector({cam1_pos})
cam1.rotation_euler = dir1.to_track_quat('-Z', 'Y').to_euler()

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
    }}, f, indent=2)
'''


def run_single_mixed_fov_experiment(task: dict) -> dict:
    exp_dir = Path(task["exp_dir"])
    marker_size = task["marker_size"]
    marker_id = task["marker_id"]
    marker_pos = task["marker_pos"]
    cans = task["cans"]
    cam0_pos = task["cam0_pos"]
    cam0_fov = task["cam0_fov"]
    cam1_pos = task["cam1_pos"]
    cam1_fov = task["cam1_fov"]
    blender_bin = task.get("blender_bin", "/snap/bin/blender")
    yolo_model_path = task.get("yolo_model_path", "/home/mfk01/Projects/econ-systems/demos/demo2/yolo26s.pt")

    shared_dir = exp_dir.parent / "shared"
    inputs_dir = exp_dir / "inputs"
    artifacts_dir = exp_dir / "artifacts"
    outputs_dir = exp_dir / "outputs"

    for d in (shared_dir, inputs_dir, artifacts_dir, outputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    marker_path = ensure_marker_image(shared_dir, marker_id)

    # Intrinsic matrices for Cam0 and Cam1
    K0 = make_K(cam0_fov)
    K1 = make_K(cam1_fov)

    corner_world = np.array([
        marker_pos[0] - marker_size / 2.0,
        marker_pos[1] - marker_size / 2.0,
        marker_pos[2]
    ])
    for c in cans:
        p = np.array(c["position"])
        c["gt_dist_to_marker_corner_m"] = float(np.linalg.norm(p - corner_world))

    gt_data = {
        "marker": {"id": marker_id, "size_m": marker_size, "center": list(marker_pos), "corner": corner_world.tolist()},
        "cans": cans,
        "cam0": {"pos": list(cam0_pos), "fov_deg": cam0_fov, "fx": float(K0[0, 0])},
        "cam1": {"pos": list(cam1_pos), "fov_deg": cam1_fov, "fx": float(K1[0, 0])},
    }
    (inputs_dir / "ground_truth.json").write_text(json.dumps(gt_data, indent=2) + "\n", encoding="utf-8")

    # Render scene
    script_str = generate_blender_scene_script_mixed_fov(
        artifacts_dir, inputs_dir, marker_path, marker_size, marker_pos, cans, cam0_pos, cam0_fov, cam1_pos, cam1_fov
    )
    (artifacts_dir / "build_scene.py").write_text(script_str, encoding="utf-8")
    cmd = [blender_bin, "--background", "--factory-startup", "--python", str(artifacts_dir / "build_scene.py")]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    cam_data = json.loads((inputs_dir / "ground_truth_cameras.json").read_text(encoding="utf-8"))
    M0 = np.array(cam_data["cam0_matrix_world"])
    M1 = np.array(cam_data["cam1_matrix_world"])
    R0, t0 = get_cv_extrinsics(M0)
    R1, t1 = get_cv_extrinsics(M1)

    P0 = K0 @ np.hstack((R0, t0.reshape(3, 1)))
    P1 = K1 @ np.hstack((R1, t1.reshape(3, 1)))

    img0 = cv2.imread(str(inputs_dir / "cam0_render.png"))
    img1 = cv2.imread(str(inputs_dir / "cam1_render.png"))

    # Marker pose detection using Cam0 intrinsics
    detector = Detector(families="tag36h11", nthreads=1, quad_decimate=1.0)
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    intrinsics0 = (float(K0[0, 0]), float(K0[1, 1]), float(K0[0, 2]), float(K0[1, 2]))
    tag_dets = detector.detect(gray0, estimate_tag_pose=True, camera_params=intrinsics0, tag_size=marker_size)
    tag_det = next((d for d in tag_dets if d.tag_id == marker_id), None)
    if tag_det is not None:
        t_c0_m = np.asarray(tag_det.pose_t, dtype=float).reshape(3)
        marker_center_world_est = R0.T @ (t_c0_m - t0)
    else:
        marker_center_world_est = np.array(marker_pos)
    corner_world_est = marker_center_world_est + np.array([-marker_size / 2.0, -marker_size / 2.0, 0.0])

    yolo_model = YOLO(yolo_model_path)
    yolo_dets0 = detect_cans_yolo(img0, yolo_model)
    yolo_dets1 = detect_cans_yolo(img1, yolo_model)
    contour_dets0 = detect_cans_color_contour(img0)
    contour_dets1 = detect_cans_color_contour(img1)

    res_yolo_dlt = evaluate_methodology(
        "yolo_dlt", yolo_dets0, yolo_dets1,
        lambda p0, p1: triangulate_dlt(p0, p1, P0, P1),
        cans, corner_world_est, img0, img1, outputs_dir / "yolo_dlt"
    )

    res_cnt_dlt = evaluate_methodology(
        "contour_dlt", contour_dets0, contour_dets1,
        lambda p0, p1: triangulate_dlt(p0, p1, P0, P1),
        cans, corner_world_est, img0, img1, outputs_dir / "contour_dlt"
    )

    return {
        "exp_id": task["exp_id"],
        "cam0_fov": cam0_fov,
        "cam1_fov": cam1_fov,
        "yolo_err_cm": res_yolo_dlt["mean_corner_distance_error_cm"],
        "contour_err_cm": res_cnt_dlt["mean_corner_distance_error_cm"],
        "yolo_status": res_yolo_dlt["status"],
        "contour_status": res_cnt_dlt["status"]
    }


def main():
    rng = np.random.default_rng(555)
    family_dir = Path("experiment/can_mixed_fovs")
    reports_dir = family_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    print("\nGenerating 10 Mixed-FOV Parallel Experiment Tasks...")

    # Varied FOV pairs: wide-angle vs telephoto, medium vs ultrawide
    fov_pairs = [
        (45.0, 75.0),
        (50.0, 85.0),
        (40.0, 70.0),
        (55.0, 80.0),
        (48.0, 90.0),
        (75.0, 45.0),
        (85.0, 50.0),
        (70.0, 42.0),
        (60.0, 85.0),
        (50.0, 75.0),
    ]

    for i in range(1, 11):
        exp_id = f"{i:03d}"
        exp_dir = family_dir / exp_id
        cam0_fov, cam1_fov = fov_pairs[i - 1]

        marker_size = round(float(rng.uniform(0.075, 0.095)), 3)
        marker_id = int(rng.choice([0, 1, 2, 3, 4]))
        marker_pos = (round(float(rng.uniform(-0.03, 0.03)), 3), round(float(rng.uniform(-0.06, -0.01)), 3), 0.001)

        cans = [
            {"name": "can_red", "color": [0.85, 0.08, 0.08],
             "position": [round(float(rng.uniform(-0.20, -0.10)), 3), round(float(rng.uniform(0.02, 0.08)), 3), round(float(rng.uniform(0.18, 0.28)), 3)],
             "rotation": [round(float(rng.uniform(-0.2, 0.2)), 3), round(float(rng.uniform(-0.2, 0.2)), 3), round(float(rng.uniform(-0.3, 0.3)), 3)],
             "radius": CAN_RADIUS_M, "height": CAN_HEIGHT_M},
            {"name": "can_green", "color": [0.08, 0.85, 0.08],
             "position": [round(float(rng.uniform(-0.04, 0.04)), 3), round(float(rng.uniform(0.06, 0.14)), 3), round(float(rng.uniform(0.28, 0.40)), 3)],
             "rotation": [round(float(rng.uniform(-0.2, 0.2)), 3), round(float(rng.uniform(-0.2, 0.2)), 3), round(float(rng.uniform(-0.3, 0.3)), 3)],
             "radius": CAN_RADIUS_M, "height": CAN_HEIGHT_M},
            {"name": "can_blue", "color": [0.08, 0.20, 0.88],
             "position": [round(float(rng.uniform(0.10, 0.20)), 3), round(float(rng.uniform(0.02, 0.08)), 3), round(float(rng.uniform(0.16, 0.26)), 3)],
             "rotation": [round(float(rng.uniform(-0.2, 0.2)), 3), round(float(rng.uniform(-0.2, 0.2)), 3), round(float(rng.uniform(-0.3, 0.3)), 3)],
             "radius": CAN_RADIUS_M, "height": CAN_HEIGHT_M},
        ]

        cam0_pos = (-0.35, -1.05, 0.42)
        cam1_pos = (0.35, -1.05, 0.42)

        tasks.append({
            "exp_id": exp_id,
            "exp_dir": str(exp_dir),
            "marker_size": marker_size,
            "marker_id": marker_id,
            "marker_pos": marker_pos,
            "cans": cans,
            "cam0_pos": cam0_pos,
            "cam0_fov": cam0_fov,
            "cam1_pos": cam1_pos,
            "cam1_fov": cam1_fov,
            "blender_bin": "/snap/bin/blender",
            "yolo_model_path": "/home/mfk01/Projects/econ-systems/demos/demo2/yolo26s.pt"
        })

    results = []
    print(f"Executing 10 scenes in parallel across worker processes...")
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run_single_mixed_fov_experiment, t): t["exp_id"] for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            print(f"[{res['exp_id']}/010] FOVs: ({res['cam0_fov']}°, {res['cam1_fov']}°) | YOLO: {res['yolo_err_cm']} cm | Contour: {res['contour_err_cm']} cm")

    results.sort(key=lambda r: r["exp_id"])
    (reports_dir / "mixed_fov_summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with open(reports_dir / "mixed_fov_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 80)
    print("PARALLEL MIXED-FOV EXPERIMENTS SUMMARY")
    print("=" * 80)
    print(f"{'Exp ID':<8} {'Cam0 FOV':<12} {'Cam1 FOV':<12} {'YOLO DLT (cm)':<16} {'Contour DLT (cm)':<18}")
    print("-" * 80)
    for r in results:
        print(f"{r['exp_id']:<8} {r['cam0_fov']:<12.1f} {r['cam1_fov']:<12.1f} {r['yolo_err_cm']:<16.2f} {r['contour_err_cm']:<18.2f}")
    print("-" * 80)
    y_mean = np.mean([r["yolo_err_cm"] for r in results if r["yolo_err_cm"] is not None])
    c_mean = np.mean([r["contour_err_cm"] for r in results if r["contour_err_cm"] is not None])
    print(f"Mean Error Across Mixed FOVs: YOLO = {y_mean:.2f} cm ({y_mean*10:.1f} mm), Contour = {c_mean:.2f} cm ({c_mean*10:.1f} mm)")
    print("=" * 80)


if __name__ == "__main__":
    main()
