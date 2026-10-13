#!/usr/bin/env python3
"""Experiment 11: Complex Textured Backgrounds and Floor Planes (Challenging Contour Segmentation).

Features:
  - Textured floor plane with procedural multi-color Voronoi / Checker patterns
    containing colored elements (red, green, blue tones), designed to challenge naive
    color segmentation by producing realistic background clutter and distractor blobs.
  - Background wall / poster plane behind the scene with high-frequency textures and colored shapes.
  - 3 target cans (Red, Green, Blue) with subtle labels.
  - AprilTag fiducial marker placed on the floor plane.
  - Multi-method comparison:
      - YOLO + DLT (`yolo_dlt`)
      - YOLO + Ray Midpoint (`yolo_midpoint`)
      - Color Contour + DLT (`contour_dlt`)
      - Color Contour + Ray Midpoint (`contour_midpoint`)
  - Evaluates how color segmentation degrades when backgrounds are non-uniform / textured,
    while deep-learning (YOLO) leverages semantic object shape features.

Follows TESTING_PROCEDURE.md sequential ID numbering:
  experiment/11_can_textured_background/
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
H_FOV_DEG = 60.0
FOCAL_PX = (WIDTH / 2.0) / math.tan(math.radians(H_FOV_DEG / 2.0))
K = np.array([[FOCAL_PX, 0.0, WIDTH / 2.0], [0.0, FOCAL_PX, HEIGHT / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def generate_blender_scene_script_textured(
    artifacts_dir: Path,
    inputs_dir: Path,
    marker_path: Path,
    marker_size: float,
    marker_pos: tuple[float, float, float],
    cans: list[dict],
    cam0_pos: tuple[float, float, float],
    cam1_pos: tuple[float, float, float],
) -> str:
    escaped_art = str(artifacts_dir.resolve()).replace("\\", "\\\\").replace("'", "\\'")
    escaped_inp = str(inputs_dir.resolve()).replace("\\", "\\\\").replace("'", "\\'")
    escaped_marker = str(marker_path.resolve()).replace("\\", "\\\\").replace("'", "\\'")
    cans_repr = repr(cans)

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
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.7, 0.7, 0.7, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
scene.world = world

# 1. Textured Floor with Checker & Voronoi Clutter (colors: red, green, blue elements)
bpy.ops.mesh.primitive_plane_add(size=12.0, location=(0, 0, 0))
floor = bpy.context.active_object
floor.name = "Textured_Floor"
mat_f = bpy.data.materials.new("FloorTexturedMat")
mat_f.use_nodes = True
fn = mat_f.node_tree.nodes
fl = mat_f.node_tree.links
fn.clear()

f_out = fn.new('ShaderNodeOutputMaterial')
f_bsdf = fn.new('ShaderNodeBsdfPrincipled')
f_checker = fn.new('ShaderNodeTexChecker')
f_checker.inputs['Scale'].default_value = 12.0
# Checker tiles with subtle warm/cool tones
f_checker.inputs['Color1'].default_value = (0.45, 0.38, 0.35, 1.0)
f_checker.inputs['Color2'].default_value = (0.28, 0.32, 0.36, 1.0)

# Add Voronoi color noise layer for high frequency texture clutter
f_voronoi = fn.new('ShaderNodeTexVoronoi')
f_voronoi.inputs['Scale'].default_value = 25.0
f_mix = fn.new('ShaderNodeMix')
f_mix.data_type = 'RGBA'
f_mix.inputs['Factor'].default_value = 0.35

fl.new(f_checker.outputs['Color'], f_mix.inputs[6])
fl.new(f_voronoi.outputs['Color'], f_mix.inputs[7])
fl.new(f_mix.outputs[2], f_bsdf.inputs['Base Color'])
fl.new(f_bsdf.outputs['BSDF'], f_out.inputs['Surface'])
floor.data.materials.append(mat_f)

# 2. Textured Background Wall / Poster (standing at y = 0.8m)
bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0, 0.85, 0.6))
wall = bpy.context.active_object
wall.rotation_euler = (math.radians(90), 0, 0)
wall.scale = (1.5, 0.6, 1.0)
mat_w = bpy.data.materials.new("WallPosterMat")
mat_w.use_nodes = True
wn = mat_w.node_tree.nodes
wl = mat_w.node_tree.links
wn.clear()
w_out = wn.new('ShaderNodeOutputMaterial')
w_bsdf = wn.new('ShaderNodeBsdfPrincipled')
w_checker = wn.new('ShaderNodeTexChecker')
w_checker.inputs['Scale'].default_value = 8.0
# Colorful distractor background pattern
w_checker.inputs['Color1'].default_value = (0.75, 0.15, 0.15, 1.0) # Red distractors
w_checker.inputs['Color2'].default_value = (0.15, 0.65, 0.25, 1.0) # Green distractors
wl.new(w_checker.outputs['Color'], w_bsdf.inputs['Base Color'])
wl.new(w_bsdf.outputs['BSDF'], w_out.inputs['Surface'])
wall.data.materials.append(mat_w)

# 3. AprilTag Marker (Emission shader)
image = bpy.data.images.load(marker_img, check_existing=True)
image.colorspace_settings.name = "Non-Color"
mat_m = bpy.data.materials.new("MarkerMat")
mat_m.use_nodes = True
mn = mat_m.node_tree.nodes
mn.clear()
tex = mn.new("ShaderNodeTexImage")
tex.image = image
tex.interpolation = "Closest"
emission = mn.new("ShaderNodeEmission")
m_out = mn.new("ShaderNodeOutputMaterial")
mat_m.node_tree.links.new(tex.outputs["Color"], emission.inputs["Color"])
mat_m.node_tree.links.new(emission.outputs["Emission"], m_out.inputs["Surface"])

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

# 4. Target Cans
cans = {cans_repr}
for c in cans:
    pos = c["position"]
    radius = c["radius"]
    height = c["height"]
    name = c["name"]
    color = c["color"]
    
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=radius, depth=height, location=(pos[0], pos[1], pos[2]))
    obj = bpy.context.active_object
    obj.name = name
    
    mat = bpy.data.materials.new(name + "_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
    bsdf.inputs["Roughness"].default_value = 0.35
    obj.data.materials.append(mat)

# Dual directional lights
for lx, ly, lz in [(2.0, -2.0, 4.0), (-2.0, 2.0, 4.0)]:
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

def create_cam(name, loc):
    c_data = bpy.data.cameras.new(name)
    c_data.lens = {FOCAL_PX} * 36.0 / {WIDTH}
    c_data.sensor_width = 36.0
    c_data.sensor_fit = "HORIZONTAL"
    c_data.clip_start = 0.01
    c_data.clip_end = 100.0
    c_obj = bpy.data.objects.new(name, c_data)
    scene.collection.objects.link(c_obj)
    c_obj.location = loc
    dir_vec = target_loc - mathutils.Vector(loc)
    c_obj.rotation_euler = dir_vec.to_track_quat('-Z', 'Y').to_euler()
    return c_obj

cam0 = create_cam("Camera_0", {cam0_pos})
cam1 = create_cam("Camera_1", {cam1_pos})

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
    }}, f, indent=2)
'''


def run_textured_scene(
    exp_dir: Path,
    yolo_model: YOLO,
    marker_size: float = DEFAULT_MARKER_SIZE_M,
    marker_id: int = DEFAULT_MARKER_ID,
    marker_pos: tuple[float, float, float] = (0.0, -0.05, 0.001),
    cans: list[dict] | None = None,
    cam0_pos: tuple[float, float, float] = (-0.35, -0.95, 0.45),
    cam1_pos: tuple[float, float, float] = (0.35, -0.95, 0.45),
    blender_bin: str = "/snap/bin/blender",
) -> dict:
    shared_dir = exp_dir.parent / "shared"
    inputs_dir = exp_dir / "inputs"
    artifacts_dir = exp_dir / "artifacts"
    outputs_dir = exp_dir / "outputs"

    for d in (shared_dir, inputs_dir, artifacts_dir, outputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    marker_path = ensure_marker_image(shared_dir, marker_id)

    if cans is None:
        cans = [
            {"name": "can_red", "color": [0.85, 0.08, 0.08], "position": [-0.18, 0.08, CAN_HEIGHT_M / 2.0], "radius": CAN_RADIUS_M, "height": CAN_HEIGHT_M},
            {"name": "can_green", "color": [0.08, 0.85, 0.08], "position": [0.02, 0.15, CAN_HEIGHT_M / 2.0], "radius": CAN_RADIUS_M, "height": CAN_HEIGHT_M},
            {"name": "can_blue", "color": [0.08, 0.20, 0.88], "position": [0.18, 0.05, CAN_HEIGHT_M / 2.0], "radius": CAN_RADIUS_M, "height": CAN_HEIGHT_M},
        ]

    corner_world = np.array([
        marker_pos[0] - marker_size / 2.0,
        marker_pos[1] - marker_size / 2.0,
        marker_pos[2]
    ])
    for c in cans:
        p = np.array(c["position"])
        c["gt_dist_to_marker_corner_m"] = float(np.linalg.norm(p - corner_world))

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
        "intrinsics": {"fx": FOCAL_PX, "fy": FOCAL_PX, "cx": WIDTH / 2.0, "cy": HEIGHT / 2.0, "hfov_deg": H_FOV_DEG}
    }
    (inputs_dir / "ground_truth.json").write_text(json.dumps(gt_data, indent=2) + "\n", encoding="utf-8")

    # Render Blender Scene
    script_str = generate_blender_scene_script_textured(
        artifacts_dir=artifacts_dir,
        inputs_dir=inputs_dir,
        marker_path=marker_path,
        marker_size=marker_size,
        marker_pos=marker_pos,
        cans=cans,
        cam0_pos=cam0_pos,
        cam1_pos=cam1_pos,
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

    P0 = K @ np.hstack((R0, t0.reshape(3, 1)))
    P1 = K @ np.hstack((R1, t1.reshape(3, 1)))

    img0 = cv2.imread(str(inputs_dir / "cam0_render.png"))
    img1 = cv2.imread(str(inputs_dir / "cam1_render.png"))

    # Marker pose
    detector = Detector(families="tag36h11", nthreads=2, quad_decimate=1.0)
    cparams = (FOCAL_PX, FOCAL_PX, WIDTH / 2.0, HEIGHT / 2.0)
    dets0_m = detector.detect(cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY), estimate_tag_pose=True, camera_params=cparams, tag_size=marker_size)
    dets1_m = detector.detect(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), estimate_tag_pose=True, camera_params=cparams, tag_size=marker_size)

    det0 = next((d for d in dets0_m if d.tag_id == marker_id), None)
    det1 = next((d for d in dets1_m if d.tag_id == marker_id), None)

    if det0 is not None:
        t_c0_m = np.asarray(det0.pose_t, dtype=float).reshape(3)
        marker_center_world_est = R0.T @ (t_c0_m - t0)
    elif det1 is not None:
        t_c1_m = np.asarray(det1.pose_t, dtype=float).reshape(3)
        marker_center_world_est = R1.T @ (t_c1_m - t1)
    else:
        raise RuntimeError("Marker not detected in textured scene views")

    corner_offset_world = np.array([-marker_size / 2.0, -marker_size / 2.0, 0.0])
    marker_corner_world_est = marker_center_world_est + corner_offset_world

    # Run detectors
    yolo_dets0 = detect_cans_yolo(img0, yolo_model, conf=0.005)
    yolo_dets1 = detect_cans_yolo(img1, yolo_model, conf=0.005)
    cntr_dets0 = detect_cans_color_contour(img0)
    cntr_dets1 = detect_cans_color_contour(img1)

    # Triangulation function closures
    def tri_dlt(p0, p1):
        return triangulate_dlt(p0, p1, P0, P1)

    def tri_mid(p0, p1):
        return triangulate_ray_midpoint(p0, p1, R0, t0, R1, t1, K)

    rep_ydlt = evaluate_methodology("yolo_dlt", yolo_dets0, yolo_dets1, tri_dlt, cans, marker_corner_world_est, img0, img1, outputs_dir / "yolo_dlt")
    rep_ymid = evaluate_methodology("yolo_midpoint", yolo_dets0, yolo_dets1, tri_mid, cans, marker_corner_world_est, img0, img1, outputs_dir / "yolo_midpoint")
    rep_cdlt = evaluate_methodology("contour_dlt", cntr_dets0, cntr_dets1, tri_dlt, cans, marker_corner_world_est, img0, img1, outputs_dir / "contour_dlt")
    rep_cmid = evaluate_methodology("contour_midpoint", cntr_dets0, cntr_dets1, tri_mid, cans, marker_corner_world_est, img0, img1, outputs_dir / "contour_midpoint")

    return {
        "yolo_dlt": rep_ydlt,
        "yolo_midpoint": rep_ymid,
        "contour_dlt": rep_cdlt,
        "contour_midpoint": rep_cmid,
        "raw_counts": {
            "yolo_cam0": len(yolo_dets0), "yolo_cam1": len(yolo_dets1),
            "contour_cam0": len(cntr_dets0), "contour_cam1": len(cntr_dets1),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Run Textured Background and Floor Clutter Experiment")
    parser.add_argument("--base-dir", type=Path, default=Path("experiment/11_can_textured_background"))
    parser.add_argument("--yolo-model", type=Path, default=Path(DEFAULT_YOLO_MODEL))
    parser.add_argument("--num-scenes", type=int, default=5)
    args = parser.parse_args()

    args.base_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = args.base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO model from {args.yolo_model}...")
    yolo_model = YOLO(str(args.yolo_model))

    summary_rows = []
    np.random.seed(303)

    print(f"=== Starting Experiment 11: Textured Background & Clutter ({args.num_scenes} scenes) ===")

    for i in range(1, args.num_scenes + 1):
        scene_name = f"{i:03d}"
        scene_dir = args.base_dir / scene_name

        cans = [
            {
                "name": "can_red",
                "color": [0.85, 0.08, 0.08],
                "position": [float(np.random.uniform(-0.24, -0.06)), float(np.random.uniform(0.05, 0.18)), CAN_HEIGHT_M / 2.0],
                "radius": CAN_RADIUS_M,
                "height": CAN_HEIGHT_M,
            },
            {
                "name": "can_green",
                "color": [0.08, 0.85, 0.08],
                "position": [float(np.random.uniform(-0.04, 0.09)), float(np.random.uniform(0.12, 0.25)), CAN_HEIGHT_M / 2.0],
                "radius": CAN_RADIUS_M,
                "height": CAN_HEIGHT_M,
            },
            {
                "name": "can_blue",
                "color": [0.08, 0.20, 0.88],
                "position": [float(np.random.uniform(0.11, 0.26)), float(np.random.uniform(0.04, 0.16)), CAN_HEIGHT_M / 2.0],
                "radius": CAN_RADIUS_M,
                "height": CAN_HEIGHT_M,
            },
        ]

        cam0_pos = (float(np.random.uniform(-0.45, -0.25)), float(np.random.uniform(-1.05, -0.85)), float(np.random.uniform(0.35, 0.55)))
        cam1_pos = (float(np.random.uniform(0.25, 0.45)), float(np.random.uniform(-1.05, -0.85)), float(np.random.uniform(0.35, 0.55)))

        print(f"Rendering & Evaluating Scene {scene_name} (with textured floor + colorful wall poster)...")
        results = run_textured_scene(
            exp_dir=scene_dir,
            yolo_model=yolo_model,
            cans=cans,
            cam0_pos=cam0_pos,
            cam1_pos=cam1_pos,
        )

        for m_name in ["yolo_dlt", "yolo_midpoint", "contour_dlt", "contour_midpoint"]:
            rep = results[m_name]
            err = rep.get("mean_corner_distance_error_cm")
            summary_rows.append({
                "scene_id": scene_name,
                "methodology": m_name,
                "mean_error_cm": err if err is not None else 999.0,
                "passed": rep.get("status") == "PASS",
                "num_cans_evaluated": len(rep.get("detections", [])),
                "cam0_candidates": results["raw_counts"]["contour_cam0"] if "contour" in m_name else results["raw_counts"]["yolo_cam0"],
                "cam1_candidates": results["raw_counts"]["contour_cam1"] if "contour" in m_name else results["raw_counts"]["yolo_cam1"],
            })

    # Save summary reports
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
