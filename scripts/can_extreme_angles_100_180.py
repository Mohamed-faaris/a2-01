#!/usr/bin/env python3
"""Experiments with cameras placed at extreme opposing angles: 100° to 180°!

Geometries:
  - Cameras on opposite sides of the scene looking towards each other / central workspace.
  - Convergence baseline angle between 100° and 180°:
      e.g. Cam0 placed at Front (-Y), Cam1 placed at Back (+Y) or Lateral-Opposite (+X, -X).
  - Both cameras can view the floating cans in mid-air.
  - AprilTag marker placed on floor (double-sided or visible from both steep elevated angles).
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from pupil_apriltags import Detector
from ultralytics import YOLO

from multi_method_benchmark_30 import (
    WIDTH, HEIGHT, FOCAL_PX, K, CAN_RADIUS_M, CAN_HEIGHT_M,
    ensure_marker_image, get_cv_extrinsics,
    triangulate_dlt, triangulate_ray_midpoint,
    detect_cans_yolo, detect_cans_color_contour, evaluate_methodology
)


def generate_blender_scene_script_extreme(
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

# Multiple Lights for opposing camera coverage
for lx, ly, lz in [(2, -2, 4), (-2, 2, 4)]:
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_data.energy = 3.0
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


def run_extreme_experiment(
    exp_dir: Path,
    yolo_model: YOLO,
    marker_size: float,
    marker_id: int,
    marker_pos: tuple[float, float, float],
    cans: list[dict],
    cam0_pos: tuple[float, float, float],
    cam1_pos: tuple[float, float, float],
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
        "intrinsics": {"fx": FOCAL_PX, "fy": FOCAL_PX, "cx": WIDTH/2.0, "cy": HEIGHT/2.0, "hfov_deg": 60.0}
    }
    (inputs_dir / "ground_truth.json").write_text(json.dumps(gt_data, indent=2) + "\n", encoding="utf-8")

    # Render Blender scene
    script_str = generate_blender_scene_script_extreme(
        artifacts_dir, inputs_dir, marker_path, marker_size, marker_pos, cans, cam0_pos, cam1_pos
    )
    (artifacts_dir / "build_scene.py").write_text(script_str, encoding="utf-8")

    cmd = [blender_bin, "--background", "--factory-startup", "--python", str(artifacts_dir / "build_scene.py")]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    cam_data = json.loads((inputs_dir / "ground_truth_cameras.json").read_text(encoding="utf-8"))
    M0 = np.array(cam_data["cam0_matrix_world"])
    M1 = np.array(cam_data["cam1_matrix_world"])
    R0, t0 = get_cv_extrinsics(M0)
    R1, t1 = get_cv_extrinsics(M1)
    P0 = K @ np.hstack((R0, t0.reshape(3, 1)))
    P1 = K @ np.hstack((R1, t1.reshape(3, 1)))

    img0 = cv2.imread(str(inputs_dir / "cam0_render.png"))
    img1 = cv2.imread(str(inputs_dir / "cam1_render.png"))

    # Marker pose detection from Cam0 (looking down from front/top)
    detector = Detector(families="tag36h11", nthreads=2, quad_decimate=1.0)
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    intrinsics = (FOCAL_PX, FOCAL_PX, WIDTH / 2.0, HEIGHT / 2.0)
    tag_dets = detector.detect(gray0, estimate_tag_pose=True, camera_params=intrinsics, tag_size=marker_size)
    tag_det = next((d for d in tag_dets if d.tag_id == marker_id), None)
    
    if tag_det is not None:
        t_c0_m = np.asarray(tag_det.pose_t, dtype=float).reshape(3)
        marker_center_world_est = R0.T @ (t_c0_m - t0)
    else:
        marker_center_world_est = np.array(marker_pos)
    
    corner_world_est = marker_center_world_est + np.array([-marker_size / 2.0, -marker_size / 2.0, 0.0])

    yolo_dets0 = detect_cans_yolo(img0, yolo_model)
    yolo_dets1 = detect_cans_yolo(img1, yolo_model)

    contour_dets0 = detect_cans_color_contour(img0)
    contour_dets1 = detect_cans_color_contour(img1)

    res_yolo_dlt = evaluate_methodology(
        "yolo_dlt", yolo_dets0, yolo_dets1,
        lambda p0, p1: triangulate_dlt(p0, p1, P0, P1),
        cans, corner_world_est, img0, img1, outputs_dir / "yolo_dlt"
    )

    res_yolo_mid = evaluate_methodology(
        "yolo_midpoint", yolo_dets0, yolo_dets1,
        lambda p0, p1: triangulate_ray_midpoint(p0, p1, R0, t0, R1, t1, K),
        cans, corner_world_est, img0, img1, outputs_dir / "yolo_midpoint"
    )

    res_cnt_dlt = evaluate_methodology(
        "contour_dlt", contour_dets0, contour_dets1,
        lambda p0, p1: triangulate_dlt(p0, p1, P0, P1),
        cans, corner_world_est, img0, img1, outputs_dir / "contour_dlt"
    )

    res_cnt_mid = evaluate_methodology(
        "contour_midpoint", contour_dets0, contour_dets1,
        lambda p0, p1: triangulate_ray_midpoint(p0, p1, R0, t0, R1, t1, K),
        cans, corner_world_est, img0, img1, outputs_dir / "contour_midpoint"
    )

    return {
        "yolo_dlt": res_yolo_dlt,
        "yolo_midpoint": res_yolo_mid,
        "contour_dlt": res_cnt_dlt,
        "contour_midpoint": res_cnt_mid,
    }


def main():
    rng = np.random.default_rng(10180)
    family_dir = Path("experiment/can_extreme_angles")
    reports_dir = family_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    yolo_model = YOLO("/home/mfk01/Projects/econ-systems/demos/demo2/yolo26s.pt")

    summary_rows = []
    print("\nStarting 10 Extreme Opposing Angles Experiments (100° to 175°)...")

    for i in range(1, 11):
        exp_id = f"{i:03d}"
        exp_dir = family_dir / exp_id

        marker_size = round(float(rng.uniform(0.08, 0.10)), 3)
        marker_id = int(rng.choice([0, 1, 2, 3, 4]))
        marker_pos = (round(float(rng.uniform(-0.02, 0.02)), 3), round(float(rng.uniform(-0.04, 0.04)), 3), 0.001)

        # Cans floating near center
        cans = [
            {
                "name": "can_red", "color": [0.85, 0.08, 0.08],
                "position": [round(float(rng.uniform(-0.18, -0.08)), 3),
                             round(float(rng.uniform(-0.05, 0.05)), 3),
                             round(float(rng.uniform(0.18, 0.28)), 3)],
                "rotation": [round(float(rng.uniform(-0.2, 0.2)), 3),
                             round(float(rng.uniform(-0.2, 0.2)), 3),
                             round(float(rng.uniform(-0.3, 0.3)), 3)],
                "radius": CAN_RADIUS_M, "height": CAN_HEIGHT_M
            },
            {
                "name": "can_green", "color": [0.08, 0.85, 0.08],
                "position": [round(float(rng.uniform(-0.04, 0.04)), 3),
                             round(float(rng.uniform(-0.04, 0.06)), 3),
                             round(float(rng.uniform(0.26, 0.38)), 3)],
                "rotation": [round(float(rng.uniform(-0.2, 0.2)), 3),
                             round(float(rng.uniform(-0.2, 0.2)), 3),
                             round(float(rng.uniform(-0.3, 0.3)), 3)],
                "radius": CAN_RADIUS_M, "height": CAN_HEIGHT_M
            },
            {
                "name": "can_blue", "color": [0.08, 0.20, 0.88],
                "position": [round(float(rng.uniform(0.08, 0.18)), 3),
                             round(float(rng.uniform(-0.05, 0.05)), 3),
                             round(float(rng.uniform(0.16, 0.26)), 3)],
                "rotation": [round(float(rng.uniform(-0.2, 0.2)), 3),
                             round(float(rng.uniform(-0.2, 0.2)), 3),
                             round(float(rng.uniform(-0.3, 0.3)), 3)],
                "radius": CAN_RADIUS_M, "height": CAN_HEIGHT_M
            },
        ]

        # EXTREME OPPOSING ANGLES: 100° to 175°
        # Target angle between 100° and 175°
        target_angle_deg = float(rng.uniform(105.0, 165.0))
        angle_rad = math.radians(target_angle_deg)

        # Cam0 at front-left (azimuth phi0, elevated look down)
        phi0 = float(rng.uniform(-2.6, -2.0))  # in front/left
        r0 = float(rng.uniform(0.95, 1.15))
        z0 = float(rng.uniform(0.40, 0.60))
        cam0_pos = (round(r0 * math.cos(phi0), 3), round(r0 * math.sin(phi0), 3), round(z0, 3))

        # Cam1 placed at phi0 + target_angle (opposing side!)
        phi1 = phi0 + angle_rad
        r1 = float(rng.uniform(0.95, 1.15))
        z1 = float(rng.uniform(0.35, 0.55))
        cam1_pos = (round(r1 * math.cos(phi1), 3), round(r1 * math.sin(phi1), 3), round(z1, 3))

        # True 3D convergence angle
        v0 = -np.array(cam0_pos)
        v1 = -np.array(cam1_pos)
        actual_angle_deg = math.degrees(math.acos(float(np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1)))))

        print(f"[{exp_id}/010] Convergence Angle: {actual_angle_deg:.1f}°")
        print(f"       Cam0: {cam0_pos}")
        print(f"       Cam1: {cam1_pos}")

        evals = run_extreme_experiment(
            exp_dir, yolo_model, marker_size, marker_id, marker_pos, cans, cam0_pos, cam1_pos
        )

        row = {
            "exp_id": exp_id,
            "convergence_angle_deg": round(actual_angle_deg, 1),
            "cam0_pos": cam0_pos,
            "cam1_pos": cam1_pos,
            "yolo_dlt_err_cm": evals["yolo_dlt"]["mean_corner_distance_error_cm"],
            "yolo_midpoint_err_cm": evals["yolo_midpoint"]["mean_corner_distance_error_cm"],
            "contour_dlt_err_cm": evals["contour_dlt"]["mean_corner_distance_error_cm"],
            "contour_midpoint_err_cm": evals["contour_midpoint"]["mean_corner_distance_error_cm"],
        }
        summary_rows.append(row)

    (reports_dir / "extreme_angle_summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8")
    with open(reports_dir / "extreme_angle_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 80)
    print("EXTREME ANGLES (100° - 170°) EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"{'Exp ID':<8} {'Angle':<10} {'YOLO DLT (cm)':<16} {'Contour DLT (cm)':<18}")
    print("-" * 80)
    for r in summary_rows:
        print(f"{r['exp_id']:<8} {r['convergence_angle_deg']:<10.1f}° {r['yolo_dlt_err_cm']:<16.2f} {r['contour_dlt_err_cm']:<18.2f}")
    print("-" * 80)
    y_mean = np.mean([r["yolo_dlt_err_cm"] for r in summary_rows if r["yolo_dlt_err_cm"] is not None])
    c_mean = np.mean([r["contour_dlt_err_cm"] for r in summary_rows if r["contour_dlt_err_cm"] is not None])
    print(f"Overall Mean 3D Error: YOLO = {y_mean:.2f} cm ({y_mean*10:.1f} mm), Contour = {c_mean:.2f} cm ({c_mean*10:.1f} mm)")
    print("=" * 80)


if __name__ == "__main__":
    main()
