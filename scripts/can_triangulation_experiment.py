#!/usr/bin/env python3
"""Can (cylinder) stereo triangulation experiment using YOLO26s.

Features:
  - Simple solid colored cylinders (Red, Green, Blue) without texture maps.
  - Generates ground truth 3D poses and camera extrinsics.
  - Follows directory hierarchy:
      experiment/<experiment_family>/
        shared/
        reports/
        <id>/
          inputs/
          artifacts/
          outputs/
            yolo_triangulation/
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
from ultralytics import YOLO


WIDTH, HEIGHT = 1080, 720
H_FOV_DEG = 60.0
FOCAL_PX = (WIDTH / 2.0) / math.tan(math.radians(H_FOV_DEG / 2.0))
K = np.array([
    [FOCAL_PX, 0.0, WIDTH / 2.0],
    [0.0, FOCAL_PX, HEIGHT / 2.0],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# Standard cylinder dimensions
CAN_RADIUS_M = 0.035
CAN_HEIGHT_M = 0.125


def generate_blender_scene_script(
    artifacts_dir: Path,
    inputs_dir: Path,
    cans: list[dict],
    cam0_pos: tuple[float, float, float],
    cam1_pos: tuple[float, float, float],
) -> str:
    escaped_art = str(artifacts_dir.resolve()).replace("\\", "\\\\").replace("'", "\\'")
    escaped_inp = str(inputs_dir.resolve()).replace("\\", "\\\\").replace("'", "\\'")

    cans_repr = repr(cans)

    return f'''import bpy
import math
import os
import json
import mathutils

art_dir = r'{escaped_art}'
inp_dir = r'{escaped_inp}'
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

# Ambient world lighting
world = bpy.data.worlds.new("World")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.75, 0.75, 0.75, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
scene.world = world

# Neutral floor plane
bpy.ops.mesh.primitive_plane_add(size=12.0, location=(0, 0, 0))
floor = bpy.context.active_object
mat_f = bpy.data.materials.new("FloorMat")
mat_f.use_nodes = True
mat_f.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.35, 0.35, 0.35, 1.0)
floor.data.materials.append(mat_f)

cans = {cans_repr}
for c in cans:
    pos = c["position"]
    radius = c["radius"]
    height = c["height"]
    name = c["name"]
    color = c["color"] # (r, g, b)
    
    # Simple cylinder
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=radius, depth=height, location=(pos[0], pos[1], pos[2]))
    obj = bpy.context.active_object
    obj.name = name
    
    mat = bpy.data.materials.new(name + "_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
    bsdf.inputs["Roughness"].default_value = 0.3
    obj.data.materials.append(mat)

# Overhead Key Light
light_data = bpy.data.lights.new(name="Sun", type='SUN')
light_data.energy = 4.0
light_obj = bpy.data.objects.new("Sun", light_data)
scene.collection.objects.link(light_obj)
light_obj.location = (2.0, -2.0, 4.0)
light_obj.rotation_euler = (math.radians(35), math.radians(20), math.radians(45))

# Target point (centroid of cans)
avg_x = sum(c["position"][0] for c in cans) / len(cans)
avg_y = sum(c["position"][1] for c in cans) / len(cans)
avg_z = sum(c["position"][2] for c in cans) / len(cans)
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


def get_cv_extrinsics(M_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R_b = M_world[:3, :3]
    t_b = M_world[:3, 3]
    R_cv2b = np.diag([1.0, -1.0, -1.0])
    R_w2c = R_cv2b @ R_b.T
    t_w2c = -R_w2c @ t_b
    return R_w2c, t_w2c


def triangulate_2d_points(
    pt0_px: tuple[float, float],
    pt1_px: tuple[float, float],
    P0: np.ndarray,
    P1: np.ndarray,
) -> tuple[np.ndarray, float]:
    pts0 = np.array(pt0_px, dtype=np.float64).reshape(2, 1)
    pts1 = np.array(pt1_px, dtype=np.float64).reshape(2, 1)
    homo = cv2.triangulatePoints(P0, P1, pts0, pts1).reshape(4)
    if abs(homo[3]) < 1e-9:
        return np.zeros(3), float("inf")
    pt3d = homo[:3] / homo[3]

    pt3d_h = np.append(pt3d, 1.0)
    p0_proj = P0 @ pt3d_h
    p1_proj = P1 @ pt3d_h
    proj0_px = p0_proj[:2] / p0_proj[2]
    proj1_px = p1_proj[:2] / p1_proj[2]

    err0 = float(np.linalg.norm(proj0_px - np.array(pt0_px)))
    err1 = float(np.linalg.norm(proj1_px - np.array(pt1_px)))
    return pt3d, (err0 + err1) / 2.0


def detect_cans_yolo(
    image_bgr: np.ndarray,
    model: YOLO,
    conf: float = 0.005,
) -> list[dict]:
    """Detect cylinders/cans with YOLO and suppress duplicates."""
    res = model.predict(source=image_bgr, conf=conf, verbose=False)[0]
    raw_detections = []
    # Accepted beverage container categories in COCO
    target_classes = {"bottle", "cup", "vase"}
    for box in res.boxes:
        cls_id = int(box.cls[0].item())
        name = res.names[cls_id]
        if name in target_classes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].cpu().numpy()]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            
            # Determine dominant color (red, green, blue)
            patch = image_bgr[int(y1):int(y2), int(x1):int(x2)]
            color_name = "unknown"
            if patch.size > 0:
                mean_bgr = patch.mean(axis=(0, 1))
                b, g, r = mean_bgr
                if r > g + 25 and r > b + 25:
                    color_name = "red"
                elif g > r + 25 and g > b + 25:
                    color_name = "green"
                elif b > r + 25 and b > g + 25:
                    color_name = "blue"

            raw_detections.append({
                "class": name,
                "color": color_name,
                "confidence": float(box.conf[0].item()),
                "bbox_xyxy": [x1, y1, x2, y2],
                "center_px": [cx, cy],
            })

    # Non-maximum suppression (IoU > 0.35)
    raw_detections = sorted(raw_detections, key=lambda d: d["confidence"], reverse=True)
    kept = []
    for cand in raw_detections:
        c_bb = cand["bbox_xyxy"]
        dup = False
        for prev in kept:
            p_bb = prev["bbox_xyxy"]
            iw = max(0.0, min(c_bb[2], p_bb[2]) - max(c_bb[0], p_bb[0]))
            ih = max(0.0, min(c_bb[3], p_bb[3]) - max(c_bb[1], p_bb[1]))
            inter = iw * ih
            union = (c_bb[2]-c_bb[0])*(c_bb[3]-c_bb[1]) + (p_bb[2]-p_bb[0])*(p_bb[3]-p_bb[1]) - inter
            if union > 0 and (inter / union) > 0.35:
                dup = True
                break
        if not dup:
            kept.append(cand)
    return kept


def run_can_experiment(
    exp_dir: Path,
    yolo_model_path: Path,
    blender_bin: str = "/snap/bin/blender",
    cans: list[dict] | None = None,
    cam0_pos: tuple[float, float, float] = (-0.28, -0.85, 0.28),
    cam1_pos: tuple[float, float, float] = (0.28, -0.85, 0.28),
) -> dict:
    inputs_dir = exp_dir / "inputs"
    artifacts_dir = exp_dir / "artifacts"
    outputs_dir = exp_dir / "outputs" / "yolo_triangulation"

    for d in (inputs_dir, artifacts_dir, outputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    if cans is None:
        # 3 solid colored cans (Red, Green, Blue)
        cans = [
            {"name": "can_red", "color": [0.85, 0.08, 0.08], "position": [-0.16, 0.0, 0.0625], "radius": CAN_RADIUS_M, "height": CAN_HEIGHT_M},
            {"name": "can_green", "color": [0.08, 0.85, 0.08], "position": [0.00, 0.06, 0.0625], "radius": CAN_RADIUS_M, "height": CAN_HEIGHT_M},
            {"name": "can_blue", "color": [0.08, 0.20, 0.88], "position": [0.18, -0.04, 0.0625], "radius": CAN_RADIUS_M, "height": CAN_HEIGHT_M},
        ]

    # Save Ground Truth
    gt_scene = {
        "cans": cans,
        "camera0_position": list(cam0_pos),
        "camera1_position": list(cam1_pos),
        "intrinsics": {
            "fx": FOCAL_PX, "fy": FOCAL_PX,
            "cx": WIDTH / 2.0, "cy": HEIGHT / 2.0,
            "width": WIDTH, "height": HEIGHT, "hfov_deg": H_FOV_DEG
        }
    }
    (inputs_dir / "ground_truth.json").write_text(json.dumps(gt_scene, indent=2) + "\n", encoding="utf-8")

    # 1. Blender scene building script
    script_str = generate_blender_scene_script(
        artifacts_dir=artifacts_dir,
        inputs_dir=inputs_dir,
        cans=cans,
        cam0_pos=cam0_pos,
        cam1_pos=cam1_pos,
    )
    script_path = artifacts_dir / "build_scene.py"
    script_path.write_text(script_str, encoding="utf-8")

    # 2. Render
    cmd = [blender_bin, "--background", "--factory-startup", "--python", str(script_path)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Blender failed:\n{res.stderr}\n{res.stdout}")

    # 3. Read camera calibration matrices
    cam_data = json.loads((inputs_dir / "ground_truth_cameras.json").read_text(encoding="utf-8"))
    M0 = np.array(cam_data["cam0_matrix_world"])
    M1 = np.array(cam_data["cam1_matrix_world"])
    R0, t0 = get_cv_extrinsics(M0)
    R1, t1 = get_cv_extrinsics(M1)

    P0 = K @ np.hstack((R0, t0.reshape(3, 1)))
    P1 = K @ np.hstack((R1, t1.reshape(3, 1)))

    # 4. Load renders and detect cans
    img0 = cv2.imread(str(inputs_dir / "cam0_render.png"))
    img1 = cv2.imread(str(inputs_dir / "cam1_render.png"))

    model = YOLO(str(yolo_model_path))
    dets0 = detect_cans_yolo(img0, model)
    dets1 = detect_cans_yolo(img1, model)

    annotated0 = img0.copy()
    annotated1 = img1.copy()

    for d in dets0:
        bb = [int(v) for v in d["bbox_xyxy"]]
        cv2.rectangle(annotated0, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
        cv2.putText(annotated0, f"{d['color']} {d['confidence']:.2f}", (bb[0], bb[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    for d in dets1:
        bb = [int(v) for v in d["bbox_xyxy"]]
        cv2.rectangle(annotated1, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
        cv2.putText(annotated1, f"{d['color']} {d['confidence']:.2f}", (bb[0], bb[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Match by color first, then x-coordinate
    matched_pairs = []
    used_1 = set()
    for i0, d0 in enumerate(dets0):
        for i1, d1 in enumerate(dets1):
            if i1 in used_1:
                continue
            if d0["color"] == d1["color"] and d0["color"] != "unknown":
                matched_pairs.append((d0, d1))
                used_1.add(i1)
                break
    
    # Fallback to coordinate ordering if color matching incomplete
    if len(matched_pairs) < min(len(dets0), len(dets1)):
        dets0_sorted = sorted(dets0, key=lambda d: d["center_px"][0])
        dets1_sorted = sorted(dets1, key=lambda d: d["center_px"][0])
        matched_pairs = list(zip(dets0_sorted, dets1_sorted))

    triangulation_results = []
    for d0, d1 in matched_pairs:
        p0 = tuple(d0["center_px"])
        p1 = tuple(d1["center_px"])

        pt3d, reproj_err = triangulate_2d_points(p0, p1, P0, P1)

        gt_can = None
        min_dist = float("inf")
        for c in cans:
            dist = float(np.linalg.norm(pt3d - np.array(c["position"])))
            if dist < min_dist:
                min_dist = dist
                gt_can = c

        err_m = min_dist
        err_cm = min_dist * 100.0

        triangulation_results.append({
            "detected_color": d0["color"],
            "detected_cam0": d0,
            "detected_cam1": d1,
            "estimated_3d_position": [round(float(v), 4) for v in pt3d],
            "reprojection_error_px": round(float(reproj_err), 3),
            "ground_truth_name": gt_can["name"] if gt_can else "unknown",
            "ground_truth_position": gt_can["position"] if gt_can else None,
            "position_error_m": round(float(err_m), 4),
            "position_error_cm": round(float(err_cm), 2),
        })

        bb0 = [int(v) for v in d0["bbox_xyxy"]]
        bb1 = [int(v) for v in d1["bbox_xyxy"]]
        cv2.putText(annotated0, f"3D:[{pt3d[0]:.2f},{pt3d[1]:.2f},{pt3d[2]:.2f}]",
                    (bb0[0], bb0[3] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(annotated1, f"3D:[{pt3d[0]:.2f},{pt3d[1]:.2f},{pt3d[2]:.2f}]",
                    (bb1[0], bb1[3] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    cv2.imwrite(str(outputs_dir / "cam0_annotated.png"), annotated0)
    cv2.imwrite(str(outputs_dir / "cam1_annotated.png"), annotated1)

    mean_err_cm = float(np.mean([r["position_error_cm"] for r in triangulation_results])) if triangulation_results else float("inf")
    passed = len(triangulation_results) >= len(cans) and mean_err_cm < 5.0

    report = {
        "status": "PASS" if passed else "FAIL",
        "num_cans_gt": len(cans),
        "num_cans_triangulated": len(triangulation_results),
        "mean_position_error_cm": round(mean_err_cm, 2) if mean_err_cm != float("inf") else None,
        "results": triangulation_results,
    }

    (outputs_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", default="/snap/bin/blender")
    parser.add_argument("--family", default="can_triangulation")
    parser.add_argument("--id", default="002")
    parser.add_argument("--weights", default="/home/mfk01/Projects/econ-systems/demos/demo2/yolo26s.pt")
    args = parser.parse_args()

    exp_dir = Path("experiment") / args.family / args.id
    print(f"Running Can Triangulation Experiment: {exp_dir}")
    report = run_can_experiment(exp_dir, yolo_model_path=Path(args.weights), blender_bin=args.blender)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
