#!/usr/bin/env python3
"""Single Camera, Depth-Assisted 3-Can 3D (X, Y, Z) Localization relative to AprilTag Corner Origin.

Setup:
  - 1 Camera
  - 1 AprilTag marker (defines coordinate reference frame and corner origin)
  - 3 Cans placed freely in 3D space (unconstrained Z heights / varied positions)
  - Input: 2D detection center + ground-truth depth from camera center (Z_cam)
  - Output: Full 3D (x, y, z) coordinates of each can in the reference frame of the tag,
            where (0, 0, 0) is the corner of the tag.

Follows TESTING_PROCEDURE.md directory structure under:
  experiment/one_cam_depth_cans_3d/
    shared/
    reports/
    <id>/
      inputs/
      artifacts/
      outputs/depth_backprojection_3d/
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

WIDTH, HEIGHT = 1080, 720
H_FOV_DEG = 60.0
FOCAL_PX = (WIDTH / 2.0) / math.tan(math.radians(H_FOV_DEG / 2.0))
K = np.array([
    [FOCAL_PX, 0.0, WIDTH / 2.0],
    [0.0, FOCAL_PX, HEIGHT / 2.0],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

DEFAULT_MARKER_SIZE_M = 0.088
DEFAULT_MARKER_ID = 1
CAN_RADIUS_M = 0.035
CAN_HEIGHT_M = 0.125
DEFAULT_YOLO_MODEL = "/home/mfk01/Projects/econ-systems/demos/demo2/yolo26s.pt"


def ensure_marker_image(shared_dir: Path, marker_id: int) -> Path:
    shared_dir.mkdir(parents=True, exist_ok=True)
    marker_path = shared_dir / f"marker_{marker_id}.png"
    if not marker_path.exists():
        family = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        marker = cv2.aruco.generateImageMarker(family, marker_id, 512, borderBits=1)
        cv2.imwrite(str(marker_path), marker)
    return marker_path


def generate_blender_scene_script(
    artifacts_dir: Path,
    inputs_dir: Path,
    marker_path: Path,
    marker_size: float,
    marker_pos: tuple[float, float, float],
    cans: list[dict],
    cam_pos: tuple[float, float, float],
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

# AprilTag Marker Material (emission for crisp detection)
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

# Cans placed in 3D space
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
    bsdf.inputs["Roughness"].default_value = 0.3
    obj.data.materials.append(mat)

# Directional light
light_data = bpy.data.lights.new(name="Sun", type='SUN')
light_data.energy = 4.0
light_obj = bpy.data.objects.new("Sun", light_data)
scene.collection.objects.link(light_obj)
light_obj.location = (2.0, -2.0, 4.0)
light_obj.rotation_euler = (math.radians(35), math.radians(20), math.radians(45))

# Camera target
all_pts = [c["position"] for c in cans] + [{marker_pos}]
avg_x = sum(p[0] for p in all_pts) / len(all_pts)
avg_y = sum(p[1] for p in all_pts) / len(all_pts)
avg_z = sum(p[2] for p in all_pts) / len(all_pts)
target_loc = mathutils.Vector((avg_x, avg_y, avg_z))

c_data = bpy.data.cameras.new("Camera_0")
c_data.lens = {FOCAL_PX} * 36.0 / {WIDTH}
c_data.sensor_width = 36.0
c_data.sensor_fit = "HORIZONTAL"
c_data.clip_start = 0.01
c_data.clip_end = 100.0
c_obj = bpy.data.objects.new("Camera_0", c_data)
scene.collection.objects.link(c_obj)
c_obj.location = {cam_pos}
dir_vec = target_loc - mathutils.Vector({cam_pos})
c_obj.rotation_euler = dir_vec.to_track_quat('-Z', 'Y').to_euler()

bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(art_dir, "scene.blend"))

scene.camera = c_obj
scene.render.filepath = os.path.join(inp_dir, "cam0_render.png")
bpy.ops.render.render(write_still=True)

with open(os.path.join(inp_dir, "ground_truth_cameras.json"), "w") as f:
    json.dump({{
        "cam0_matrix_world": [list(row) for row in c_obj.matrix_world],
        "cam0_position": list(c_obj.location),
    }}, f, indent=2)
'''


def get_cv_extrinsics(M_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert Blender camera matrix_world to OpenCV extrinsics (R_w2c, t_w2c)."""
    R_b = M_world[:3, :3]
    t_b = M_world[:3, 3]
    R_cv2b = np.diag([1.0, -1.0, -1.0])
    R_w2c = R_cv2b @ R_b.T
    t_w2c = -R_w2c @ t_b
    return R_w2c, t_w2c


def detect_marker_pose(image_bgr: np.ndarray, marker_id: int, marker_size: float, detector: Detector):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    intrinsics = (FOCAL_PX, FOCAL_PX, WIDTH / 2.0, HEIGHT / 2.0)
    detections = detector.detect(gray, estimate_tag_pose=True, camera_params=intrinsics, tag_size=marker_size)
    for det in detections:
        if det.tag_id == marker_id:
            R = np.asarray(det.pose_R, dtype=float)
            t = np.asarray(det.pose_t, dtype=float).reshape(3)
            return det, R, t
    return None, None, None


def detect_cans_yolo(image_bgr: np.ndarray, model: YOLO, conf: float = 0.005) -> list[dict]:
    res = model.predict(source=image_bgr, conf=conf, verbose=False)[0]
    raw_detections = []
    target_classes = {"bottle", "cup", "vase", "can"}
    for box in res.boxes:
        cls_id = int(box.cls[0].item())
        name = res.names[cls_id]
        if name in target_classes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].cpu().numpy()]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

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

    # Non-maximum suppression by overlap
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


def backproject_pixel_with_depth(u: float, v: float, depth_z: float) -> np.ndarray:
    """Backproject a 2D pixel (u, v) and depth Z_cam to 3D point in OpenCV camera coordinates."""
    x = (u - WIDTH / 2.0) * depth_z / FOCAL_PX
    y = (v - HEIGHT / 2.0) * depth_z / FOCAL_PX
    return np.array([x, y, depth_z], dtype=np.float64)


def run_experiment_scene_3d(
    exp_dir: Path,
    yolo_model_path: Path,
    marker_size: float = DEFAULT_MARKER_SIZE_M,
    marker_id: int = DEFAULT_MARKER_ID,
    marker_pos: tuple[float, float, float] = (0.029, -0.038, 0.001),
    cans: list[dict] | None = None,
    cam_pos: tuple[float, float, float] = (0.0, -0.96, 0.55),
    blender_bin: str = "/snap/bin/blender",
) -> dict:
    shared_dir = exp_dir.parent / "shared"
    inputs_dir = exp_dir / "inputs"
    artifacts_dir = exp_dir / "artifacts"
    outputs_dir = exp_dir / "outputs" / "depth_backprojection_3d"

    for d in (shared_dir, inputs_dir, artifacts_dir, outputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    marker_path = ensure_marker_image(shared_dir, marker_id)

    if cans is None:
        cans = [
            {"name": "can_red", "color": [0.85, 0.08, 0.08], "position": [-0.22, 0.12, 0.15], "radius": CAN_RADIUS_M, "height": CAN_HEIGHT_M},
            {"name": "can_green", "color": [0.08, 0.85, 0.08], "position": [0.03, 0.16, 0.28], "radius": CAN_RADIUS_M, "height": CAN_HEIGHT_M},
            {"name": "can_blue", "color": [0.08, 0.20, 0.88], "position": [0.15, 0.07, 0.09], "radius": CAN_RADIUS_M, "height": CAN_HEIGHT_M},
        ]

    # Tag corner origin in world coordinates:
    tag_corner_world = np.array([
        marker_pos[0] - marker_size / 2.0,
        marker_pos[1] - marker_size / 2.0,
        marker_pos[2]
    ])

    for c in cans:
        p = np.array(c["position"])
        # Full 3D relative to tag corner
        rel_corner = p - tag_corner_world
        c["gt_x_rel_tag_corner_m"] = float(rel_corner[0])
        c["gt_y_rel_tag_corner_m"] = float(rel_corner[1])
        c["gt_z_rel_tag_corner_m"] = float(rel_corner[2])
        c["gt_dist_to_tag_corner_m"] = float(np.linalg.norm(rel_corner))

    # Render scene via Blender
    script_str = generate_blender_scene_script(
        artifacts_dir=artifacts_dir,
        inputs_dir=inputs_dir,
        marker_path=marker_path,
        marker_size=marker_size,
        marker_pos=marker_pos,
        cans=cans,
        cam_pos=cam_pos,
    )
    (artifacts_dir / "build_scene.py").write_text(script_str, encoding="utf-8")

    cmd = [blender_bin, "--background", "--factory-startup", "--python", str(artifacts_dir / "build_scene.py")]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Blender render failed:\n{res.stderr}\n{res.stdout}")

    # Read camera matrices
    cam_data = json.loads((inputs_dir / "ground_truth_cameras.json").read_text(encoding="utf-8"))
    M_world = np.array(cam_data["cam0_matrix_world"])
    R_w2c, t_w2c = get_cv_extrinsics(M_world)

    # Calculate ground truth camera depth Z_cam for each can center
    for c in cans:
        p_world = np.array(c["position"])
        p_cam = R_w2c @ p_world + t_w2c
        c["gt_depth_from_cam_m"] = float(p_cam[2])

    gt_data = {
        "marker": {
            "id": marker_id,
            "size_m": marker_size,
            "center_position": list(marker_pos),
            "corner_origin_position": tag_corner_world.tolist(),
        },
        "cans": cans,
        "camera_position": list(cam_pos),
        "intrinsics": {
            "fx": FOCAL_PX, "fy": FOCAL_PX,
            "cx": WIDTH / 2.0, "cy": HEIGHT / 2.0,
            "width": WIDTH, "height": HEIGHT, "hfov_deg": H_FOV_DEG
        }
    }
    (inputs_dir / "ground_truth.json").write_text(json.dumps(gt_data, indent=2) + "\n", encoding="utf-8")

    # Detect AprilTag
    img = cv2.imread(str(inputs_dir / "cam0_render.png"))
    detector = Detector(families="tag36h11", nthreads=2, quad_decimate=1.0)
    det_m, R_c_tag, t_c_tag = detect_marker_pose(img, marker_id, marker_size, detector)

    if det_m is None:
        raise RuntimeError("AprilTag marker not detected in frame")

    # Detect cans using YOLO
    model = YOLO(str(yolo_model_path))
    detections = detect_cans_yolo(img, model)

    annotated = img.copy()
    corners = np.asarray(det_m.corners, dtype=int)
    cv2.polylines(annotated, [corners], True, (0, 255, 255), 2)
    cv2.circle(annotated, (int(corners[0, 0]), int(corners[0, 1])), 6, (0, 0, 255), -1)
    cv2.putText(annotated, f"Tag #{marker_id} Corner Origin (0,0,0)",
                (corners[0, 0] - 10, corners[0, 1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    evaluations = []
    for d in detections:
        matched_gt = next((c for c in cans if c.get("color_name") == d["color"] or c["name"].endswith(d["color"])), None)
        if matched_gt is None:
            continue

        depth_input = matched_gt["gt_depth_from_cam_m"]
        cx, cy = d["center_px"]

        # Backproject 2D detection center + depth input to 3D camera frame
        can_3d_cam = backproject_pixel_with_depth(cx, cy, depth_input)

        # Express 3D location in AprilTag reference frame
        p_in_tag = R_c_tag.T @ (can_3d_cam - t_c_tag)

        # 3D coordinates relative to Tag Corner Origin:
        # Origin is at [+marker_size/2, -marker_size/2, 0] in tag frame
        est_x = float(-(p_in_tag[0] - marker_size / 2.0))
        est_y = float(p_in_tag[1] - (-marker_size / 2.0))
        est_z = float(-p_in_tag[2])

        gt_x = matched_gt["gt_x_rel_tag_corner_m"]
        gt_y = matched_gt["gt_y_rel_tag_corner_m"]
        gt_z = matched_gt["gt_z_rel_tag_corner_m"]

        err_x_cm = abs(est_x - gt_x) * 100.0
        err_y_cm = abs(est_y - gt_y) * 100.0
        err_z_cm = abs(est_z - gt_z) * 100.0
        dist_3d_err_cm = math.sqrt(err_x_cm**2 + err_y_cm**2 + err_z_cm**2)

        evaluations.append({
            "can_name": matched_gt["name"],
            "detected_color": d["color"],
            "depth_input_m": round(depth_input, 4),
            "estimated_3d_xyz_m": [round(est_x, 4), round(est_y, 4), round(est_z, 4)],
            "ground_truth_3d_xyz_m": [round(gt_x, 4), round(gt_y, 4), round(gt_z, 4)],
            "error_x_cm": round(err_x_cm, 2),
            "error_y_cm": round(err_y_cm, 2),
            "error_z_cm": round(err_z_cm, 2),
            "euclidean_3d_error_cm": round(dist_3d_err_cm, 2),
        })

        bb = [int(v) for v in d["bbox_xyxy"]]
        cv2.rectangle(annotated, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
        cv2.putText(annotated, f"{d['color']}: x={est_x:.2f}, y={est_y:.2f}, z={est_z:.2f}m",
                    (bb[0], bb[3] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1)

    cv2.imwrite(str(outputs_dir / "cam0_annotated.png"), annotated)

    mean_err_cm = float(np.mean([e["euclidean_3d_error_cm"] for e in evaluations])) if evaluations else float("inf")
    passed = len(evaluations) == len(cans) and mean_err_cm < 5.0

    report = {
        "status": "PASS" if passed else "FAIL",
        "inputs": {
            "camera_count": 1,
            "marker_id": marker_id,
            "marker_size_m": marker_size,
            "depth_input_mode": "depth_from_can_center",
            "constraint": "none (free 3D space, unconstrained height/position)",
        },
        "mean_3d_error_cm": round(mean_err_cm, 2),
        "cans_evaluated": evaluations,
    }

    (outputs_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Run 1-cam depth unconstrained 3D (x,y,z) localization")
    parser.add_argument("--base-dir", type=Path, default=Path("experiment/one_cam_depth_cans_3d"))
    parser.add_argument("--yolo-model", type=Path, default=Path(DEFAULT_YOLO_MODEL))
    parser.add_argument("--num-scenes", type=int, default=5)
    args = parser.parse_args()

    args.base_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = args.base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    print(f"=== Starting 1-Camera Unconstrained 3D (X, Y, Z) Localization ({args.num_scenes} scenes) ===")
    np.random.seed(101)

    for i in range(1, args.num_scenes + 1):
        scene_name = f"{i:03d}"
        scene_dir = args.base_dir / scene_name

        # Unconstrained 3D positions: variable x, y, and heights z (floating in 3D air)
        cans = [
            {
                "name": "can_red",
                "color_name": "red",
                "color": [0.85, 0.08, 0.08],
                "position": [
                    float(np.random.uniform(-0.28, -0.08)),
                    float(np.random.uniform(0.05, 0.22)),
                    float(np.random.uniform(0.08, 0.32)),  # unconstrained height z
                ],
                "radius": CAN_RADIUS_M,
                "height": CAN_HEIGHT_M,
            },
            {
                "name": "can_green",
                "color_name": "green",
                "color": [0.08, 0.85, 0.08],
                "position": [
                    float(np.random.uniform(-0.05, 0.10)),
                    float(np.random.uniform(0.12, 0.28)),
                    float(np.random.uniform(0.12, 0.45)),  # unconstrained height z
                ],
                "radius": CAN_RADIUS_M,
                "height": CAN_HEIGHT_M,
            },
            {
                "name": "can_blue",
                "color_name": "blue",
                "color": [0.08, 0.20, 0.88],
                "position": [
                    float(np.random.uniform(0.12, 0.28)),
                    float(np.random.uniform(0.04, 0.18)),
                    float(np.random.uniform(0.06, 0.25)),  # unconstrained height z
                ],
                "radius": CAN_RADIUS_M,
                "height": CAN_HEIGHT_M,
            },
        ]

        # Vary camera pose viewing the 3D volume
        cam_x = float(np.random.uniform(-0.15, 0.15))
        cam_y = float(np.random.uniform(-1.10, -0.90))
        cam_z = float(np.random.uniform(0.40, 0.65))

        print(f"Rendering & Evaluating Scene {scene_name}...")
        report = run_experiment_scene_3d(
            exp_dir=scene_dir,
            yolo_model_path=args.yolo_model,
            cans=cans,
            cam_pos=(cam_x, cam_y, cam_z),
        )

        for c_eval in report["cans_evaluated"]:
            summary_rows.append({
                "scene_id": scene_name,
                "can": c_eval["can_name"],
                "color": c_eval["detected_color"],
                "depth_input_m": c_eval["depth_input_m"],
                "gt_x_m": c_eval["ground_truth_3d_xyz_m"][0],
                "gt_y_m": c_eval["ground_truth_3d_xyz_m"][1],
                "gt_z_m": c_eval["ground_truth_3d_xyz_m"][2],
                "est_x_m": c_eval["estimated_3d_xyz_m"][0],
                "est_y_m": c_eval["estimated_3d_xyz_m"][1],
                "est_z_m": c_eval["estimated_3d_xyz_m"][2],
                "err_x_cm": c_eval["error_x_cm"],
                "err_y_cm": c_eval["error_y_cm"],
                "err_z_cm": c_eval["error_z_cm"],
                "err_3d_cm": c_eval["euclidean_3d_error_cm"],
                "status": report["status"],
            })

    # Save summary reports
    csv_path = reports_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    (reports_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8")

    mean_all_err = np.mean([r["err_3d_cm"] for r in summary_rows])
    print(f"\n[DONE] Finished {args.num_scenes} scenes.")
    print(f"Overall Mean 3D Error: {mean_all_err:.2f} cm")
    print(f"Results saved to: {reports_dir}")


if __name__ == "__main__":
    main()
