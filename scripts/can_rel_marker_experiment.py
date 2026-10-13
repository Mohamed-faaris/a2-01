#!/usr/bin/env python3
"""Stereo can triangulation with AprilTag marker reference frame.

Inputs:
  - marker size (e.g. 0.08 m)
  - AprilTag placed on the scene plane/floor
  - Stereo camera pair (cam0, cam1)
  - Solid colored cylinder cans (Red, Green, Blue)

Outputs:
  - Detect marker pose in camera frame (R_c_m, t_c_m)
  - Detect & triangulate cans in 3D camera/world frame
  - Transform triangulated 3D can locations relative to the marker coordinate frame:
      P_rel_marker = R_marker_to_cam^T * (P_cam - t_marker_to_cam)
      or in marker top-left corner reference (origin at corner):
      P_rel_corner = P_rel_marker - [-size/2, -size/2, 0]
  - Error evaluation comparing estimated relative coordinates to ground truth.

Folder structure:
  experiment/can_rel_to_marker/
    shared/
    reports/
    001/
      inputs/
      artifacts/
      outputs/
        yolo_marker_rel_pose/
    002/
      inputs/
      artifacts/
      outputs/
        yolo_marker_rel_pose/
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

# Default marker parameters
DEFAULT_MARKER_SIZE_M = 0.08  # 8 cm
DEFAULT_MARKER_ID = 0

CAN_RADIUS_M = 0.035
CAN_HEIGHT_M = 0.125


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

# Neutral floor plane
bpy.ops.mesh.primitive_plane_add(size=12.0, location=(0, 0, 0))
floor = bpy.context.active_object
mat_f = bpy.data.materials.new("FloorMat")
mat_f.use_nodes = True
mat_f.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.35, 0.35, 0.35, 1.0)
floor.data.materials.append(mat_f)

# AprilTag Marker Material (Emission to keep crisp contrast)
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

# Marker Mesh (plane on floor, z=0.001 to prevent z-fighting)
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

# Cans
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

# Light
light_data = bpy.data.lights.new(name="Sun", type='SUN')
light_data.energy = 4.0
light_obj = bpy.data.objects.new("Sun", light_data)
scene.collection.objects.link(light_obj)
light_obj.location = (2.0, -2.0, 4.0)
light_obj.rotation_euler = (math.radians(35), math.radians(20), math.radians(45))

# Target point: center of all cans + marker
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


def detect_cans_yolo(
    image_bgr: np.ndarray,
    model: YOLO,
    conf: float = 0.005,
) -> list[dict]:
    res = model.predict(source=image_bgr, conf=conf, verbose=False)[0]
    raw_detections = []
    target_classes = {"bottle", "cup", "vase"}
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


def run_experiment(
    exp_dir: Path,
    yolo_model_path: Path,
    marker_size: float = DEFAULT_MARKER_SIZE_M,
    marker_id: int = DEFAULT_MARKER_ID,
    marker_pos: tuple[float, float, float] = (0.0, -0.05, 0.001),
    cans: list[dict] | None = None,
    cam0_pos: tuple[float, float, float] = (-0.30, -0.95, 0.35),
    cam1_pos: tuple[float, float, float] = (0.30, -0.95, 0.35),
    blender_bin: str = "/snap/bin/blender",
) -> dict:
    shared_dir = exp_dir.parent / "shared"
    inputs_dir = exp_dir / "inputs"
    artifacts_dir = exp_dir / "artifacts"
    outputs_dir = exp_dir / "outputs" / "yolo_marker_rel_pose"

    for d in (shared_dir, inputs_dir, artifacts_dir, outputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    marker_path = ensure_marker_image(shared_dir, marker_id)

    if cans is None:
        cans = [
            {"name": "can_red", "color": [0.85, 0.08, 0.08], "position": [-0.18, 0.05, 0.0625], "radius": CAN_RADIUS_M, "height": CAN_HEIGHT_M},
            {"name": "can_green", "color": [0.08, 0.85, 0.08], "position": [0.00, 0.12, 0.0625], "radius": CAN_RADIUS_M, "height": CAN_HEIGHT_M},
            {"name": "can_blue", "color": [0.08, 0.20, 0.88], "position": [0.18, 0.03, 0.0625], "radius": CAN_RADIUS_M, "height": CAN_HEIGHT_M},
        ]

    # Ground truth relative coordinates
    # Marker coordinate origin:
    # 1. Center of marker (x=0, y=0, z=0 in marker frame)
    # 2. Measurable Corner: Bottom-Left / Top-Left corner of the physical marker square
    # Corner in world: (marker_pos[0] - marker_size/2, marker_pos[1] - marker_size/2, marker_pos[2])
    corner_world = np.array([
        marker_pos[0] - marker_size / 2.0,
        marker_pos[1] - marker_size / 2.0,
        marker_pos[2]
    ])

    for c in cans:
        p = np.array(c["position"])
        c["gt_rel_to_marker_center_m"] = (p - np.array(marker_pos)).tolist()
        c["gt_rel_to_marker_corner_m"] = (p - corner_world).tolist()
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
        "intrinsics": {
            "fx": FOCAL_PX, "fy": FOCAL_PX,
            "cx": WIDTH / 2.0, "cy": HEIGHT / 2.0,
            "width": WIDTH, "height": HEIGHT, "hfov_deg": H_FOV_DEG
        }
    }
    (inputs_dir / "ground_truth.json").write_text(json.dumps(gt_data, indent=2) + "\n", encoding="utf-8")

    # 1. Blender scene building script
    script_str = generate_blender_scene_script(
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

    # 2. Render
    cmd = [blender_bin, "--background", "--factory-startup", "--python", str(artifacts_dir / "build_scene.py")]
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

    img0 = cv2.imread(str(inputs_dir / "cam0_render.png"))
    img1 = cv2.imread(str(inputs_dir / "cam1_render.png"))

    # 4. Marker detection (AprilTag) in both cameras
    detector = Detector(families="tag36h11", nthreads=2, quad_decimate=1.0)
    det0_m, R_c0_m, t_c0_m = detect_marker_pose(img0, marker_id, marker_size, detector)
    det1_m, R_c1_m, t_c1_m = detect_marker_pose(img1, marker_id, marker_size, detector)

    if det0_m is None and det1_m is None:
        raise RuntimeError("Marker not detected in either camera view")

    # Marker center in world space from Camera 0:
    # P_c0 = R0 * P_w + t0  => P_w = R0.T * (P_c0 - t0)
    marker_center_world_est = R0.T @ (t_c0_m - t0)
    
    # 5. Detect Cans with YOLO26s
    model = YOLO(str(yolo_model_path))
    dets0 = detect_cans_yolo(img0, model)
    dets1 = detect_cans_yolo(img1, model)

    annotated0 = img0.copy()
    annotated1 = img1.copy()

    # Draw marker on images
    if det0_m:
        corners = np.asarray(det0_m.corners, dtype=int)
        cv2.polylines(annotated0, [corners], True, (0, 255, 255), 2)
        cv2.putText(annotated0, f"Marker #{marker_id}", (corners[0, 0], corners[0, 1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    if det1_m:
        corners = np.asarray(det1_m.corners, dtype=int)
        cv2.polylines(annotated1, [corners], True, (0, 255, 255), 2)
        cv2.putText(annotated1, f"Marker #{marker_id}", (corners[0, 0], corners[0, 1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Match by color
    matched_pairs = []
    used_1 = set()
    for d0 in dets0:
        for i1, d1 in enumerate(dets1):
            if i1 in used_1:
                continue
            if d0["color"] == d1["color"] and d0["color"] != "unknown":
                matched_pairs.append((d0, d1))
                used_1.add(i1)
                break

    if len(matched_pairs) < min(len(dets0), len(dets1)):
        dets0_sorted = sorted(dets0, key=lambda d: d["center_px"][0])
        dets1_sorted = sorted(dets1, key=lambda d: d["center_px"][0])
        matched_pairs = list(zip(dets0_sorted, dets1_sorted))

    can_evaluations = []
    # Marker measurable corner offset in world
    corner_offset_world = np.array([-marker_size / 2.0, -marker_size / 2.0, 0.0])
    marker_corner_world_est = marker_center_world_est + corner_offset_world

    for d0, d1 in matched_pairs:
        p0 = tuple(d0["center_px"])
        p1 = tuple(d1["center_px"])

        can_3d_world, reproj = triangulate_2d_points(p0, p1, P0, P1)

        # Match to closest GT can
        gt_can = None
        min_dist = float("inf")
        for c in cans:
            dist = float(np.linalg.norm(can_3d_world - np.array(c["position"])))
            if dist < min_dist:
                min_dist = dist
                gt_can = c

        # Position relative to marker center and corner
        rel_to_center_est = can_3d_world - marker_center_world_est
        rel_to_corner_est = can_3d_world - marker_corner_world_est
        dist_to_corner_est = float(np.linalg.norm(rel_to_corner_est))

        gt_corner_dist = gt_can["gt_dist_to_marker_corner_m"] if gt_can else 0.0
        dist_err_m = abs(dist_to_corner_est - gt_corner_dist)
        dist_err_cm = dist_err_m * 100.0

        can_evaluations.append({
            "can_name": gt_can["name"] if gt_can else "unknown",
            "detected_color": d0["color"],
            "estimated_world_position_m": [round(float(v), 4) for v in can_3d_world],
            "estimated_rel_marker_center_m": [round(float(v), 4) for v in rel_to_center_est],
            "estimated_rel_marker_corner_m": [round(float(v), 4) for v in rel_to_corner_est],
            "estimated_dist_to_corner_m": round(dist_to_corner_est, 4),
            "ground_truth_dist_to_corner_m": round(gt_corner_dist, 4),
            "corner_distance_error_cm": round(dist_err_cm, 2),
            "reprojection_error_px": round(float(reproj), 3),
        })

        # Draw overlays
        bb0 = [int(v) for v in d0["bbox_xyxy"]]
        bb1 = [int(v) for v in d1["bbox_xyxy"]]
        cv2.rectangle(annotated0, (bb0[0], bb0[1]), (bb0[2], bb0[3]), (0, 255, 0), 2)
        cv2.rectangle(annotated1, (bb1[0], bb1[1]), (bb1[2], bb1[3]), (0, 255, 0), 2)
        cv2.putText(annotated0, f"{d0['color']} d={dist_to_corner_est*100:.1f}cm",
                    (bb0[0], bb0[3] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(annotated1, f"{d1['color']} d={dist_to_corner_est*100:.1f}cm",
                    (bb1[0], bb1[3] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    cv2.imwrite(str(outputs_dir / "cam0_annotated.png"), annotated0)
    cv2.imwrite(str(outputs_dir / "cam1_annotated.png"), annotated1)

    mean_err_cm = float(np.mean([r["corner_distance_error_cm"] for r in can_evaluations])) if can_evaluations else float("inf")
    passed = len(can_evaluations) >= len(cans) and mean_err_cm < 5.0

    report = {
        "status": "PASS" if passed else "FAIL",
        "marker_input": {
            "marker_id": marker_id,
            "marker_size_m": marker_size,
        },
        "num_cans_triangulated": len(can_evaluations),
        "mean_corner_distance_error_cm": round(mean_err_cm, 2) if mean_err_cm != float("inf") else None,
        "cans_relative_to_marker_corner": can_evaluations,
    }

    (outputs_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", default="/snap/bin/blender")
    parser.add_argument("--family", default="can_rel_to_marker")
    parser.add_argument("--id", default="001")
    parser.add_argument("--marker-size", type=float, default=0.08)
    parser.add_argument("--marker-id", type=int, default=0)
    parser.add_argument("--weights", default="/home/mfk01/Projects/econ-systems/demos/demo2/yolo26s.pt")
    args = parser.parse_args()

    exp_dir = Path("experiment") / args.family / args.id
    print(f"Running Can Relative to Marker Experiment: {exp_dir} (marker_size={args.marker_size}m)")
    report = run_experiment(
        exp_dir,
        yolo_model_path=Path(args.weights),
        marker_size=args.marker_size,
        marker_id=args.marker_id,
        blender_bin=args.blender
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
