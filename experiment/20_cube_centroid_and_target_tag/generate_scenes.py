"""Synthetic Scene Generator for 2-Camera, 1 Base Tag, 6-Face Cube, and Target Marker.

According to TESTING_PROCEDURE.md:
- 2 Cameras (cam0, cam1)
- 1 Large Base Marker: Tag 0 (134 mm) at origin (0, -0.22, 0)
- 1 Target Marker: Tag 1 (124 mm) at (-0.18, 0.12, 0.10)
- 1 Rigid 6-Faced Cube: Tags 6..11 (57.5 mm) centered at (0.12, 0.10, 0.14)
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
import numpy as np

WIDTH = 1280
HEIGHT = 720
FOCAL_PX = 900.0
SENSOR_WIDTH_MM = 36.0
LENS_MM = FOCAL_PX * SENSOR_WIDTH_MM / WIDTH

EXP_DIR = Path(__file__).resolve().parent
SHARED_DIR = EXP_DIR.parent / "18_three_cam_platform_ransac" / "shared"


def build_blender_script(
    scene_dir: Path,
    scene_spec: dict,
) -> str:
    return f'''import bpy
import json
import math
import os
import numpy as np

out_dir = r"{scene_dir.resolve()}"
inputs_dir = os.path.join(out_dir, "inputs")
artifacts_dir = os.path.join(out_dir, "artifacts")
os.makedirs(inputs_dir, exist_ok=True)
os.makedirs(artifacts_dir, exist_ok=True)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

scene = bpy.context.scene
available_engines = {{item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}}
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in available_engines else "BLENDER_EEVEE"
scene.render.resolution_x = {WIDTH}
scene.render.resolution_y = {HEIGHT}
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0

world = bpy.data.worlds.new("NeutralWorld")
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.55, 0.55, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
scene.world = world

def create_tag_material(name, img_path):
    mat = bpy.data.materials.new(name)
    nodes = mat.node_tree.nodes
    nodes.clear()
    img = bpy.data.images.load(img_path, check_existing=True)
    img.colorspace_settings.name = "Non-Color"
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Closest"
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    out_node = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(tex.outputs["Color"], emission.inputs["Color"])
    mat.node_tree.links.new(emission.outputs["Emission"], out_node.inputs["Surface"])
    return mat

def get_white_mat():
    mat = bpy.data.materials.get("WhiteBackingMat")
    if mat:
        return mat
    mat = bpy.data.materials.new("WhiteBackingMat")
    nodes = mat.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    out_node = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(emission.outputs["Emission"], out_node.inputs["Surface"])
    return mat

def create_tag_object(name, size_m, mat, location, rotation_euler):
    h = size_m / 2.0
    mesh = bpy.data.meshes.new(f"{{name}}_Mesh")
    mesh.from_pydata([(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)], [], [(0, 1, 2, 3)])
    mesh.update()
    uv = mesh.uv_layers.new(name="UVMap")
    uv_coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for loop_idx, c in zip(mesh.polygons[0].loop_indices, uv_coords):
        uv.data[loop_idx].uv = c
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = rotation_euler
    obj.data.materials.append(mat)

    # For planar tags (base, target), add quiet white backing plate
    if "Cube" not in name:
        hb = h * 1.2
        back_mesh = bpy.data.meshes.new(f"{{name}}_BackMesh")
        back_mesh.from_pydata([(-hb, -hb, -0.0005), (hb, -hb, -0.0005), (hb, hb, -0.0005), (-hb, hb, -0.0005)], [], [(0, 1, 2, 3)])
        back_mesh.update()
        back_obj = bpy.data.objects.new(f"{{name}}_Back", back_mesh)
        scene.collection.objects.link(back_obj)
        back_obj.location = location
        back_obj.rotation_euler = rotation_euler
        back_obj.data.materials.append(get_white_mat())
    return obj

shared_dir = r"{SHARED_DIR.resolve()}"

# 1. Base Tag (ID 0)
base_mat = create_tag_material("BaseMat", os.path.join(shared_dir, "tag_0.png"))
create_tag_object("Tag_0_Base", {scene_spec["base_tag_size_m"]}, base_mat, {scene_spec["base_tag_location"]}, (0.0, 0.0, 0.0))

# 2. Target Marker (Tag 1)
target_mat = create_tag_material("TargetMat", os.path.join(shared_dir, "tag_1.png"))
create_tag_object("Tag_1_Target", {scene_spec["target_tag_size_m"]}, target_mat, {scene_spec["target_tag_location"]}, {scene_spec["target_tag_rotation"]})

# 3. Rigid 6-Faced Cube (Tags 6..11)
cube_spec = {json.dumps(scene_spec["cube"])}
side_L = cube_spec["side_length_m"]
faces = cube_spec["faces"]

# Create each cube face quad facing outward
for tid_str, fdata in faces.items():
    tid = int(tid_str)
    mat = create_tag_material(f"Mat_Tag_{{tid}}", os.path.join(shared_dir, f"tag_{{tid}}.png"))
    create_tag_object(
        f"Cube_Face_{{tid}}",
        side_L,
        mat,
        fdata["location"],
        fdata["rotation_euler"]
    )

target_dummy = bpy.data.objects.new("LookAtDummy", None)
scene.collection.objects.link(target_dummy)
target_dummy.location = {scene_spec["target_look_at"]}

# 2 Cameras
cam_configs = {json.dumps(scene_spec["cameras"])}
cam_objects = {{}}

for cam_id, ccfg in cam_configs.items():
    c_data = bpy.data.cameras.new(cam_id)
    c_data.lens = {LENS_MM}
    c_data.sensor_width = {SENSOR_WIDTH_MM}
    c_data.sensor_fit = "HORIZONTAL"
    c_data.clip_start = 0.01
    c_data.clip_end = 20.0
    c_obj = bpy.data.objects.new(cam_id, c_data)
    scene.collection.objects.link(c_obj)
    c_obj.location = ccfg["location"]
    
    track = c_obj.constraints.new(type="TRACK_TO")
    track.target = target_dummy
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    cam_objects[cam_id] = c_obj

bpy.context.view_layer.update()

# Save scene blend
blend_path = os.path.join(artifacts_dir, "scene.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_path)

K = [
    [{FOCAL_PX}, 0.0, {WIDTH / 2.0}],
    [0.0, {FOCAL_PX}, {HEIGHT / 2.0}],
    [0.0, 0.0, 1.0]
]

gt = {{
    "intrinsics": {{
        "width": {WIDTH},
        "height": {HEIGHT},
        "fx": {FOCAL_PX},
        "fy": {FOCAL_PX},
        "cx": {WIDTH / 2.0},
        "cy": {HEIGHT / 2.0},
        "K": K
    }},
    "base_tag": {{
        "id": 0,
        "size_m": {scene_spec["base_tag_size_m"]},
        "position_world": {scene_spec["base_tag_location"]},
        "rotation_euler": [0.0, 0.0, 0.0]
    }},
    "target_tag": {{
        "id": 1,
        "size_m": {scene_spec["target_tag_size_m"]},
        "position_world": {scene_spec["target_tag_location"]},
        "rotation_euler": {scene_spec["target_tag_rotation"]}
    }},
    "cube": {{
        "side_length_m": side_L,
        "centroid_world": cube_center,
        "rotation_euler": cube_spec["rotation_euler"],
        "faces": faces
    }},
    "cameras": {{}}
}}

R_b_cv = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0]
])

for cam_id, c_obj in cam_objects.items():
    mw = np.array(c_obj.matrix_world)
    R_b_cam = mw[:3, :3]
    t_b_cam = mw[:3, 3]

    R_w_c = np.array(R_b_cam @ R_b_cv, dtype=float)
    t_w_c = np.array(t_b_cam, dtype=float)

    R_c_w = R_w_c.T
    t_c_w = -R_c_w @ t_w_c

    tr = np.trace(R_c_w)
    angle = math.acos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-6:
        rvec = [0.0, 0.0, 0.0]
    else:
        rx = float((R_c_w[2, 1] - R_c_w[1, 2]) / (2.0 * math.sin(angle)))
        ry = float((R_c_w[0, 2] - R_c_w[2, 0]) / (2.0 * math.sin(angle)))
        rz = float((R_c_w[1, 0] - R_c_w[0, 1]) / (2.0 * math.sin(angle)))
        rvec = [rx * angle, ry * angle, rz * angle]

    gt["cameras"][cam_id] = {{
        "position_world": [float(x) for x in t_b_cam],
        "R_camera_to_world": [[float(x) for x in row] for row in R_w_c],
        "R_world_to_camera": [[float(x) for x in row] for row in R_c_w],
        "t_world_to_camera": [float(x) for x in t_c_w],
        "rvec": [float(x) for x in rvec]
    }}

    scene.camera = c_obj
    render_path = os.path.join(inputs_dir, f"{{cam_id}}_render.png")
    scene.render.filepath = render_path
    bpy.ops.render.render(write_still=True)
    print(f"Rendered {{render_path}}")

gt_path = os.path.join(inputs_dir, "ground_truth.json")
with open(gt_path, "w") as f:
    json.dump(gt, f, indent=2)
print(f"Exported ground truth to {{gt_path}}")
'''


def generate_cube_faces(center: list[float], side_L: float, rot_euler: list[float]) -> dict:
    rx, ry, rz = rot_euler
    # Rotation matrix
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    R_cube = Rz @ Ry @ Rx

    half_L = side_L / 2.0
    c_arr = np.array(center)

    # Local face definitions (outward normal, rotation euler relative to cube)
    # Tag 6: Top (+Z)
    # Tag 7: Bottom (-Z)
    # Tag 8: Front (+Y)
    # Tag 9: Back (-Y)
    # Tag 10: Right (+X)
    # Tag 11: Left (-X)
    # Local face definitions (outward normal, rotation euler relative to cube)
    # Tag 6: Top (+Z) -> normal (0, 0, 1) -> rot (0, 0, 0)
    # Tag 7: Bottom (-Z) -> normal (0, 0, -1) -> rot (pi, 0, 0)
    # Tag 8: Front (-Y in Blender coordinate frame, facing cameras at -Y)
    # In Blender, cameras are at negative Y (-0.7).
    # A face with normal pointing to -Y has rotation (pi/2, 0, 0)
    # A face with normal pointing to +Y has rotation (-pi/2, 0, 0)
    # A face with normal pointing to +X has rotation (0, pi/2, 0)
    # A face with normal pointing to -X has rotation (0, -pi/2, 0)
    local_defs = {
        6:  (np.array([ 0.0,  0.0,  half_L]), [0.0, 0.0, 0.0]),           # Top (+Z)
        7:  (np.array([ 0.0,  0.0, -half_L]), [math.pi, 0.0, 0.0]),      # Bottom (-Z)
        8:  (np.array([ 0.0, -half_L,  0.0]), [math.pi/2, 0.0, 0.0]),     # Front facing -Y (directly towards cameras)
        9:  (np.array([ 0.0,  half_L,  0.0]), [-math.pi/2, 0.0, 0.0]),    # Back facing +Y
        10: (np.array([ half_L,  0.0,  0.0]), [0.0, math.pi/2, 0.0]),     # Right facing +X
        11: (np.array([-half_L,  0.0,  0.0]), [0.0, -math.pi/2, 0.0]),    # Left facing -X
    }

    faces = {}
    for tid, (lpos, lrot) in local_defs.items():
        wpos = R_cube @ lpos + c_arr
        faces[str(tid)] = {
            "size_m": side_L,
            "location": [float(x) for x in wpos],
            "rotation_euler": [float(x + y) for x, y in zip(rot_euler, lrot)],
            "local_offset": [float(x) for x in lpos]
        }
    return faces


def run_scene_generation():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    side_L = 0.0885  # 88.5 mm per face

    # Tilt cube slightly backward (rx > 0) so Top (+Z) and Front (-Y) are visible
    # Scene 001: Front (8) and Top (6) facing cameras
    cube_center_001 = [0.12, 0.10, 0.14]
    cube_rot_001 = [0.45, 0.0, 0.0]
    faces_001 = generate_cube_faces(cube_center_001, side_L, cube_rot_001)

    # Scene 002: Front (8), Top (6), and Left/Right face cameras
    cube_center_002 = [0.10, 0.08, 0.16]
    cube_rot_002 = [0.40, 0.35, -0.20]
    faces_002 = generate_cube_faces(cube_center_002, side_L, cube_rot_002)

    # Scene 003: Front (8) and Top (6) facing cameras (with synthetic outlier in benchmark)
    cube_center_003 = [0.12, 0.10, 0.14]
    cube_rot_003 = [0.45, 0.0, 0.0]
    faces_003 = generate_cube_faces(cube_center_003, side_L, cube_rot_003)

    scenes_spec = {
        "001": {
            "name": "2-Cam Cube Centroid and Target Marker Nominal",
            "base_tag_size_m": 0.134,
            "base_tag_location": [0.0, -0.22, 0.0],
            "target_tag_size_m": 0.124,
            "target_tag_location": [-0.18, 0.12, 0.10],
            "target_tag_rotation": [0.0, 0.0, 0.0],
            "cube": {
                "side_length_m": side_L,
                "center": cube_center_001,
                "rotation_euler": cube_rot_001,
                "faces": faces_001
            },
            "target_look_at": [0.0, -0.05, 0.08],
            "cameras": {
                "cam0": {"location": [-0.38, -0.68, 0.72]},
                "cam1": {"location": [ 0.38, -0.68, 0.72]}
            }
        },
        "002": {
            "name": "2-Cam Cube Centroid and Target Marker Tilted Angles",
            "base_tag_size_m": 0.134,
            "base_tag_location": [0.0, -0.22, 0.0],
            "target_tag_size_m": 0.124,
            "target_tag_location": [-0.16, 0.10, 0.12],
            "target_tag_rotation": [math.radians(8), math.radians(-5), 0.0],
            "cube": {
                "side_length_m": side_L,
                "center": cube_center_002,
                "rotation_euler": cube_rot_002,
                "faces": faces_002
            },
            "target_look_at": [0.0, -0.05, 0.08],
            "cameras": {
                "cam0": {"location": [-0.40, -0.70, 0.75]},
                "cam1": {"location": [ 0.40, -0.70, 0.75]}
            }
        },
        "003": {
            "name": "2-Cam Cube Centroid with Synthetic Outlier Noise",
            "base_tag_size_m": 0.134,
            "base_tag_location": [0.0, -0.22, 0.0],
            "target_tag_size_m": 0.124,
            "target_tag_location": [-0.20, 0.14, 0.08],
            "target_tag_rotation": [0.0, 0.0, 0.0],
            "cube": {
                "side_length_m": side_L,
                "center": cube_center_003,
                "rotation_euler": cube_rot_003,
                "faces": faces_003
            },
            "target_look_at": [0.0, -0.05, 0.08],
            "cameras": {
                "cam0": {"location": [-0.38, -0.68, 0.72]},
                "cam1": {"location": [ 0.38, -0.68, 0.72]}
            }
        }
    }

    for scene_id, spec in scenes_spec.items():
        sdir = EXP_DIR / scene_id
        inputs_dir = sdir / "inputs"
        artifacts_dir = sdir / "artifacts"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        with open(inputs_dir / "scene_spec.json", "w") as f:
            json.dump(spec, f, indent=2)

        script = build_blender_script(sdir, spec)
        build_py_path = artifacts_dir / "build_scene.py"
        with open(build_py_path, "w") as f:
            f.write(script)

        print(f"\n--- Generating Scene {scene_id}: {spec['name']} ---")
        cmd = ["/snap/bin/blender", "-b", "-P", str(build_py_path)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Error rendering scene {scene_id}:\n", res.stderr)
        else:
            print(f"Scene {scene_id} generated successfully!")


if __name__ == "__main__":
    run_scene_generation()
