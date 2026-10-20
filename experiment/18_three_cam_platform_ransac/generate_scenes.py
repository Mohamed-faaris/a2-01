"""Synthetic Scene Generator for 3-Camera AprilTag Tool Calibrator Benchmark.

Generates 3 synthetic multi-camera scenes following TESTING_PROCEDURE.md:
- 1 Large Base Marker: Tag 0 (134 mm) at world origin (0, 0, 0)
- 4 Platform Corner Markers: Tags 2, 3, 4, 5 (88.5 mm) on a coplanar platform
- 3 Cameras with known intrinsics viewing the scene simultaneously
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
FOCAL_PX = 900.0  # fx = fy = 900
SENSOR_WIDTH_MM = 36.0
LENS_MM = FOCAL_PX * SENSOR_WIDTH_MM / WIDTH  # 25.3125 mm

EXP_DIR = Path(__file__).resolve().parent
SHARED_DIR = EXP_DIR / "shared"


def build_blender_script(
    scene_dir: Path,
    scene_spec: dict,
) -> str:
    """Generate standalone reproducible build_scene.py for Blender."""
    script_content = f'''import bpy
import json
import math
import os
import mathutils
import numpy as np

out_dir = r"{scene_dir.resolve()}"
inputs_dir = os.path.join(out_dir, "inputs")
artifacts_dir = os.path.join(out_dir, "artifacts")
os.makedirs(inputs_dir, exist_ok=True)
os.makedirs(artifacts_dir, exist_ok=True)

# 1. Clean Scene
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

# Neutral ambient lighting
world = bpy.data.worlds.new("NeutralWorld")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.55, 0.55, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
scene.world = world

def create_tag_material(name, img_path):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
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
    mat.use_nodes = True
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
    # Tag corners matching OpenCV AprilTag orientation:
    # 0: [-h, -h], 1: [h, -h], 2: [h, h], 3: [-h, h]
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

    # White quiet margin plate
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

# Create Base Tag (ID 0)
shared_dir = r"{SHARED_DIR.resolve()}"
base_mat = create_tag_material("BaseMat", os.path.join(shared_dir, "tag_0.png"))
base_loc = {scene_spec["base_tag_location"]}
base_obj = create_tag_object("Tag_0_Base", {scene_spec["base_tag_size_m"]}, base_mat, base_loc, (0.0, 0.0, 0.0))

# Create Platform Corner Tags
corner_specs = {json.dumps(scene_spec["platform_corners"])}
platform_objs = []
for tid_str, cdata in corner_specs.items():
    tid = int(tid_str)
    mat = create_tag_material(f"Mat_Tag_{{tid}}", os.path.join(shared_dir, f"tag_{{tid}}.png"))
    obj = create_tag_object(
        f"Tag_{{tid}}",
        cdata["size_m"],
        mat,
        cdata["location"],
        cdata["rotation_euler"]
    )
    platform_objs.append(obj)

# Create Target Dummy Object for Camera Look-At
target_dummy = bpy.data.objects.new("TargetCenter", None)
scene.collection.objects.link(target_dummy)
target_dummy.location = {scene_spec["target_look_at"]}

# Create Cameras
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
    
    # Track-to constraint to look at target
    track = c_obj.constraints.new(type="TRACK_TO")
    track.target = target_dummy
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    cam_objects[cam_id] = c_obj

bpy.context.view_layer.update()

# Save scene blend file
blend_path = os.path.join(artifacts_dir, "scene.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_path)

# Export Ground Truth
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
    "platform": {{
        "normal_world": {scene_spec["platform_normal"]},
        "d_offset": {scene_spec["platform_d_offset"]},
        "corners_world": {json.dumps(scene_spec["platform_corners"])},
        "width_m": {scene_spec["platform_width_m"]},
        "height_m": {scene_spec["platform_height_m"]}
    }},
    "cameras": {{}}
}}

# Compute OpenCV Extrinsics for each camera
# Change of basis from Blender camera coords (Right, Up, Back) to OpenCV coords (Right, Down, Forward)
R_b_cv = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0]
])

for cam_id, c_obj in cam_objects.items():
    # Blender camera world matrix: (4, 4)
    mw = np.array(c_obj.matrix_world)
    R_b_cam = mw[:3, :3]
    t_b_cam = mw[:3, 3]

    # Convert to OpenCV Camera Pose in World coordinates
    R_w_c = np.array(R_b_cam @ R_b_cv, dtype=float)
    t_w_c = np.array(t_b_cam, dtype=float)

    # OpenCV Extrinsics: X_c = R_c_w @ X_w + t_c_w
    R_c_w = R_w_c.T
    t_c_w = -R_c_w @ t_w_c

    # Rodrigues vector
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

    # Render Frame
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
    return script_content


def run_scene_generation():
    """Build scenes 001, 002, 003."""
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    # Scene 002 exact tilted platform
    rx, ry = math.radians(10), math.radians(-6)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    R_tilt = np.array([
        [cy,  0, sy],
        [sx*sy, cx, -sx*cy],
        [-cx*sy, sx, cx*cy]
    ])
    plat_center_002 = np.array([0.0, 0.10, 0.15])
    normal_002 = R_tilt @ np.array([0.0, 0.0, 1.0])
    normal_002 = normal_002 / np.linalg.norm(normal_002)
    d_002 = -float(np.dot(normal_002, plat_center_002))

    local_corners = {
        "2": np.array([-0.14,  0.10, 0.0]),
        "3": np.array([ 0.14,  0.10, 0.0]),
        "4": np.array([ 0.14, -0.10, 0.0]),
        "5": np.array([-0.14, -0.10, 0.0])
    }
    corners_002 = {}
    for tid_str, lpt in local_corners.items():
        world_pt = R_tilt @ lpt + plat_center_002
        corners_002[tid_str] = {
            "size_m": 0.0885,
            "location": [float(x) for x in world_pt],
            "rotation_euler": [rx, ry, 0.0]
        }

    scenes_spec = {
        "001": {
            "name": "Flat Elevated Platform (12 cm)",
            "base_tag_size_m": 0.134,
            "base_tag_location": [0.0, -0.22, 0.0],
            "platform_normal": [0.0, 0.0, 1.0],
            "platform_d_offset": -0.12,
            "platform_width_m": 0.28,
            "platform_height_m": 0.20,
            "target_look_at": [0.0, -0.05, 0.06],
            "platform_corners": {
                "2": {"size_m": 0.0885, "location": [-0.14,  0.20, 0.12], "rotation_euler": [0.0, 0.0, 0.0]},
                "3": {"size_m": 0.0885, "location": [ 0.14,  0.20, 0.12], "rotation_euler": [0.0, 0.0, 0.0]},
                "4": {"size_m": 0.0885, "location": [ 0.14,  0.00, 0.12], "rotation_euler": [0.0, 0.0, 0.0]},
                "5": {"size_m": 0.0885, "location": [-0.14,  0.00, 0.12], "rotation_euler": [0.0, 0.0, 0.0]}
            },
            "cameras": {
                "cam0": {"location": [-0.42, -0.72, 0.75]},
                "cam1": {"location": [ 0.00, -0.70, 0.88]},
                "cam2": {"location": [ 0.42, -0.72, 0.75]}
            }
        },
        "002": {
            "name": "Tilted Platform (10 deg X, -6 deg Y tilt at 15 cm)",
            "base_tag_size_m": 0.134,
            "base_tag_location": [0.0, -0.22, 0.0],
            "platform_normal": [float(x) for x in normal_002],
            "platform_d_offset": float(d_002),
            "platform_width_m": 0.28,
            "platform_height_m": 0.20,
            "target_look_at": [0.0, -0.05, 0.08],
            "platform_corners": corners_002,
            "cameras": {
                "cam0": {"location": [-0.45, -0.75, 0.80]},
                "cam1": {"location": [ 0.05, -0.72, 0.90]},
                "cam2": {"location": [ 0.45, -0.75, 0.80]}
            }
        },
        "003": {
            "name": "Challenging Oblique Multi-Camera View",
            "base_tag_size_m": 0.134,
            "base_tag_location": [0.0, -0.22, 0.0],
            "platform_normal": [0.0, 0.0, 1.0],
            "platform_d_offset": -0.10,
            "platform_width_m": 0.28,
            "platform_height_m": 0.20,
            "target_look_at": [0.0, -0.05, 0.06],
            "platform_corners": {
                "2": {"size_m": 0.0885, "location": [-0.14,  0.20, 0.10], "rotation_euler": [0.0, 0.0, 0.0]},
                "3": {"size_m": 0.0885, "location": [ 0.14,  0.20, 0.10], "rotation_euler": [0.0, 0.0, 0.0]},
                "4": {"size_m": 0.0885, "location": [ 0.14,  0.00, 0.10], "rotation_euler": [0.0, 0.0, 0.0]},
                "5": {"size_m": 0.0885, "location": [-0.14,  0.00, 0.10], "rotation_euler": [0.0, 0.0, 0.0]}
            },
            "cameras": {
                "cam0": {"location": [-0.55, -0.65, 0.65]},
                "cam1": {"location": [ 0.00, -0.75, 0.85]},
                "cam2": {"location": [ 0.55, -0.65, 0.65]}
            }
        }
    }

    for scene_id, spec in scenes_spec.items():
        sdir = EXP_DIR / scene_id
        inputs_dir = sdir / "inputs"
        artifacts_dir = sdir / "artifacts"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Write scene_spec.json
        with open(inputs_dir / "scene_spec.json", "w") as f:
            json.dump(spec, f, indent=2)

        # Generate build_scene.py
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
