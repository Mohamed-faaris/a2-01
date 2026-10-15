#!/usr/bin/env python3
"""Generator and benchmark runner for Experiment 14:
Pringles Can 3D Cylinder Construction and Localization via RealSense Metric Depth & AprilTag Origin.

Models: Pringles cans (5 flavours: Original, BBQ, Extra Hot, French Onion, Paprika)
Inputs:
  - RGB image (cam0_render.png)
  - Metric Depth pass (depth_pass.exr, depth_map.npy, depth_vis.png, depth_realsense_16bit.png)
  - AprilTag marker & dimensions (defines world/table coordinate frame)
  - Can dimensions (radius=0.0318m, height=0.2695m)
Pipelines & Uncertainty Analysis:
  1. Depth Pointcloud Cylinder Fitting (RANSAC + Table Geometry + Radius Compensation)
  2. Exp-13 SIFT / LightGlue 2D Transfer + Epipolar Depth Backprojection
  3. RealSense Direct Surface Depth Backprojection (with & without radius compensation)
  4. Physical uncertainty quantification: Covariance ellipsoid, focal length drift sensitivity, tag pose uncertainty
"""

import os
import sys
import json
import math
import subprocess
import argparse
from pathlib import Path
import numpy as np

WIDTH = 1280
HEIGHT = 720
FOCAL_PX = 925.09  # RealSense D415 FOV ~69 deg
CX = 640.0
CY = 360.0

CAN_RADIUS_M = 0.0318  # 3.18 cm
CAN_HEIGHT_M = 0.2695  # 26.95 cm
TAG_SIZE_M = 0.088
TAG_ID = 1

GLB_PATH = "/home/mfk01/Downloads/pringles_cans__5_different_flavours__gameready.glb"
EXP_ROOT = Path("/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/14_pringles_cylinder_depth_realsense")

# 10 Experiment configurations:
# 1-3: 1 can (including rolling)
# 4-5: 3 cans (including rolling)
# 6-10: 5 to 10 cans (varied flavours, standing and rolling)
SCENE_CONFIGS = [
    {
        "id": "001",
        "desc": "Single upright can (Original Red)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [0.0, -0.85, 0.45],
        "cam_target": [0.0, 0.15, 0.10],
        "cans": [
            {"name": "can_original", "flavor": "Original", "state": "standing", "pos": [0.08, 0.22, CAN_HEIGHT_M / 2.0], "rot_z_deg": 15.0}
        ]
    },
    {
        "id": "002",
        "desc": "Single rolling can (BBQ Purple, rolling along X axis)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [0.0, -0.85, 0.45],
        "cam_target": [0.0, 0.18, 0.05],
        "cans": [
            {"name": "can_bbq", "flavor": "BBQ", "state": "rolling", "pos": [0.05, 0.20, CAN_RADIUS_M], "roll_axis_deg": 0.0} # axis along X
        ]
    },
    {
        "id": "003",
        "desc": "Single rolling can (French Onion Green, rolling at 45 deg angle)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [0.05, -0.88, 0.48],
        "cam_target": [0.0, 0.20, 0.05],
        "cans": [
            {"name": "can_french_onion", "flavor": "French_Onion", "state": "rolling", "pos": [-0.06, 0.24, CAN_RADIUS_M], "roll_axis_deg": 45.0}
        ]
    },
    {
        "id": "004",
        "desc": "3 cans (2 standing, 1 rolling - Original, Paprika, Extra Hot)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [0.0, -0.92, 0.50],
        "cam_target": [0.0, 0.25, 0.12],
        "cans": [
            {"name": "can_original", "flavor": "Original", "state": "standing", "pos": [-0.18, 0.28, CAN_HEIGHT_M / 2.0], "rot_z_deg": -10.0},
            {"name": "can_paprika", "flavor": "Paprika", "state": "standing", "pos": [0.18, 0.26, CAN_HEIGHT_M / 2.0], "rot_z_deg": 40.0},
            {"name": "can_extra_hot", "flavor": "Extra_Hot", "state": "rolling", "pos": [0.02, 0.18, CAN_RADIUS_M], "roll_axis_deg": 15.0}
        ]
    },
    {
        "id": "005",
        "desc": "3 cans (2 standing, 1 rolling perpendicular - BBQ, French Onion, Original)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [-0.10, -0.90, 0.52],
        "cam_target": [0.0, 0.24, 0.10],
        "cans": [
            {"name": "can_bbq", "flavor": "BBQ", "state": "standing", "pos": [-0.15, 0.22, CAN_HEIGHT_M / 2.0], "rot_z_deg": 60.0},
            {"name": "can_french_onion", "flavor": "French_Onion", "state": "standing", "pos": [0.15, 0.32, CAN_HEIGHT_M / 2.0], "rot_z_deg": -30.0},
            {"name": "can_original", "flavor": "Original", "state": "rolling", "pos": [-0.02, 0.35, CAN_RADIUS_M], "roll_axis_deg": 90.0} # axis along Y
        ]
    },
    {
        "id": "006",
        "desc": "5 cans all 5 flavors (3 standing, 2 rolling)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [0.0, -1.05, 0.60],
        "cam_target": [0.0, 0.30, 0.12],
        "cans": [
            {"name": "can_original", "flavor": "Original", "state": "standing", "pos": [-0.22, 0.30, CAN_HEIGHT_M / 2.0], "rot_z_deg": 0.0},
            {"name": "can_bbq", "flavor": "BBQ", "state": "standing", "pos": [0.0, 0.42, CAN_HEIGHT_M / 2.0], "rot_z_deg": 45.0},
            {"name": "can_french_onion", "flavor": "French_Onion", "state": "standing", "pos": [0.22, 0.32, CAN_HEIGHT_M / 2.0], "rot_z_deg": -45.0},
            {"name": "can_extra_hot", "flavor": "Extra_Hot", "state": "rolling", "pos": [-0.12, 0.18, CAN_RADIUS_M], "roll_axis_deg": -25.0},
            {"name": "can_paprika", "flavor": "Paprika", "state": "rolling", "pos": [0.14, 0.20, CAN_RADIUS_M], "roll_axis_deg": 30.0}
        ]
    },
    {
        "id": "007",
        "desc": "6 cans cluster (4 standing, 2 rolling)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [0.15, -1.10, 0.62],
        "cam_target": [0.0, 0.32, 0.12],
        "cans": [
            {"name": "can_original", "flavor": "Original", "state": "standing", "pos": [-0.25, 0.28, CAN_HEIGHT_M / 2.0], "rot_z_deg": 10.0},
            {"name": "can_bbq", "flavor": "BBQ", "state": "standing", "pos": [-0.08, 0.40, CAN_HEIGHT_M / 2.0], "rot_z_deg": -20.0},
            {"name": "can_paprika", "flavor": "Paprika", "state": "standing", "pos": [0.10, 0.42, CAN_HEIGHT_M / 2.0], "rot_z_deg": 75.0},
            {"name": "can_french_onion", "flavor": "French_Onion", "state": "standing", "pos": [0.26, 0.30, CAN_HEIGHT_M / 2.0], "rot_z_deg": -15.0},
            {"name": "can_extra_hot", "flavor": "Extra_Hot", "state": "rolling", "pos": [-0.15, 0.18, CAN_RADIUS_M], "roll_axis_deg": 10.0},
            {"name": "can_orig_roll", "flavor": "Original", "state": "rolling", "pos": [0.15, 0.19, CAN_RADIUS_M], "roll_axis_deg": -60.0}
        ]
    },
    {
        "id": "008",
        "desc": "7 cans spread across table (5 standing, 2 rolling)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [-0.10, -1.15, 0.65],
        "cam_target": [0.0, 0.35, 0.12],
        "cans": [
            {"name": "can_1", "flavor": "Original", "state": "standing", "pos": [-0.30, 0.32, CAN_HEIGHT_M / 2.0], "rot_z_deg": 0.0},
            {"name": "can_2", "flavor": "BBQ", "state": "standing", "pos": [-0.16, 0.45, CAN_HEIGHT_M / 2.0], "rot_z_deg": 35.0},
            {"name": "can_3", "flavor": "Extra_Hot", "state": "standing", "pos": [0.0, 0.48, CAN_HEIGHT_M / 2.0], "rot_z_deg": -50.0},
            {"name": "can_4", "flavor": "French_Onion", "state": "standing", "pos": [0.18, 0.42, CAN_HEIGHT_M / 2.0], "rot_z_deg": 20.0},
            {"name": "can_5", "flavor": "Paprika", "state": "standing", "pos": [0.32, 0.30, CAN_HEIGHT_M / 2.0], "rot_z_deg": -40.0},
            {"name": "can_6", "flavor": "Original", "state": "rolling", "pos": [-0.18, 0.18, CAN_RADIUS_M], "roll_axis_deg": 40.0},
            {"name": "can_7", "flavor": "BBQ", "state": "rolling", "pos": [0.16, 0.18, CAN_RADIUS_M], "roll_axis_deg": -30.0}
        ]
    },
    {
        "id": "009",
        "desc": "8 cans (4 standing, 4 rolling across various directions)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [0.0, -1.20, 0.70],
        "cam_target": [0.0, 0.38, 0.10],
        "cans": [
            {"name": "can_1", "flavor": "Original", "state": "standing", "pos": [-0.28, 0.40, CAN_HEIGHT_M / 2.0], "rot_z_deg": 10.0},
            {"name": "can_2", "flavor": "Extra_Hot", "state": "standing", "pos": [-0.10, 0.52, CAN_HEIGHT_M / 2.0], "rot_z_deg": -45.0},
            {"name": "can_3", "flavor": "French_Onion", "state": "standing", "pos": [0.10, 0.52, CAN_HEIGHT_M / 2.0], "rot_z_deg": 30.0},
            {"name": "can_4", "flavor": "Paprika", "state": "standing", "pos": [0.28, 0.42, CAN_HEIGHT_M / 2.0], "rot_z_deg": -15.0},
            {"name": "can_5", "flavor": "BBQ", "state": "rolling", "pos": [-0.26, 0.20, CAN_RADIUS_M], "roll_axis_deg": 20.0},
            {"name": "can_6", "flavor": "Original", "state": "rolling", "pos": [-0.08, 0.28, CAN_RADIUS_M], "roll_axis_deg": 80.0},
            {"name": "can_7", "flavor": "French_Onion", "state": "rolling", "pos": [0.12, 0.26, CAN_RADIUS_M], "roll_axis_deg": -70.0},
            {"name": "can_8", "flavor": "Extra_Hot", "state": "rolling", "pos": [0.26, 0.18, CAN_RADIUS_M], "roll_axis_deg": -15.0}
        ]
    },
    {
        "id": "010",
        "desc": "10 cans dense tabletop layout (6 standing, 4 rolling)",
        "marker_pos": [0.0, 0.0, 0.001],
        "cam_pos": [0.05, -1.25, 0.75],
        "cam_target": [0.0, 0.40, 0.12],
        "cans": [
            {"name": "can_1", "flavor": "Original", "state": "standing", "pos": [-0.34, 0.42, CAN_HEIGHT_M / 2.0], "rot_z_deg": 25.0},
            {"name": "can_2", "flavor": "BBQ", "state": "standing", "pos": [-0.20, 0.54, CAN_HEIGHT_M / 2.0], "rot_z_deg": -10.0},
            {"name": "can_3", "flavor": "Extra_Hot", "state": "standing", "pos": [-0.02, 0.58, CAN_HEIGHT_M / 2.0], "rot_z_deg": 65.0},
            {"name": "can_4", "flavor": "French_Onion", "state": "standing", "pos": [0.18, 0.54, CAN_HEIGHT_M / 2.0], "rot_z_deg": -35.0},
            {"name": "can_5", "flavor": "Paprika", "state": "standing", "pos": [0.34, 0.42, CAN_HEIGHT_M / 2.0], "rot_z_deg": 40.0},
            {"name": "can_6", "flavor": "Original_2", "flavor_model": "Original", "state": "standing", "pos": [0.0, 0.36, CAN_HEIGHT_M / 2.0], "rot_z_deg": -80.0},
            {"name": "can_7", "flavor": "BBQ", "state": "rolling", "pos": [-0.28, 0.22, CAN_RADIUS_M], "roll_axis_deg": 15.0},
            {"name": "can_8", "flavor": "French_Onion", "state": "rolling", "pos": [-0.12, 0.16, CAN_RADIUS_M], "roll_axis_deg": -45.0},
            {"name": "can_9", "flavor": "Paprika", "state": "rolling", "pos": [0.12, 0.16, CAN_RADIUS_M], "roll_axis_deg": 45.0},
            {"name": "can_10", "flavor": "Extra_Hot", "state": "rolling", "pos": [0.28, 0.22, CAN_RADIUS_M], "roll_axis_deg": -20.0}
        ]
    }
]

def make_blender_scene_script(exp_id: str, cfg: dict, art_dir: Path, inp_dir: Path) -> str:
    escaped_art = str(art_dir.resolve()).replace("\\", "\\\\").replace("'", "\\'")
    escaped_inp = str(inp_dir.resolve()).replace("\\", "\\\\").replace("'", "\\'")
    escaped_glb = str(Path(GLB_PATH).resolve()).replace("\\", "\\\\").replace("'", "\\'")
    marker_png = str((EXP_ROOT / "shared" / "marker_1.png").resolve()).replace("\\", "\\\\").replace("'", "\\'")
    cfg_json = json.dumps(cfg)

    script = f'''import bpy
import math
import os
import json
import mathutils
import numpy as np

art_dir = r'{escaped_art}'
inp_dir = r'{escaped_inp}'
glb_path = r'{escaped_glb}'
marker_png = r'{marker_png}'
cfg = json.loads(r\'\'\'{cfg_json}\'\'\')

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
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.85, 0.85, 0.85, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
scene.world = world

# Directional key and fill lights
sun1 = bpy.data.lights.new("Sun1", 'SUN')
sun1.energy = 3.5
sun1_obj = bpy.data.objects.new("Sun1", sun1)
scene.collection.objects.link(sun1_obj)
sun1_obj.location = (1.5, -2.0, 3.0)
sun1_obj.rotation_euler = (math.radians(45), math.radians(20), math.radians(35))

sun2 = bpy.data.lights.new("Sun2", 'SUN')
sun2.energy = 1.5
sun2_obj = bpy.data.objects.new("Sun2", sun2)
scene.collection.objects.link(sun2_obj)
sun2_obj.location = (-1.5, -1.0, 2.5)
sun2_obj.rotation_euler = (math.radians(40), math.radians(-25), math.radians(-30))

# Table surface (wood/gray tabletop)
bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0, 0, 0))
table = bpy.context.active_object
table.name = "TableSurface"
table_mat = bpy.data.materials.new("TableMat")
table_mat.use_nodes = True
table_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.28, 0.28, 0.30, 1.0)
table_mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.45
table.data.materials.append(table_mat)

# AprilTag Marker (tag36h11 id=1, 0.088m size)
tag_size = {TAG_SIZE_M}
hm = tag_size / 2.0
m_mesh = bpy.data.meshes.new("MarkerMesh")
m_mesh.from_pydata([(-hm, -hm, 0), (hm, -hm, 0), (hm, hm, 0), (-hm, hm, 0)], [], [(0, 1, 2, 3)])
m_mesh.update()
uv = m_mesh.uv_layers.new(name="UVMap")
for loop_idx, uv_c in zip(m_mesh.polygons[0].loop_indices, [(0, 0), (1, 0), (1, 1), (0, 1)]):
    uv.data[loop_idx].uv = uv_c

m_img = bpy.data.images.load(marker_png, check_existing=True)
m_img.colorspace_settings.name = "Non-Color"
mat_m = bpy.data.materials.new("MarkerMat")
mat_m.use_nodes = True
m_nodes = mat_m.node_tree.nodes
m_nodes.clear()
tex = m_nodes.new("ShaderNodeTexImage")
tex.image = m_img
tex.interpolation = "Closest"
emission = m_nodes.new("ShaderNodeEmission")
out_node = m_nodes.new("ShaderNodeOutputMaterial")
mat_m.node_tree.links.new(tex.outputs["Color"], emission.inputs["Color"])
mat_m.node_tree.links.new(emission.outputs["Emission"], out_node.inputs["Surface"])

marker_obj = bpy.data.objects.new("AprilTag_Marker", m_mesh)
scene.collection.objects.link(marker_obj)
m_pos = cfg["marker_pos"]
marker_obj.location = (m_pos[0], m_pos[1], m_pos[2])
marker_obj.data.materials.append(mat_m)

# Import GLB models
bpy.ops.import_scene.gltf(filepath=glb_path)

# Dictionary of raw flavor meshes
flavor_mesh_map = {{
    "Original": ("Original_Pringels_0", "Original_Transparent.001_0"),
    "BBQ": ("BBQ_Pringels_BBQ_0", "BBQ_Transparent_0"),
    "Extra_Hot": ("Extra_Hot_Pringels_Extra_Hot_0", "Extra_Hot_Transparent_0"),
    "French_Onion": ("French_Onion_Pringels_French_Onion_0", "French_Onion_Transparent_0"),
    "Paprika": ("Paprika_Pringels_Paprika_0", "Paprika_Transparent_0"),
}}

# Normalize each master flavor model to world origin & scale
master_cans = {{}}
for flv, (body_name, cap_name) in flavor_mesh_map.items():
    body = bpy.data.objects.get(body_name)
    cap = bpy.data.objects.get(cap_name)
    if not body:
        continue
    # Clear parent keeping transform
    for o in (body, cap):
        if o:
            o.select_set(True)
            bpy.context.view_layer.objects.active = o
            bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    if cap:
        body.select_set(True)
        cap.select_set(True)
        bpy.context.view_layer.objects.active = body
        bpy.ops.object.join()
    # Apply transform so coordinates are in meters with base at z=0, center (0,0)
    body.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # Hide master from render
    body.hide_render = True
    master_cans[flv] = body

# Remove initial empties
for o in list(bpy.context.scene.objects):
    if o.type == 'EMPTY':
        bpy.data.objects.remove(o, do_unlink=True)

# Instantiate cans according to config
instantiated_cans = []
can_gt_metadata = []

for c_info in cfg["cans"]:
    flv = c_info.get("flavor_model", c_info["flavor"])
    master = master_cans.get(flv, master_cans["Original"])
    # Duplicate mesh
    can_obj = master.copy()
    can_obj.data = master.data.copy()
    scene.collection.objects.link(can_obj)
    can_obj.hide_render = False
    can_obj.name = c_info["name"]

    state = c_info["state"]
    pos = c_info["pos"]

    if state == "standing":
        # Upright: base on table (z=0), center at (x, y, H/2)
        # In GLB, vertices are already z in [0, 0.27], centered at (0, 0)
        rz = math.radians(c_info.get("rot_z_deg", 0.0))
        can_obj.location = (pos[0], pos[1], 0.0)
        can_obj.rotation_euler = (0, 0, rz)
        center_world = [pos[0], pos[1], {CAN_HEIGHT_M} / 2.0]
        axis_world = [0.0, 0.0, 1.0]
    elif state == "rolling":
        # Rolling: lying horizontal on table.
        # Cylinder axis lies along roll_axis in table plane.
        # Bottom of cylinder touches table at z=0 -> center axis at z = CAN_RADIUS_M
        roll_angle_rad = math.radians(c_info.get("roll_axis_deg", 0.0))
        # Rotate by 90 deg around Y so cylinder axis is along X, then rotate around Z
        can_obj.rotation_mode = 'XYZ'
        can_obj.rotation_euler = (0, math.radians(90), roll_angle_rad)
        # Shift so center of can is at pos
        # In master, z was [0, H]. Rotated around Y by 90 deg, the axis extends along X in [-H/2, H/2]
        # We offset location so center aligns with pos
        dx = -({CAN_HEIGHT_M} / 2.0) * math.cos(roll_angle_rad)
        dy = -({CAN_HEIGHT_M} / 2.0) * math.sin(roll_angle_rad)
        can_obj.location = (pos[0] + dx, pos[1] + dy, {CAN_RADIUS_M})
        center_world = [pos[0], pos[1], {CAN_RADIUS_M}]
        axis_world = [float(math.cos(roll_angle_rad)), float(math.sin(roll_angle_rad)), 0.0]

    bpy.context.view_layer.update()
    
    can_gt_metadata.append({{
        "name": c_info["name"],
        "flavor": c_info["flavor"],
        "state": state,
        "center_world_m": center_world,
        "axis_world": axis_world,
        "radius_m": {CAN_RADIUS_M},
        "height_m": {CAN_HEIGHT_M},
        "matrix_world": [list(row) for row in can_obj.matrix_world]
    }})

# Setup Camera
cam_pos = cfg["cam_pos"]
cam_target = cfg["cam_target"]
cam_data = bpy.data.cameras.new("Camera_0")
cam_data.lens = {FOCAL_PX} * 36.0 / {WIDTH}
cam_data.sensor_width = 36.0
cam_data.sensor_fit = "HORIZONTAL"
cam_data.clip_start = 0.01
cam_data.clip_end = 50.0

cam_obj = bpy.data.objects.new("Camera_0", cam_data)
scene.collection.objects.link(cam_obj)
cam_obj.location = tuple(cam_pos)
dir_vec = mathutils.Vector(cam_target) - mathutils.Vector(cam_pos)
cam_obj.rotation_euler = dir_vec.to_track_quat('-Z', 'Y').to_euler()
scene.camera = cam_obj

# Setup Compositor for 32-bit float Metric Depth pass EXR
scene.view_layers['ViewLayer'].use_pass_z = True
tree = bpy.data.node_groups.new("Compositing", "CompositorNodeTree")
scene.compositing_node_group = tree
rl = tree.nodes.new("CompositorNodeRLayers")

fo = tree.nodes.new("CompositorNodeOutputFile")
fo.directory = inp_dir
fo.file_name = "depth_pass"
fo.file_output_items.clear()
item = fo.file_output_items.new('FLOAT', 'Z')
item.format.file_format = 'OPEN_EXR'
item.format.color_depth = '32'
tree.links.new(rl.outputs['Depth'], fo.inputs['Z'])

bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(art_dir, "scene.blend"))

# Render RGB and Depth Pass
scene.render.filepath = os.path.join(inp_dir, "cam0_render.png")
bpy.ops.render.render(write_still=True)

# Export ground truth cameras & objects
# Compute OpenCV camera extrinsics: R_w2c, t_w2c
M_cam = np.array(cam_obj.matrix_world)
R_b = M_cam[:3, :3]
t_b = M_cam[:3, 3]
R_cv2b = np.diag([1.0, -1.0, -1.0])
R_w2c = R_cv2b @ R_b.T
t_w2c = -R_w2c @ t_b

# AprilTag corner coords in world:
# center is at m_pos, orientation is flat on table (Z=0)
tag_corners_world = [
    [m_pos[0] - hm, m_pos[1] - hm, m_pos[2]],
    [m_pos[0] + hm, m_pos[1] - hm, m_pos[2]],
    [m_pos[0] + hm, m_pos[1] + hm, m_pos[2]],
    [m_pos[0] - hm, m_pos[1] + hm, m_pos[2]]
]

gt_doc = {{
    "experiment_id": "{exp_id}",
    "description": cfg["desc"],
    "intrinsics": {{
        "width": {WIDTH},
        "height": {HEIGHT},
        "fx": {FOCAL_PX},
        "fy": {FOCAL_PX},
        "cx": {CX},
        "cy": {CY},
        "camera_matrix": [
            [{FOCAL_PX}, 0.0, {CX}],
            [0.0, {FOCAL_PX}, {CY}],
            [0.0, 0.0, 1.0]
        ]
    }},
    "camera": {{
        "position_world": list(cam_obj.location),
        "rotation_euler": [float(v) for v in cam_obj.rotation_euler],
        "matrix_world": [list(row) for row in cam_obj.matrix_world],
        "R_w2c": [list(row) for row in R_w2c],
        "t_w2c": list(t_w2c)
    }},
    "apriltag": {{
        "tag_id": {TAG_ID},
        "tag_size_m": {TAG_SIZE_M},
        "center_world": m_pos,
        "corners_world": tag_corners_world
    }},
    "cans": can_gt_metadata
}}

with open(os.path.join(inp_dir, "ground_truth.json"), "w") as f:
    json.dump(gt_doc, f, indent=2)

print(f"Scene {exp_id} build and render complete!")
'''
    return script

def build_scene(exp_id: str, cfg: dict):
    exp_dir = EXP_ROOT / exp_id
    art_dir = exp_dir / "artifacts"
    inp_dir = exp_dir / "inputs"
    art_dir.mkdir(parents=True, exist_ok=True)
    inp_dir.mkdir(parents=True, exist_ok=True)

    script_path = art_dir / "build_scene.py"
    script_content = make_blender_scene_script(exp_id, cfg, art_dir, inp_dir)
    script_path.write_text(script_content, encoding="utf-8")

    cmd = ["/snap/bin/blender", "-b", "--python", str(script_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error rendering scene {exp_id}:\n{res.stderr}\n{res.stdout}")
        sys.exit(1)

    # Post-process depth pass: extract NPY and 16-bit PNG
    depth_exr = inp_dir / "depth_pass.exr"
    if depth_exr.exists():
        import pyexr
        f = pyexr.open(str(depth_exr))
        z = f.get('Z.V')[:, :, 0]
        # Replace inf with 0
        z[z > 1e6] = 0.0
        np.save(str(inp_dir / "depth_map.npy"), z.astype(np.float32))
        
        # Save RealSense 16-bit uint (millimeters)
        z_mm = np.clip(z * 1000.0, 0, 65535).astype(np.uint16)
        import cv2
        cv2.imwrite(str(inp_dir / "depth_realsense_16bit.png"), z_mm)

        # Depth colormap visualization
        valid = z > 0
        vis = np.zeros_like(z, dtype=np.uint8)
        if valid.any():
            min_d, max_d = z[valid].min(), z[valid].max()
            vis[valid] = ((z[valid] - min_d) / (max_d - min_d + 1e-6) * 255).astype(np.uint8)
        cv2.imwrite(str(inp_dir / "depth_vis.png"), cv2.applyColorMap(vis, cv2.COLORMAP_TURBO))
        print(f"[{exp_id}] Rendered & extracted metric depth map ({WIDTH}x{HEIGHT}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="all")
    args = parser.parse_args()

    scenes_to_run = SCENE_CONFIGS if args.scene == "all" else [c for c in SCENE_CONFIGS if c["id"] == args.scene]
    for cfg in scenes_to_run:
        print(f"Building scene {cfg['id']}: {cfg['desc']}...")
        build_scene(cfg["id"], cfg)
