import bpy
import math
import os
import json
import mathutils
import numpy as np

art_dir = r'/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/15_pringles_isolated_cylinder_boxernet/017/artifacts'
inp_dir = r'/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/15_pringles_isolated_cylinder_boxernet/017/inputs'
marker_png = r'/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/15_pringles_isolated_cylinder_boxernet/shared/marker_1.png'
cfg = json.loads(r'''{"id": "017", "desc": "9 cans large array (6 standing, 3 rolling)", "marker_pos": [0.0, 0.0, 0.001], "cam_pos": [0.18, -1.22, 0.72], "cam_target": [0.0, 0.38, 0.12], "cans": [{"name": "can_1", "flavor": "Original", "state": "standing", "pos": [-0.3, 0.36, 0.13475], "rot_z_deg": 20.0}, {"name": "can_2", "flavor": "BBQ", "state": "standing", "pos": [-0.14, 0.5, 0.13475], "rot_z_deg": -35.0}, {"name": "can_3", "flavor": "Extra_Hot", "state": "standing", "pos": [0.02, 0.54, 0.13475], "rot_z_deg": 70.0}, {"name": "can_4", "flavor": "French_Onion", "state": "standing", "pos": [0.18, 0.5, 0.13475], "rot_z_deg": -20.0}, {"name": "can_5", "flavor": "Paprika", "state": "standing", "pos": [0.32, 0.38, 0.13475], "rot_z_deg": 45.0}, {"name": "can_6", "flavor": "Original", "state": "standing", "pos": [0.0, 0.36, 0.13475], "rot_z_deg": -60.0}, {"name": "can_7", "flavor": "BBQ", "state": "rolling", "pos": [-0.22, 0.2, 0.0318], "roll_axis_deg": 25.0}, {"name": "can_8", "flavor": "French_Onion", "state": "rolling", "pos": [-0.02, 0.22, 0.0318], "roll_axis_deg": -30.0}, {"name": "can_9", "flavor": "Extra_Hot", "state": "rolling", "pos": [0.2, 0.18, 0.0318], "roll_axis_deg": 45.0}]}''')
flavor_files = json.loads(r'''{"Original": "/home/mfk01/Projects/econ-systems/demos/a2-01/assets/pringles_isolated/pringles_original.glb", "BBQ": "/home/mfk01/Projects/econ-systems/demos/a2-01/assets/pringles_isolated/pringles_bbq.glb", "Extra_Hot": "/home/mfk01/Projects/econ-systems/demos/a2-01/assets/pringles_isolated/pringles_extra_hot.glb", "French_Onion": "/home/mfk01/Projects/econ-systems/demos/a2-01/assets/pringles_isolated/pringles_french_onion.glb", "Paprika": "/home/mfk01/Projects/econ-systems/demos/a2-01/assets/pringles_isolated/pringles_paprika.glb"}''')

os.makedirs(art_dir, exist_ok=True)
os.makedirs(inp_dir, exist_ok=True)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

scene = bpy.context.scene
available_engines = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in available_engines else "BLENDER_EEVEE"
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.view_settings.view_transform = "Standard"

# Ambient world lighting
world = bpy.data.worlds.new("World")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.85, 0.85, 0.85, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
scene.world = world

# Directional lights
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

# Table surface
bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0, 0, 0))
table = bpy.context.active_object
table.name = "TableSurface"
table_mat = bpy.data.materials.new("TableMat")
table_mat.use_nodes = True
table_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.28, 0.28, 0.30, 1.0)
table_mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.45
table.data.materials.append(table_mat)

# AprilTag Marker (tag36h11 id=1, 0.088m size)
tag_size = 0.088
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
marker_obj.location = tuple(m_pos)
marker_obj.data.materials.append(mat_m)

# Import isolated standalone models for each can
can_gt_metadata = []

for c_info in cfg["cans"]:
    flv = c_info["flavor"]
    model_file = flavor_files.get(flv, flavor_files["Original"])
    
    # Import isolated GLB
    bpy.ops.import_scene.gltf(filepath=model_file)
    # The newly imported object is the active/selected one
    imported = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    can_obj = imported[0]
    can_obj.name = c_info["name"]

    state = c_info["state"]
    pos = c_info["pos"]

    if state == "standing":
        rz = math.radians(c_info.get("rot_z_deg", 0.0))
        can_obj.location = (pos[0], pos[1], 0.0)
        can_obj.rotation_euler = (0, 0, rz)
        center_world = [pos[0], pos[1], 0.2695 / 2.0]
        axis_world = [0.0, 0.0, 1.0]
    elif state == "rolling":
        roll_angle_rad = math.radians(c_info.get("roll_axis_deg", 0.0))
        can_obj.rotation_mode = 'XYZ'
        can_obj.rotation_euler = (0, math.radians(90), roll_angle_rad)
        dx = -(0.2695 / 2.0) * math.cos(roll_angle_rad)
        dy = -(0.2695 / 2.0) * math.sin(roll_angle_rad)
        can_obj.location = (pos[0] + dx, pos[1] + dy, 0.0318)
        center_world = [pos[0], pos[1], 0.0318]
        axis_world = [float(math.cos(roll_angle_rad)), float(math.sin(roll_angle_rad)), 0.0]

    bpy.context.view_layer.update()
    
    can_gt_metadata.append({
        "name": c_info["name"],
        "flavor": flv,
        "model_file": model_file,
        "state": state,
        "center_world_m": center_world,
        "axis_world": axis_world,
        "radius_m": 0.0318,
        "height_m": 0.2695,
        "matrix_world": [list(row) for row in can_obj.matrix_world]
    })

# Setup Camera
cam_pos = cfg["cam_pos"]
cam_target = cfg["cam_target"]
cam_data = bpy.data.cameras.new("Camera_0")
cam_data.lens = 925.09 * 36.0 / 1280
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

# Compositor for Metric Depth Pass (32-bit float EXR)
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

# Render RGB & Depth
scene.render.filepath = os.path.join(inp_dir, "cam0_render.png")
bpy.ops.render.render(write_still=True)

# Export ground truth cameras & objects
M_cam = np.array(cam_obj.matrix_world)
R_b = M_cam[:3, :3]
t_b = M_cam[:3, 3]
R_cv2b = np.diag([1.0, -1.0, -1.0])
R_w2c = R_cv2b @ R_b.T
t_w2c = -R_w2c @ t_b

tag_corners_world = [
    [m_pos[0] - hm, m_pos[1] - hm, m_pos[2]],
    [m_pos[0] + hm, m_pos[1] - hm, m_pos[2]],
    [m_pos[0] + hm, m_pos[1] + hm, m_pos[2]],
    [m_pos[0] - hm, m_pos[1] + hm, m_pos[2]]
]

gt_doc = {
    "experiment_id": "017",
    "description": cfg["desc"],
    "intrinsics": {
        "width": 1280,
        "height": 720,
        "fx": 925.09,
        "fy": 925.09,
        "cx": 640.0,
        "cy": 360.0,
        "camera_matrix": [
            [925.09, 0.0, 640.0],
            [0.0, 925.09, 360.0],
            [0.0, 0.0, 1.0]
        ]
    },
    "camera": {
        "position_world": list(cam_obj.location),
        "rotation_euler": [float(v) for v in cam_obj.rotation_euler],
        "matrix_world": [list(row) for row in cam_obj.matrix_world],
        "R_w2c": [list(row) for row in R_w2c],
        "t_w2c": list(t_w2c)
    },
    "apriltag": {
        "tag_id": 1,
        "tag_size_m": 0.088,
        "center_world": m_pos,
        "corners_world": tag_corners_world
    },
    "cans": can_gt_metadata
}

with open(os.path.join(inp_dir, "ground_truth.json"), "w") as f:
    json.dump(gt_doc, f, indent=2)

print(f"Scene 017 build and render complete!")
