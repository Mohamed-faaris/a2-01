import bpy
import math
import os
import json
import mathutils

art_dir = r'/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/10_can_mixed_fov_optical_roll/011/artifacts'
inp_dir = r'/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/10_can_mixed_fov_optical_roll/011/inputs'
marker_img = r'/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/10_can_mixed_fov_optical_roll/shared/marker_3.png'
os.makedirs(art_dir, exist_ok=True)
os.makedirs(inp_dir, exist_ok=True)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

scene = bpy.context.scene
available_engines = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in available_engines else "BLENDER_EEVEE"
scene.render.resolution_x = 1080
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.view_settings.view_transform = "Standard"

# Ambient lighting
world = bpy.data.worlds.new("World")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.75, 0.75, 0.75, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
scene.world = world

# Floor
bpy.ops.mesh.primitive_plane_add(size=12.0, location=(0, 0, 0))
floor = bpy.context.active_object
mat_f = bpy.data.materials.new("FloorMat")
mat_f.use_nodes = True
mat_f.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.35, 0.35, 0.35, 1.0)
floor.data.materials.append(mat_f)

# AprilTag Marker (Emission)
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

hm = 0.088 / 2.0
m_mesh = bpy.data.meshes.new("MarkerMesh")
m_mesh.from_pydata([(-hm, -hm, 0), (hm, -hm, 0), (hm, hm, 0), (-hm, hm, 0)], [], [(0, 1, 2, 3)])
m_mesh.update()
uv = m_mesh.uv_layers.new(name="UVMap")
for loop_idx, uv_c in zip(m_mesh.polygons[0].loop_indices, [(0, 0), (1, 0), (1, 1), (0, 1)]):
    uv.data[loop_idx].uv = uv_c

marker_obj = bpy.data.objects.new("AprilTag_Marker", m_mesh)
scene.collection.objects.link(marker_obj)
marker_obj.location = (0.0, -0.05, 0.001)
marker_obj.data.materials.append(mat_m)

# 3D Cans
cans = [{'name': 'can_red', 'color': [0.85, 0.08, 0.08], 'position': [-0.1289280824284895, 0.07794749222206057, 0.24058461052684127], 'rotation_euler': [0.055000346847271925, 0.06692522682647384, 0.0], 'radius': 0.035, 'height': 0.125, 'gt_dist_to_marker_corner_m': 0.30688679487350723}, {'name': 'can_green', 'color': [0.08, 0.85, 0.08], 'position': [-0.0123926773995576, 0.1538002306991001, 0.30187387652280107], 'rotation_euler': [-0.2991807143774023, 0.32733955149792604, 0.0], 'radius': 0.035, 'height': 0.125, 'gt_dist_to_marker_corner_m': 0.39106146160207783}, {'name': 'can_blue', 'color': [0.08, 0.2, 0.88], 'position': [0.14612879572092008, 0.05920008786986913, 0.16695696739773463], 'rotation_euler': [0.17308753362595847, 0.2413413601407915, 0.0], 'radius': 0.035, 'height': 0.125, 'gt_dist_to_marker_corner_m': 0.2952303184184777}]
for c in cans:
    pos = c["position"]
    radius = c["radius"]
    height = c["height"]
    name = c["name"]
    color = c["color"]
    rot = c.get("rotation_euler", [0, 0, 0])
    
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

# Dual Lights
for lx, ly, lz in [(2.0, -2.0, 4.0), (-2.0, 2.0, 4.0)]:
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_data.energy = 3.5
    light_obj = bpy.data.objects.new("Sun", light_data)
    scene.collection.objects.link(light_obj)
    light_obj.location = (lx, ly, lz)

# Target point: centroid of all items
all_pts = [c["position"] for c in cans] + [(0.0, -0.05, 0.001)]
avg_x = sum(p[0] for p in all_pts) / len(all_pts)
avg_y = sum(p[1] for p in all_pts) / len(all_pts)
avg_z = sum(p[2] for p in all_pts) / len(all_pts)
target_loc = mathutils.Vector((avg_x, avg_y, avg_z))

def create_rolled_cam(name, loc, focal_px, roll_deg):
    c_data = bpy.data.cameras.new(name)
    c_data.lens = focal_px * 36.0 / 1080
    c_data.sensor_width = 36.0
    c_data.sensor_fit = "HORIZONTAL"
    c_data.clip_start = 0.01
    c_data.clip_end = 100.0
    c_obj = bpy.data.objects.new(name, c_data)
    scene.collection.objects.link(c_obj)
    c_obj.location = loc
    
    # Base tracking orientation towards workspace
    dir_vec = target_loc - mathutils.Vector(loc)
    q_base = dir_vec.to_track_quat('-Z', 'Y')
    
    # Rotate around optical Z axis (in-plane camera roll)
    q_roll = mathutils.Quaternion((0, 0, 1), math.radians(roll_deg))
    q_final = q_base @ q_roll
    c_obj.rotation_euler = q_final.to_euler()
    return c_obj

cam0 = create_rolled_cam("Camera_0", (-0.48108234880384176, -0.8450979128864046, 0.5495564617165036), 703.741701334251, -5.513251388274291)
cam1 = create_rolled_cam("Camera_1", (0.5999494884081957, -1.0093993486984472, 0.6178527481592333), 1158.0337370751618, 24.97105650960161)

bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(art_dir, "scene.blend"))

scene.camera = cam0
scene.render.filepath = os.path.join(inp_dir, "cam0_render.png")
bpy.ops.render.render(write_still=True)

scene.camera = cam1
scene.render.filepath = os.path.join(inp_dir, "cam1_render.png")
bpy.ops.render.render(write_still=True)

with open(os.path.join(inp_dir, "ground_truth_cameras.json"), "w") as f:
    json.dump({
        "cam0_matrix_world": [list(row) for row in cam0.matrix_world],
        "cam1_matrix_world": [list(row) for row in cam1.matrix_world],
        "cam0_position": list(cam0.location),
        "cam1_position": list(cam1.location),
        "cam0_fov_deg": 75.0,
        "cam1_fov_deg": 50.0,
        "cam0_roll_deg": -5.513251388274291,
        "cam1_roll_deg": 24.97105650960161,
    }, f, indent=2)
