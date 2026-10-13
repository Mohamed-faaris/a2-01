import bpy
import math
import os
import json
import mathutils

art_dir = r'/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/11_can_textured_background/005/artifacts'
inp_dir = r'/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/11_can_textured_background/005/inputs'
marker_img = r'/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/11_can_textured_background/shared/marker_3.png'
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

# 4. Target Cans
cans = [{'name': 'can_red', 'color': [0.85, 0.08, 0.08], 'position': [-0.19142000244742158, 0.15689704170771765, 0.0625], 'radius': 0.035, 'height': 0.125, 'gt_dist_to_marker_corner_m': 0.2974293742374515}, {'name': 'can_green', 'color': [0.08, 0.85, 0.08], 'position': [0.07698360069268265, 0.1384930781228429, 0.0625], 'radius': 0.035, 'height': 0.125, 'gt_dist_to_marker_corner_m': 0.269206822000485}, {'name': 'can_blue', 'color': [0.08, 0.2, 0.88], 'position': [0.24822802072604205, 0.06236342050892282, 0.0625], 'radius': 0.035, 'height': 0.125, 'gt_dist_to_marker_corner_m': 0.33708898731745934}]
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

all_pts = [c["position"] for c in cans] + [(0.0, -0.05, 0.001)]
avg_x = sum(p[0] for p in all_pts) / len(all_pts)
avg_y = sum(p[1] for p in all_pts) / len(all_pts)
avg_z = sum(p[2] for p in all_pts) / len(all_pts)
target_loc = mathutils.Vector((avg_x, avg_y, avg_z))

def create_cam(name, loc):
    c_data = bpy.data.cameras.new(name)
    c_data.lens = 935.3074360871938 * 36.0 / 1080
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

cam0 = create_cam("Camera_0", (-0.3076140609138833, -1.0328445749984272, 0.49057877791585214))
cam1 = create_cam("Camera_1", (0.29023637382492173, -0.887830126137747, 0.3954192039417025))

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
    }, f, indent=2)
