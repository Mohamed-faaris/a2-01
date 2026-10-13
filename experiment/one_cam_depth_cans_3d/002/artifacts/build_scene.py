import bpy
import math
import os
import json
import mathutils

art_dir = r'/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/one_cam_depth_cans_3d/002/artifacts'
inp_dir = r'/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/one_cam_depth_cans_3d/002/inputs'
marker_img = r'/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/one_cam_depth_cans_3d/shared/marker_3.png'
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

hm = 0.088 / 2.0
m_mesh = bpy.data.meshes.new("MarkerMesh")
m_mesh.from_pydata([(-hm, -hm, 0), (hm, -hm, 0), (hm, hm, 0), (-hm, hm, 0)], [], [(0, 1, 2, 3)])
m_mesh.update()
uv = m_mesh.uv_layers.new(name="UVMap")
for loop_idx, uv_c in zip(m_mesh.polygons[0].loop_indices, [(0, 0), (1, 0), (1, 1), (0, 1)]):
    uv.data[loop_idx].uv = uv_c

marker_obj = bpy.data.objects.new("AprilTag_Marker", m_mesh)
scene.collection.objects.link(marker_obj)
marker_obj.location = (0.029, -0.038, 0.001)
marker_obj.data.materials.append(mat_m)

# Cans placed in 3D space
cans = [{'name': 'can_red', 'color_name': 'red', 'color': [0.85, 0.08, 0.08], 'position': [-0.2436215194679843, 0.18355229951694096, 0.31171597337887264], 'radius': 0.035, 'height': 0.125, 'gt_x_rel_tag_corner_m': -0.2286215194679843, 'gt_y_rel_tag_corner_m': 0.2655522995169409, 'gt_z_rel_tag_corner_m': 0.31071597337887263, 'gt_dist_to_tag_corner_m': 0.46832706419271275}, {'name': 'can_green', 'color_name': 'green', 'color': [0.08, 0.85, 0.08], 'position': [-0.015146950727785893, 0.13336982938613498, 0.3191709793561021], 'radius': 0.035, 'height': 0.125, 'gt_x_rel_tag_corner_m': -0.00014695072778589738, 'gt_y_rel_tag_corner_m': 0.21536982938613497, 'gt_z_rel_tag_corner_m': 0.3181709793561021, 'gt_dist_to_tag_corner_m': 0.3842095224076968}, {'name': 'can_blue', 'color_name': 'blue', 'color': [0.08, 0.2, 0.88], 'position': [0.23663884116601885, 0.07867343598961679, 0.19020820246823386], 'radius': 0.035, 'height': 0.125, 'gt_x_rel_tag_corner_m': 0.25163884116601887, 'gt_y_rel_tag_corner_m': 0.16067343598961678, 'gt_z_rel_tag_corner_m': 0.18920820246823386, 'gt_dist_to_tag_corner_m': 0.35346542022855154}]
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
all_pts = [c["position"] for c in cans] + [(0.029, -0.038, 0.001)]
avg_x = sum(p[0] for p in all_pts) / len(all_pts)
avg_y = sum(p[1] for p in all_pts) / len(all_pts)
avg_z = sum(p[2] for p in all_pts) / len(all_pts)
target_loc = mathutils.Vector((avg_x, avg_y, avg_z))

c_data = bpy.data.cameras.new("Camera_0")
c_data.lens = 935.3074360871938 * 36.0 / 1080
c_data.sensor_width = 36.0
c_data.sensor_fit = "HORIZONTAL"
c_data.clip_start = 0.01
c_data.clip_end = 100.0
c_obj = bpy.data.objects.new("Camera_0", c_data)
scene.collection.objects.link(c_obj)
c_obj.location = (0.005360242259114217, -1.09030309251472, 0.4344673093905266)
dir_vec = target_loc - mathutils.Vector((0.005360242259114217, -1.09030309251472, 0.4344673093905266))
c_obj.rotation_euler = dir_vec.to_track_quat('-Z', 'Y').to_euler()

bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(art_dir, "scene.blend"))

scene.camera = c_obj
scene.render.filepath = os.path.join(inp_dir, "cam0_render.png")
bpy.ops.render.render(write_still=True)

with open(os.path.join(inp_dir, "ground_truth_cameras.json"), "w") as f:
    json.dump({
        "cam0_matrix_world": [list(row) for row in c_obj.matrix_world],
        "cam0_position": list(c_obj.location),
    }, f, indent=2)
