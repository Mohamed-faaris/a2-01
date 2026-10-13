import bpy
import math
import os
import json
import mathutils

art_dir = r'/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/12_realsense_sim_physical_gap/005/artifacts'
inp_dir = r'/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/12_realsense_sim_physical_gap/005/inputs'
marker_img = r'/home/mfk01/Projects/econ-systems/demos/a2-01/experiment/12_realsense_sim_physical_gap/shared/marker_1.png'
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

# Ambient lighting
world = bpy.data.worlds.new("World")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.85, 0.85, 0.85, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
scene.world = world

# Desk floor plane
bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0, 0, 0))
floor = bpy.context.active_object
mat_f = bpy.data.materials.new("DeskMat")
mat_f.use_nodes = True
mat_f.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.9, 0.9, 0.88, 1.0)
mat_f.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.2
floor.data.materials.append(mat_f)

# AprilTag Marker
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

hm = 0.124 / 2.0
m_mesh = bpy.data.meshes.new("MarkerMesh")
m_mesh.from_pydata([(-hm, -hm, 0.001), (hm, -hm, 0.001), (hm, hm, 0.001), (-hm, hm, 0.001)], [], [(0, 1, 2, 3)])
m_mesh.update()
uv = m_mesh.uv_layers.new(name="UVMap")
for loop_idx, uv_c in zip(m_mesh.polygons[0].loop_indices, [(0, 0), (1, 0), (1, 1), (0, 1)]):
    uv.data[loop_idx].uv = uv_c

marker_obj = bpy.data.objects.new("AprilTag_Marker", m_mesh)
scene.collection.objects.link(marker_obj)
marker_obj.location = (0.0, 0.0, 0.0)
marker_obj.data.materials.append(mat_m)

# Physical Bottles (24.1cm high, 9.0cm diameter)
bottles = [{'name': 'bottle_red', 'color_name': 'red', 'color': [0.85, 0.1, 0.1], 'position': [-0.12075854311379575, 0.1878965701623996, 0.0], 'radius': 0.045, 'height': 0.241, 'gt_x_rel_tag_corner_m': 0.18275854311379575, 'gt_y_rel_tag_corner_m': 0.2498965701623996, 'gt_z_rel_tag_corner_m': 0.0}, {'name': 'bottle_blue', 'color_name': 'blue', 'color': [0.1, 0.2, 0.85], 'position': [0.19113172778521575, 0.17557861855309373, 0.0], 'radius': 0.045, 'height': 0.241, 'gt_x_rel_tag_corner_m': -0.12913172778521576, 'gt_y_rel_tag_corner_m': 0.23757861855309373, 'gt_z_rel_tag_corner_m': 0.0}]
for b in bottles:
    pos = b["position"]
    radius = b["radius"]
    height = b["height"]
    name = b["name"]
    color = b["color"]

    # In Blender, cylinder origin is at center of cylinder, so z_center = height / 2.0
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=radius, depth=height, location=(pos[0], pos[1], pos[2] + height / 2.0))
    obj = bpy.context.active_object
    obj.name = name

    mat = bpy.data.materials.new(name + "_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
    bsdf.inputs["Roughness"].default_value = 0.3
    obj.data.materials.append(mat)

# Directional desk light
light_data = bpy.data.lights.new(name="Sun", type='SUN')
light_data.energy = 3.5
light_obj = bpy.data.objects.new("Sun", light_data)
scene.collection.objects.link(light_obj)
light_obj.location = (1.5, -1.5, 3.0)
light_obj.rotation_euler = (math.radians(40), math.radians(15), math.radians(45))

# Camera viewing at realistic grazing angle
c_data = bpy.data.cameras.new("RealSense_D415")
c_data.lens = 925.09 * 36.0 / 1280
c_data.sensor_width = 36.0
c_data.sensor_fit = "HORIZONTAL"
c_data.clip_start = 0.05
c_data.clip_end = 20.0
c_obj = bpy.data.objects.new("RealSense_D415", c_data)
scene.collection.objects.link(c_obj)
c_obj.location = (-0.08698968140294411, -0.7551114462746666, 0.37656320330745596)

target_loc = mathutils.Vector((0.0, 0.0, 0.0))
dir_vec = target_loc - mathutils.Vector((-0.08698968140294411, -0.7551114462746666, 0.37656320330745596))
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
