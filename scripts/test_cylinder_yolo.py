import bpy
import math
import os

out_dir = "/tmp/test_cylinder"
os.makedirs(out_dir, exist_ok=True)
blend_path = os.path.join(out_dir, "test.blend")
render_path = os.path.join(out_dir, "render.png")

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in [item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items] else "BLENDER_EEVEE"
scene.render.resolution_x = 640
scene.render.resolution_y = 480
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = render_path

# Add lighting
light_data = bpy.data.lights.new(name="Light", type='POINT')
light_data.energy = 500
light_obj = bpy.data.objects.new(name="Light", object_data=light_data)
scene.collection.objects.link(light_obj)
light_obj.location = (1, -1, 2)

# Cylinder (can): radius=0.033 (6.6cm dia), height=0.12 (12cm height, standard soda can)
bpy.ops.mesh.primitive_cylinder_add(radius=0.033, depth=0.12, location=(0.0, 1.0, 0.0))
can = bpy.context.active_object
mat = bpy.data.materials.new(name="CanRed")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs["Base Color"].default_value = (0.8, 0.05, 0.05, 1.0)
can.data.materials.append(mat)

# Camera at origin looking at can
cam_data = bpy.data.cameras.new("Camera")
cam_data.lens = 500.0 * 36.0 / 640.0
cam_data.sensor_width = 36.0
cam_obj = bpy.data.objects.new("Camera", cam_data)
scene.collection.objects.link(cam_obj)
cam_obj.location = (0, 0, 0)

track = cam_obj.constraints.new(type='TRACK_TO')
track.target = can
track.track_axis = 'TRACK_NEGATIVE_Z'
track.up_axis = 'UP_Y'

scene.camera = cam_obj
bpy.ops.render.render(write_still=True)
print("Render completed:", render_path)
