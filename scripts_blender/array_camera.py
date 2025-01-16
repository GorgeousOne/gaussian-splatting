import bpy
import bmesh
from mathutils import Vector, Matrix
import time

class OBJECT_OT_place_cameras(bpy.types.Operator):
    """Places a camera at the center of each face, pointing along the normal"""
    bl_idname = "object.place_cameras_on_faces"
    bl_label = "Place Cameras on Faces"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.object

        # Ensure an object is selected
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object")
            return {'CANCELLED'}

        reference_cam = context.scene.reference_camera
        if reference_cam == None and reference_cam.type == 'CAMERA':
            self.report({'ERROR'}, "Select a reference camera")
            return {'CANCELLED'}

        # Switch to object mode before working with data
        bpy.ops.object.mode_set(mode='OBJECT')

        # Get mesh data
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)

        # Ensure object transforms are applied
        obj_matrix = obj.matrix_world

        collection_name = f"Camera Array - {obj.name}"
        cam_array = bpy.data.collections.get(collection_name)

        if cam_array:
            self.del_tree(cam_array)

        cam_array = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(cam_array)

        # Create cameras for each face
        for face in sorted(bm.faces, key=lambda face: face_center(face)):
            # Compute world-space face center
            translation = face_center(face)
            forward_vec = (obj_matrix.to_3x3() @ face.normal).normalized()
            up_vec = Vector((0, 0, 1))
            right_vec = forward_vec.cross(up_vec).normalized()
            up_vec = right_vec.cross(forward_vec).normalized()
            rotation_matrix = Matrix([right_vec, up_vec, -forward_vec]).transposed()

            cam_data = bpy.data.cameras.new(name="Camera")
            cam_obj = bpy.data.objects.new(name="Camera", object_data=cam_data)
            cam_obj.location = translation
            cam_obj.rotation_euler = rotation_matrix.to_euler()
            cam_obj.data = reference_cam.data.copy()
            cam_obj.hide_render = False


            # Link camera to scene
            cam_array.objects.link(cam_obj)

        self.parent_cams(obj, cam_array)

        bm.free()
        return {'FINISHED'}

    def face_center(face):
        return = sum((v.co for v in face.verts), mathutils.Vector((0, 0, 0))) / len(face.verts)

    def del_tree(self, coll):
        for c in coll.children:
            self.del_tree(c)
        bpy.data.collections.remove(coll, do_unlink=True)

    def parent_cams(self, obj, cam_array):
        bpy.ops.object.select_all(action='DESELECT')  # Deselect everything
        for cam in cam_array.objects:
            cam.select_set(True)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)

class VIEW3D_PT_camera_panel(bpy.types.Panel):
    """Creates a Panel in the 3D Viewport"""
    bl_label = "Camera Arrays"
    bl_idname = "VIEW3D_PT_camera_arrays"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Camera Arrays"

    def draw(self, context):
        layout = self.layout
        # Add camera selector dropdown
        layout.prop(context.scene, "reference_camera", text="Reference Camera")
        layout.operator("object.place_cameras_on_faces")

def register():
    bpy.utils.register_class(OBJECT_OT_place_cameras)
    bpy.utils.register_class(VIEW3D_PT_camera_panel)
    # Add property to store reference camera selection
    bpy.types.Scene.reference_camera = bpy.props.PointerProperty(
        type=bpy.types.Camera,
        name="Reference Camera",
        description="Select a camera to copy settings from"
    )

def unregister():
    bpy.utils.unregister_class(OBJECT_OT_place_cameras)
    bpy.utils.unregister_class(VIEW3D_PT_camera_panel)
    del bpy.types.Scene.reference_camera

if __name__ == "__main__":
    register()
