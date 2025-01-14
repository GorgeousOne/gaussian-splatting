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

        collection_name = f"Camera Array {obj.name}"
        cam_array = bpy.data.collections.get(collection_name)

        if cam_array:
            self.del_tree(cam_array)

        cam_array = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(cam_array)

        # Create cameras for each face
        for face in bm.faces:
            # Compute world-space face center
            face_center = sum((obj_matrix @ v.co for v in face.verts), Vector()) / len(face.verts)

            # Get face normal in world space
            face_normal = (obj_matrix.to_3x3() @ face.normal).normalized()

            # Create new camera
            cam_data = bpy.data.cameras.new(name="Face_Camera")
            cam_obj = bpy.data.objects.new(name="Face_Camera", object_data=cam_data)

            # Position camera at face center
            cam_obj.location = face_center

            # Align camera with face normal
            # Compute a rotation matrix to align Z-axis with the normal
            up_vector = Vector((0, 0, 1))
            rotation_matrix = up_vector.rotation_difference(-face_normal).to_matrix().to_4x4()
            cam_obj.matrix_world = Matrix.Translation(face_center) @ rotation_matrix
            cam_obj.data = reference_cam.data.copy()


            # Link camera to scene
            cam_array.objects.link(cam_obj)

        self.parent_cams(obj, cam_array)

        bm.free()
        return {'FINISHED'}

    def del_tree(self, coll):
        for c in coll.children:
            self.del_tree(c)
        bpy.data.collections.remove(coll,do_unlink=True)

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
    bl_label = "Face Cameras"
    bl_idname = "VIEW3D_PT_face_cameras"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Face Cameras"

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
