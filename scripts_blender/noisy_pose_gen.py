import bpy
from mathutils import Matrix, Vector, Euler
import math
import random

random.seed(0)

def get_unscaled_cam_matrix(camera_obj):
    mat = camera_obj.matrix_world
    loc, rot, scale = mat.decompose()
    return Matrix.LocRotScale(loc, rot, (1.0, 1.0, 1.0))    

    
def add_pose_noise(matrix, dPos, dRot):
    '''add random offset and rotation (degrees) to a camera pose matrix'''
    offset_vec = Vector((
        random.uniform(-dPos, dPos),
        random.uniform(-dPos, dPos),
        random.uniform(-dPos, dPos)
        )
    )
    offset = Matrix.Translation(offset_vec)
    euler_rot = Euler((
        math.radians(random.uniform(-dRot, dRot)),
        math.radians(random.uniform(-dRot, dRot)),
        math.radians(random.uniform(-dRot, dRot))
    ), 'XYZ')    
    offrot = euler_rot.to_matrix().to_4x4()

    matrix = offrot @ matrix # rotation in local space (dunno if that makes a difference)
    matrix @= offset
    return matrix


def create_noisy_animation(cam, new_cam_name):
    scene = bpy.context.scene
    
    if new_cam_name in bpy.data.objects:
        new_cam = bpy.data.objects[new_cam_name]
        new_cam.animation_data_clear()
    else:
        cam_data = bpy.data.cameras.new(name=new_cam_name)
        new_cam = bpy.data.objects.new(new_cam_name, cam_data)
        scene.collection.objects.link(new_cam)
        
    # copy intrinsics
    new_cam.data = cam.data.copy()
    
    for frame in range(1, scene.frame_end+1):    
        scene.frame_set(frame)  # Set the frame
        transform_matrix = get_unscaled_cam_matrix(cam)
        transform_matrix = add_pose_noise(transform_matrix, dPos=0.002, dRot=.1)
        
        new_cam.matrix_world = transform_matrix
        new_cam.keyframe_insert(data_path="location", frame=frame)
        new_cam.keyframe_insert(data_path="rotation_euler", frame=frame)
    print("Created camera", new_cam_name, "with", scene.frame_end,"frames")

camera = bpy.data.objects.get("AnimationCamera")
create_noisy_animation(camera, 'AnimationCamera-NoiseyPoses')
