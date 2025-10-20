import bpy

def set_camera_size(size=0.5):
    """Sets the viewport display size of all cameras in the scene."""
    for cam in bpy.data.cameras:
        cam.display_size = size

def set_camera_lens(lens):
    """Sets the viewport display size of all cameras in the scene."""
    for cam in bpy.data.cameras:
        cam.lens = lens


def animate_camera(scene, cam_name, cams):
    if cam_name in bpy.data.objects:
        cam_obj = bpy.data.objects[cam_name]
        cam_obj.animation_data_clear()
    else:
        cam_data = bpy.data.cameras.new(name=cam_name)
        cam_obj = bpy.data.objects.new(cam_name, cam_data)
        scene.collection.objects.link(cam_obj)

    # copy intrinsics
    cam_obj.data = cams[0].data.copy()
    
    # create key frames
    for i, cam in enumerate(cams, start=1):
        cam_obj.matrix_world = cam.matrix_world
        cam_obj.keyframe_insert(data_path="location", frame=i)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=i)
    print("Created camera", cam_name, "with", len(cams),"frames")


def create_train_test_cameras():
    '''create one animation of all placed cameras as "TrainCamera" and "TestCamera'''
    scene = bpy.context.scene
    cameras = [obj for obj in scene.objects if obj.type == 'CAMERA' and not obj.hide_render]
    cameras = sorted(cameras, key=lambda obj: obj.name)
    
    if not cameras:
        print("No cameras found in the scene.")
        return
    
    cam_frames = []
    for i, cam in enumerate(cameras):
        cam_frames.append(cam)
        
    print(f"Concated {len(cam_frames)} cameras to animation")
    animate_camera(scene, "AnimationCamera", cam_frames)

    scene.frame_end = len(cam_frames) - 1

def print_trans():
    camera = bpy.context.scene.camera
    transformation_matrix = camera.matrix_world
    print(transformation_matrix)
    print(camera.matrix_world.to_translation())    


#cameras = [obj for obj in bpy.context.scene.objects if obj.type == 'CAMERA' and not obj.hide_render]
#cameras = sorted(cameras, key=lambda obj: obj.name)

#print(len(cameras))
#print(cameras)

#set_camera_size(.2)
#set_camera_lens(30)

create_train_test_cameras()
