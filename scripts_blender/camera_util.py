import bpy

def set_camera_size(size=0.5):
    """Sets the viewport display size of all cameras in the scene."""
    for cam in bpy.data.cameras:
        cam.display_size = size
        print(f"Set camera '{cam.name}' display size to {size}")


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

    for i, cam in enumerate(cams, start=1):
        cam_obj.matrix_world = cam.matrix_world
        cam_obj.keyframe_insert(data_path="location", frame=i)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=i)
    print("Created camera", cam_name, "with", len(cams),"frames")


def create_ttc_cameras():
    scene = bpy.context.scene
    cameras = [obj for obj in scene.objects if obj.type == 'CAMERA' and not obj.hide_render]
    cameras = sorted(cameras, key=lambda obj: obj.name)

    if not cameras:
        print("No cameras found in the scene.")
        return

    test_step = 1000000
    train_cam_frames = []
    test_cam_frames = []
    for i, cam in enumerate(cameras):
        if i % test_step == test_step-1:
            test_cam_frames.append(cam)
        else:
            train_cam_frames.append(cam)
    print(f"Train cameras: {len(train_cam_frames)}")
    print(f"Test cameras: {len(test_cam_frames)}")
    animate_camera(scene, "TrainCamera", train_cam_frames)
    if len(test_cam_frames) > 0:
        animate_camera(scene, "TestCamera", test_cam_frames)


def print_trans():
    camera = bpy.context.scene.camera
    transformation_matrix = camera.matrix_world
    print(transformation_matrix)
    print(camera.matrix_world.to_translation())


#cameras = [obj for obj in bpy.context.scene.objects if obj.type == 'CAMERA' and not obj.hide_render]
#cameras = sorted(cameras, key=lambda obj: obj.name)

#print(len(cameras))
#print(cameras)

#desired_size = 0.1
#set_camera_size(desired_size)

create_ttc_cameras()
