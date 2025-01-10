import bpy

def set_camera_size(size=0.5):
    """Sets the viewport display size of all cameras in the scene."""
    for cam in bpy.data.cameras:
        cam.display_size = size
        print(f"Set camera '{cam.name}' display size to {size}")

# desired_size = 0.15
# set_camera_size(desired_size)


def animate_camera(scene, cam_name, cams):
    cam_data = bpy.data.cameras.new(name=cam_name)
    cam_obj = bpy.data.objects.new(cam_name, cam_data)
    scene.collection.objects.link(cam_obj)

    # copy intrinsics
    cam_obj.data = cams[0].data.copy()

    for i, cam in enumerate(cams):
        cam_obj.location = cam.location
        cam_obj.rotation_euler = cam.rotation_euler
        cam_obj.keyframe_insert(data_path="location", frame=i)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=i)

def create_animated_cameras():
    scene = bpy.context.scene
    cameras = [obj for obj in scene.objects if obj.type == 'CAMERA']

    if not cameras:
        print("No cameras found in the scene.")
        return

    test_step = 5

    # Create new cameras for train and test
    test_cam_frames = cameras[test_step-1::test_step]
    train_cam_frames = list(set(cameras) - set(test_cam_frames))
    print(f"Train cameras: {len(train_cam_frames)}")
    print(f"Test cameras: {len(test_cam_frames)}")
    animate_camera(scene, "TrainCamera", train_cam_frames)
    animate_camera(scene, "TestCamera", test_cam_frames)

# Run the function
create_animated_cameras()