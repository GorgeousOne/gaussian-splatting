import bpy
import json
import os
import math
#from json import encoder
#encoder.FLOAT_REPR = lambda o: format(o, '.6f')

def get_camera_angle_x(camera):
    """Calculate horizontal FOV in radians."""
    scene = bpy.context.scene
    #(2 * .data.camera_angle) if camera.data.lens_unit == 'FOV'
    return 2 * math.atan((camera.data.sensor_width / 2) / camera.data.lens)

def get_keyframes(camera):
    """Extracts all keyframes of the camera."""
    keyframes = set()

    if camera.animation_data and camera.animation_data.action:
        for fcurve in camera.animation_data.action.fcurves:
            keyframes.update({int(kp.co.x) for kp in fcurve.keyframe_points})

    return sorted(keyframes)

def export_camera_transforms(camera, output_file):
    """Exports all keyframe transformations of a camera into a JSON file."""

    scene = bpy.context.scene
    frames_data = []

    for frame in range(1, 496):
        scene.frame_set(frame)  # Set the frame
        transform_matrix = [list(row) for row in camera.matrix_world]
        file_path = f"images/{frame:04d}"

        frames_data.append({
            "file_path": file_path,
            "transform_matrix": transform_matrix
        })
    data = {
        "camera_angle_x": get_camera_angle_x(camera),
        "frames": frames_data
    }

    with open(output_file, "w") as f:
        json.dump(round_floats(data), f, indent=2, default=lambda x: round(x, 6) if isinstance(x, float) else x)

    print(f"Exported camera transforms to {os.path.abspath(output_file)}")


def round_floats(o, digits=6):
    if isinstance(o, float):
        return round(o, digits)
    if isinstance(o, dict):
        return {k: round_floats(v, digits) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_floats(x, digits) for x in o]
    return o

camera = bpy.data.objects.get("TrainCamera")
export_camera_transforms(camera, "transforms.json")
