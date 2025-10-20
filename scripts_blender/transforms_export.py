import bpy
import json
import os
import math
#from json import encoder
#encoder.FLOAT_REPR = lambda o: format(o, '.6f')
import mathutils

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

def get_unscaled_cam_matrix(camera_obj):
    #yeah it's really important to not include any scaling (from parent objects)
    # because that messes up the entire matrix and how things get rendered 
    # Decompose the matrix_world
    mat = camera_obj.matrix_world
    loc, rot, scale = mat.decompose()
    
    # Recompose with unit scale
    clean_matrix = mathutils.Matrix.LocRotScale(loc, rot, (1.0, 1.0, 1.0))    
    return [list(row) for row in clean_matrix]

def export_camera_transforms(camera):
    """Exports all keyframe transformations of a camera into a JSON file."""
        
    scene = bpy.context.scene
    train_frames_data = []
    test_frames_data = []
    test_step = 1000
    
    # this assumes that the camera_util script adapted the scene length
    for frame in range(1, scene.frame_end+1):
        scene.frame_set(frame)  # Set the frame
        transform_matrix = get_unscaled_cam_matrix(camera)
        file_path = f"images/{frame:04d}"
        
        if frame % test_step == 0:
           test_frames_data.append({
                "file_path": file_path,
                "transform_matrix": transform_matrix
            })
        else:
            train_frames_data.append({
                "file_path": file_path,
                "transform_matrix": transform_matrix
            })            
            
    train_data = {
        "camera_angle_x": get_camera_angle_x(camera),
        "frames": train_frames_data
    }
    test_data = {
        "camera_angle_x": get_camera_angle_x(camera),
        "frames": test_frames_data
    }    
    
    with open('transforms_train.json', "w") as f:
        json.dump(round_floats(train_data), f, indent=2, default=lambda x: round(x, 6) if isinstance(x, float) else x)
    with open('transforms_test.json', "w") as f:
        json.dump(round_floats(test_data), f, indent=2, default=lambda x: round(x, 6) if isinstance(x, float) else x)
    
    print(f"Exported {len(train_frames_data)}/{len(test_frames_data)} camera transforms to")
    print(os.path.abspath('test_transforms.json'))
    print(os.path.abspath('train_transforms.json'))


def round_floats(o, digits=6):
    if isinstance(o, float):
        return round(o, digits)
    if isinstance(o, dict):
        return {k: round_floats(v, digits) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_floats(x, digits) for x in o]
    return o

camera = bpy.data.objects.get("AnimationCamera")
export_camera_transforms(camera)
