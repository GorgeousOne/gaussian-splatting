import bpy
import random as rnd

# 2 functions to randomly animate either render exposure or skybox light strength to simulate realworld camera artifacts


def make_fcurve_constant(name, animation_obj):
    that_fcurve = None
    for fcurve in animation_obj.animation_data.action.fcurves:
        print("curve", fcurve.data_path, "vs", name, fcurve.data_path == name)
        if fcurve.data_path == name:
            that_fcurve = fcurve
            break
    if not that_fcurve:
        raise Exception("no curve " + name)
        
    for keyframe in that_fcurve.keyframe_points:
            keyframe.interpolation = 'CONSTANT'


def animate_scene_exposure(exp_range=1., frame_step=10):
    # Render > Colormanagement > Exposure
    # ColorManagedViewSettings
    color_manage_obj = bpy.context.scene.view_settings
    scene = bpy.context.scene
    
    rnd.seed(0)        
    for i in range(1, scene.frame_end+1):
        rnd_exposure = rnd.uniform(-exp_range, exp_range)
        if i % frame_step == 1:
            color_manage_obj.exposure = rnd_exposure
            color_manage_obj.keyframe_insert(data_path="exposure", frame=i)
                    
    make_fcurve_constant("view_settings.exposure", bpy.context.scene)
    
    print(f"set exposure to +-{exp_range} for {total_frames//frame_step} frames")


def animate_back_light(min_strength=10, max_strength=20, frame_step=10):
    # World > Surface > Strength
    # NodeSocketFloat.default_value
    background_node = bpy.data.worlds["World"].node_tree.nodes["Background"]
    scene = bpy.context.scene
    
    rnd.seed(0)
    for i in range(1, scene.frame_end+1):
        rnd_strength = rnd.uniform(min_strength, max_strength)
        if i % frame_step == 1:
            print(rnd_strength)
            background_node.inputs[1].default_value = rnd_strength
            background_node.inputs[1].keyframe_insert(data_path="default_value", frame=i)
                        
    exposure_fcurve = None
    make_fcurve_constant('nodes["Background"].inputs[1].default_value', bpy.data.worlds['World'].node_tree)
    


#animate_scene_exposure()
animate_back_light()