
import bpy

import numpy as np
import cv2
import os
import json
from tqdm import tqdm

# Set output directory (relative to blend file)
root_dir = bpy.path.abspath('//')
output_dir = os.path.join(root_dir, 'depths')
os.makedirs(output_dir, exist_ok=True)


# Ensure compositor is enabled
bpy.context.scene.use_nodes = True
scene = bpy.context.scene
tree = bpy.context.scene.node_tree
nodes = tree.nodes
links = tree.links

# Clear existing nodes
for node in nodes:
    nodes.remove(node)

# Create render layer node
render_layers = nodes.new(type="CompositorNodeRLayers")

# Create a viewer node to get pixel data
viewer_node = nodes.new(type="CompositorNodeViewer")
links.new(render_layers.outputs["Depth"], viewer_node.inputs[0])

# locate the camera with the animation of alle images
camera = bpy.data.objects.get("AnimationCamera")

# load existing depth params, they cant be easily regenerated
depths_params_path = os.path.join(root_dir, 'depth_params.json')
depth_params = {}
if os.path.exists(depths_params_path):
    with open(depths_params_path, 'r') as f:
        depth_params = json.load(f)

# iterate over all frames
for frame in tqdm(range(1, scene.frame_end+1)):
    
    img_name = f"{frame:04d}"
    file_name = img_name + '.png'
    file_path = os.path.join(output_dir, file_name)
    
    if os.path.exists(file_path):
        continue
    
    # advance animation
    scene.frame_set(frame)
    
    # Render the scene
    # print(f'render {frame}...')
    bpy.ops.render.render()
    
    # Get depth data from the Viewer Node
    depth_pixels = np.array(bpy.data.images['Viewer Node'].pixels)

    # Extract only the first channel (Depth, not RGBA)
    depth_pixels = depth_pixels[::4]

    # Get render resolution
    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y

    # Reshape to 2D array and flip Y-axis
    depth = np.reshape(depth_pixels, (height, width))[::-1]  # Flip Y-axis

    # Compute inverse depth (1/Z), handling zero and infinite values
    eps = 1e-6  # Small epsilon to avoid division by zero
    inv_depth = 1.0 / (depth + eps)

    # Set anything too close (depth < 0.1) or infinite to 0
    inv_depth = np.where((depth < 0.1) | (~np.isfinite(inv_depth)), 0.0, inv_depth)

    # Normalize to 16-bit range
    max_inv_depth = np.max(inv_depth)
    min_inv_depth = np.min(inv_depth)
    scale = max_inv_depth - min_inv_depth + 1e-6  # avoid division by zero
    offset = min_inv_depth - 1e-6 # avoid 0 values at miniumum, as 0 is used for
    inv_norm_depth = ((inv_depth - offset) / scale * (2**16)).astype(np.uint16)    
    cv2.imwrite(file_path, inv_norm_depth)

    depth_params[img_name] = dict(scale=scale, offset=offset)
    #save the precious depths params every fucking image
    with open(depths_params_path, 'w') as f:
        json.dump(depth_params, f, indent=4)
        
    # print(f"Saved depth map to {file_path}")

print("All depth maps saved successfully.")

# Add a Composite node to allow normal rendering output
composite_node = nodes.new(type="CompositorNodeComposite")
links.new(render_layers.outputs["Image"], composite_node.inputs[0])