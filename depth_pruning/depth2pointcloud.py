"""Helper to convert image depth maps to point clouds
I can only get it to run with 'python -m depth_pruning.depth2pointcloud' because of weird relative package imports.
"""

import numpy as np
import argparse
import cv2
from joblib import delayed, Parallel
import json

import utils.read_write_model as rwm
import scene.dataset_readers as dr
import utils.graphics_utils as gu

import sys
import os
from typing import List

# from dataset_readers.py:160 readColmapSceneInfo
def read_depth_params(path):
    """read generated depth_params.json from colmap dir
    """
    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    try:
        with open(depth_params_file, "r") as f:
            depths_params = json.load(f)
        all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
        if (all_scales > 0).sum():
            med_scale = np.median(all_scales[all_scales > 0])
        else:
            med_scale = 0
        for key in depths_params:
            depths_params[key]["med_scale"] = med_scale

    except FileNotFoundError:
        print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
        sys.exit(1)
    return depths_params


# from make_depth_scale.py:8 get_scales
def load_norm_depth_map(args, image_meta):
    n_remove = len(image_meta.name.split('.')[-1]) + 1
    invmonodepthmap = cv2.imread(f"{args.depths_dir}/{image_meta.name[:-n_remove]}.png", cv2.IMREAD_UNCHANGED)
    
    if invmonodepthmap is None:
        return None
    
    if invmonodepthmap.ndim != 2:
        invmonodepthmap = invmonodepthmap[..., 0]

    invmonodepthmap = invmonodepthmap.astype(np.float32) / (2**16)
    invmonodepthmap[invmonodepthmap < 1e-3] = np.nan
    monodepthmap = 1 / invmonodepthmap
    return monodepthmap


# https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf_helpers.py#L123
def get_rays(H, W, focal, c2w_t, c2w_rot):
    """Get ray origins, directions from a pinhole camera."""

    # create xy coordinates for each pixel (in camera space)
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    # translate mid of image to 0,0
    dirs = np.stack([(i - W * 0.5) / focal, (j - H * 0.5) / focal, np.ones_like(i)], axis=-1)

    # apply inverse of world-to-camera rotation to all direction vectors
    rays_d = np.einsum('...ij,...j->...i', c2w_rot, dirs)
    # apply inverted translateion to all ray origins
    rays_o = np.broadcast_to(c2w_t, rays_d.shape)
    return rays_o, rays_d


def get_cloud(key, cameras, images:List[rwm.Image], depth_params=None):
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]
    depth_map = load_norm_depth_map(args, image_meta)
    depth_shape = depth_map.shape
    map_scale = depth_shape[0] / cam_intrinsic.height

    # map depth map from inverse depth to world depth
    # "A Hierarchical 3D Gaussian Representation for Real-Time Rendering of Very Large Datasets" mention something about depth not being invariant
    # TODO find you own depth map scaling
    # scale = depth_params[key]["scale"]
    # offset = depth_params[key]["offset"]
    # depth_map = 1 / (inv_depth_map * scale + offset)

    # get pinhole camera focal lengths
    f_x = cam_intrinsic.params[0]
    f_y = cam_intrinsic.params[1]

    # transform depth map pixels from image space to camera space using xys and depth
    # transform points from camera space to world space using camera position and rotation
    c2w_rot = rwm.qvec2rotmat(image_meta.qvec).T
    c2w_t = c2w_rot @ -image_meta.tvec
    
    depth_scale = 1/255
    ray_o, rays_d = get_rays(depth_shape[0], depth_shape[1], f_x*map_scale, c2w_t, c2w_rot)
    points = ray_o + rays_d * depth_map[..., np.newaxis] * depth_scale

    # flatten from 2D to 1D array of points
    x, y, z = points.shape
    points = points.reshape(x * y, z)

    
    # filter out points at infinity
    valid_mask = np.isfinite(depth_map).flatten()
    valid_points = points[valid_mask]

    return gu.BasicPointCloud(valid_points, np.ones_like(valid_points, dtype=np.uint) * 255, np.zeros_like(valid_points))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default="../datasets/tandt/train")
    parser.add_argument('--depths_dir', default="../datasets/tandt/train/depths")
    parser.add_argument('--out_dir', default="../datasets/tandt/train/pcds")
    parser.add_argument('--model_type', default="bin")
    args = parser.parse_args()

    i = 1

    cam_intrinsics, images_metas, points3d = rwm.read_model(os.path.join(args.base_dir, "sparse", "0"), ext=f".{args.model_type}")
    cloud = get_cloud(i, cam_intrinsics, images_metas)
    ply_path = os.path.join(args.out_dir, images_metas[i].name.split('.')[0] + ".ply")

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    print(ply_path)
    dr.storePly(ply_path, cloud.points, cloud.colors)


