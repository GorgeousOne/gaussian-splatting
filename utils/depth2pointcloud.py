import numpy as np
import argparse
import cv2
from joblib import delayed, Parallel
import json
from read_write_model import *
import sys

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
    return invmonodepthmap


def cam2worldmat(cam_pos, cam_rot):
    t = np.eye(4)
    t[:3, :3] = cam_rot
    t[:3, 3] = cam_pos
    return t


# https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf_helpers.py#L123
def get_rays(H, W, focal, cam2world):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -np.ones_like(i)], axis=-1)
    rays_d = np.einsum('...ij,...j->...i', cam2world[:3, :3], dirs)  # Apply rotation from c2w
    rays_o = np.broadcast_to(cam2world[:3, -1], rays_d.shape)  # Broadcast camera origin to match rays_d shape
    return rays_o, rays_d


def get_cloud(key, cameras, images, depth_params=None):
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
    depth_map = depth_map * 3

    # get pinhole camera focal lengths
    f_x = cam_intrinsic.params[0]
    f_y = cam_intrinsic.params[1]

    # transform depth map pixels from image space to camera space using xys and depth
    # transform points from camera space to world space using camera position and rotation
    cam_pos = image_meta.tvec
    cam_rot = qvec2rotmat(image_meta.qvec)
    cam2world = cam2worldmat(cam_pos, cam_rot)
    
    ray_o, rays_d = get_rays(depth_shape[0], depth_shape[1], f_x*map_scale, cam2world)
    points = ray_o + rays_d * depth_map[..., np.newaxis]

    # return points in world space
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default="../datasets/tandt/train")
    parser.add_argument('--depths_dir', default="../datasets/tandt/train/depths")
    parser.add_argument('--out-dir', default="../datasets/tandt/train")
    parser.add_argument('--model_type', default="bin")
    args = parser.parse_args()

    cam_intrinsics, images_metas, points3d = read_model(os.path.join(args.base_dir, "sparse", "0"), ext=f".{args.model_type}")
    get_cloud(1, cam_intrinsics, images_metas)
