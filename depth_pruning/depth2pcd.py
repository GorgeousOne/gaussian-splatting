"""Helper to convert image depth maps to point clouds
I can only get it to run with
python -m depth_pruning.depth2pointcloud
because of weird relative package imports.
"""

import numpy as np
import argparse
import cv2

import scene.dataset_readers as dr
import utils.graphics_utils as gu

import os

from utils.graphics_utils import fov2focal


def load_depth_map(depth_path, depth_param, is_nerf_synthetic):
    # utils.camera_utils.py
    # print('/home/mighty/Documents/blender/bedroom4/depth_32/image_0001.png')
    # print(os.path.exists(depth_path), depth_path)
    try:
        invdepthmap = cv2.imread(depth_path, -1).astype(np.float32)
        print("and now?", np.max(invdepthmap), np.min(invdepthmap))
        invdepthmap /= float(2**16)
    except FileNotFoundError:
        print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
        raise
    except IOError:
        print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
        raise    
    scale = depth_param["scale"] if depth_param else 1
    offset = depth_param["offset"] if depth_param else 0

    print("off", offset, "scale", scale)
    print("and now?", np.max(invdepthmap), np.min(invdepthmap))
    depthmap = 1. / (invdepthmap * scale + offset)
    print("and now?", np.max(depthmap), np.min(depthmap))
    return depthmap


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


def get_scene_info_cloud(cam:dr.CameraInfo, is_nerf_synthetic):
    f_x = fov2focal(cam.FovX, cam.width)
    c2w_rot = cam.R
    c2w_t = -c2w_rot @ cam.T
    print('load depth', cam.depth_path, cam.depth_params, is_nerf_synthetic)
    depth_map = load_depth_map(cam.depth_path, cam.depth_params, is_nerf_synthetic)
    return get_cloud(depth_map, cam.height, f_x, c2w_t, c2w_rot, cam.depth_params)


def get_cloud(depth_map, cam_height, f_x, c2w_t, c2w_rot, depth_param):
    depth_shape = depth_map.shape
    map_scale = depth_shape[0] / cam_height

    # depth_scale = 1/255
    ray_o, rays_d = get_rays(depth_shape[0], depth_shape[1], f_x*map_scale, c2w_t, c2w_rot)
    points = ray_o + rays_d * depth_map[..., np.newaxis]

    # flatten from 2D to 1D array of points
    x, y, z = points.shape
    points = points.reshape(x * y, z)


    # filter out points at infinity
    valid_mask = np.isfinite(depth_map).flatten()
    valid_points = points[valid_mask]

    return gu.BasicPointCloud(valid_points, np.ones_like(valid_points, dtype=np.uint) * 255, np.zeros_like(valid_points))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_path', default="../datasets/db/playroom/metashape_reco")
    parser.add_argument('-d', '--depths', default="depths")
    parser.add_argument('-r', '--rel_out_dir', default="pcds")
    args = parser.parse_args()

    # cam_intrinsics, images_metas, points3d = rwm.read_model(os.path.join(args.base_dir, "sparse", "0"), ext=f".{args.model_type}")
    # depth_params = read_depth_params(os.path.join(args.base_dir, "sparse", "0", "depth_params.json"))

    from scene.dataset_readers import sceneLoadTypeCallbacks

    #idk load the stupid dataset again, dont want to reinvent the wheel
    if os.path.exists(os.path.join(args.source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, '', False, False)
    elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, False, args.depths, '', False)
    else:
        assert False, "Could not recognize scene type!"

    force_overwrite = True
    out_dir = os.path.join(args.source_path, args.rel_out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print('Creating pcd from images:')
    for i in range(0, 1):
        cam_info = scene_info.train_cameras[i]
        ply_path = os.path.join(out_dir, cam_info.image_name + ".ply")
        if os.path.exists(ply_path) and not force_overwrite:
            continue
        cloud = get_scene_info_cloud(cam_info, scene_info.is_nerf_synthetic)
        dr.storePly(ply_path, cloud.points, cloud.colors)
        print(ply_path)

    # for key in tqdm.tqdm(images_metas.keys()):
        # ply_path = os.path.join(args.out_dir, images_metas[key].name.split('.')[0] + ".ply")
        # if os.path.exists(ply_path) and not force_overwrite:
        #     continue
        # cloud = get_colmap_cloud(key, cam_intrinsics, images_metas, depth_params)

        # print(ply_path)
        # dr.storePly(ply_path, cloud.points, cloud.colors)
