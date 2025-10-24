import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import cv2
from tqdm import tqdm
from depth_pruning.depth2pcd import load_depth_map

import scene.dataset_readers as dr
from utils.graphics_utils import fov2focal


def get_camera_rays(H, W, focal):
    """calculate ray directions for each pixel in the image"""
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32),
                          torch.arange(H, dtype=torch.float32),
                          indexing='xy')
    dirs = torch.stack([(i - W * 0.5) / focal, (j - H * 0.5) / focal, torch.ones_like(i)], dim=-1)
    return dirs


# https://atyuwen.github.io/posts/normal-reconstruction
def compute_normals(depths, points):
    # get 2 pixel depths neighbors in each direction

    neighbor_depths = [
        torch.roll(depths, shifts=(-1, 0), dims=(0, 1)),
        torch.roll(depths, shifts=(-2, 0), dims=(0, 1)),
        torch.roll(depths, shifts=(1, 0), dims=(0, 1)),
        torch.roll(depths, shifts=(2, 0), dims=(0, 1)),
        torch.roll(depths, shifts=(0, -1), dims=(0, 1)),
        torch.roll(depths, shifts=(0, -2), dims=(0, 1)),
        torch.roll(depths, shifts=(0, 1), dims=(0, 1)),
        torch.roll(depths, shifts=(0, 2), dims=(0, 1)),
    ]

    neighbors = [
        torch.roll(points, shifts=(-1, 0), dims=(0, 1)),
        torch.roll(points, shifts=(1, 0), dims=(0, 1)),
        torch.roll(points, shifts=(0, -1), dims=(0, 1)),
        torch.roll(points, shifts=(0, 1), dims=(0, 1)),
    ]

    # make a re-estimation of the center depth by extrapolating from neighbors
    # get difference between extrapolation and original depth
    extrap_l = torch.abs(2 * neighbor_depths[0] - neighbor_depths[1] - depths)
    extrap_r = torch.abs(2 * neighbor_depths[2] - neighbor_depths[3] - depths)
    extrap_u = torch.abs(2 * neighbor_depths[4] - neighbor_depths[5] - depths)
    extrap_d = torch.abs(2 * neighbor_depths[6] - neighbor_depths[7] - depths)

    # mask which extrapolation is more accurate to original depth (left vs right, up vs down)
    maks_horz = extrap_l < extrap_r
    mask_vert = extrap_u < extrap_d

    # broadcast masks to the same shape as neighbors
    maks_horz = maks_horz.unsqueeze(-1).expand(-1, -1, 3)
    mask_vert = mask_vert.unsqueeze(-1).expand(-1, -1, 3)

    # choose more accurate extrapolation for each direction
    deriv_horz = torch.where(maks_horz, points - neighbors[0], neighbors[1] - points)
    deriv_vert = torch.where(mask_vert, points - neighbors[2], neighbors[3] - points)

    # create normals from cross product of derivatives
    normal = torch.cross(deriv_vert, deriv_horz)
    normal = F.normalize(normal, dim=-1)
    return normal


def calc_normal_map(depth_map, cam:dr.CameraInfo):
    # idk, couldnt get this right without flip and reflip at end
    depth_map = depth_map.fliplr()

    depth_shape = depth_map.shape
    map_scale = depth_shape[0] / cam.height

    f_x = fov2focal(cam.FovX, cam.width)
    dirs = get_camera_rays(depth_shape[0], depth_shape[1], f_x * map_scale)
    points = dirs * depth_map[..., None]

    # ---
    # ply_path = "/home/mighty/repos/datasets/hah/esszimmer_small/example.ply"
    # x, y, z = points.shape
    # points = points.reshape(x * y, z)
    # import utils.graphics_utils as gu
    # import scene.dataset_readers as dr
    # cloud = gu.BasicPointCloud(points, np.ones_like(points, dtype=np.uint) * 255, np.zeros_like(points))
    # dr.storePly(ply_path, cloud.points, cloud.colors)
    # ---

    normal_map = compute_normals(depth_map, points)
    normal_map = normal_map.fliplr()
    return 0.5 * (normal_map + 1.0)


#okay shit, forgot about this, do i need 32 bit depth over 16 bit depth?
def load_rgba_depth_map(depth_path):
    rgba = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    inv_depth_map = rgba.view(np.float32).reshape(rgba.shape[0], rgba.shape[1])

    eps = 1e-6  # Small value to avoid division by zero
    depth_map = torch.tensor(1.0 / (inv_depth_map + eps))
    depth_map = depth_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_path', default="../datasets/db/playroom/metashape_reco")
    parser.add_argument('-d', '--depths', default="depths")
    parser.add_argument('-r', '--rel_out_dir', default="normals")
    args = parser.parse_args()

    depths_dir = os.path.join(args.source_path, args.depths)
    normals_dir = os.path.join(args.source_path, args.rel_out_dir)
    os.makedirs(normals_dir, exist_ok=True)

    from scene.dataset_readers import sceneLoadTypeCallbacks

    #we should have a function for this shit
    if os.path.exists(os.path.join(args.source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, '', False, False)
    elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, False, args.depths, '', False)
    else:
        assert False, "Could not recognize scene type!"


    exist_ok = False

    for i in tqdm(range(len(scene_info.train_cameras))): #range(9, 10)):
        cam_info = scene_info.train_cameras[i]
        out_path = os.path.join(normals_dir, cam_info.image_name + ".png")

        if exist_ok and os.path.exists(out_path):
            continue

        depth_filepath = os.path.join(depths_dir, cam_info.image_name + ".png")
        depth_map = load_depth_map(depth_filepath, cam_info.depth_params)
        normal_map = calc_normal_map(torch.from_numpy(depth_map), cam_info)


        valid_mask = depth_map > 0
        normal_map[~valid_mask] = 0

        out_img = (normal_map * 255).numpy().astype(np.uint8)
        out_img = out_img[:, :, [2, 1, 0]] # RGB -> BGR conversion
        cv2.imwrite(out_path, out_img)

        #depth test
        
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 8))
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0) # rm window padding
        # # plt.imshow(normal_map, interpolation='nearest')
        # plt.imshow(depth_map, interpolation='nearest')
        # plt.show()
        # break