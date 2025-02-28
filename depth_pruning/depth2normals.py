import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import cv2
import matplotlib.pyplot as plt
from typing import List
import utils.read_write_model as rwm

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


def get_normal_map(depth_map, key, cameras, images: List[rwm.Image]):
    # rgba = cv2.imread("/home/mighty/repos/datasets/hah/esszimmer_small/test_depth.png", cv2.IMREAD_UNCHANGED)
    # inv_depth_map = rgba.view(np.float32).reshape(rgba.shape[0], rgba.shape[1])
    # eps = 1e-6  # Small value to avoid division by zero
    # depth_map = torch.tensor(1.0 / (inv_depth_map + eps))

    # invmonodepthmap = cv2.imread("/home/mighty/repos/datasets/hah/esszimmer_small/depth_any/DSC00087_DxO_83.png", cv2.IMREAD_UNCHANGED)
    # if invmonodepthmap.ndim != 2:
    #     invmonodepthmap = invmonodepthmap[..., 0]
    # invmonodepthmap = invmonodepthmap.astype(np.float32) / (2**16)
    # invmonodepthmap[invmonodepthmap < 1e-3] = np.nan
    # depth_map =  torch.tensor(1.0 / (invmonodepthmap + 1e-6))

    # i dont know flip and reflip
    depth_map = depth_map.fliplr()

    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]
    depth_shape = depth_map.shape
    map_scale = depth_shape[0] / cam_intrinsic.height

    f_x = cam_intrinsic.params[0]
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
    return normal_map.fliplr()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default="../datasets/hah/esszimmer_small")
    parser.add_argument('--depths_dir', default="../datasets/hah/esszimmer_small/depth_32")
    parser.add_argument('--model_type', default="bin")
    args = parser.parse_args()

    cam_intrinsics, images_metas, points3d = rwm.read_model(os.path.join(args.base_dir, "sparse", "0"), ext=f".{args.model_type}")
    key = 54
    rgba = cv2.imread("/home/mighty/repos/datasets/hah/esszimmer_small/depth_32/DSC00087_DxO_83.png", cv2.IMREAD_UNCHANGED)
    # rgba = cv2.imread("/home/mighty/repos/datasets/hah/esszimmer_small/depth_32/DSC00074_DxO_71.png", cv2.IMREAD_UNCHANGED)
    # rgba = cv2.imread("/home/mighty/repos/datasets/hah/esszimmer_small/depth_32/DSC00057_DxO_54.png", cv2.IMREAD_UNCHANGED)
    # rgba = cv2.imread("/home/mighty/repos/datasets/hah/esszimmer_small/depth_32/DSC00070_DxO_67.png", cv2.IMREAD_UNCHANGED)
    inv_depth_map = rgba.view(np.float32).reshape(rgba.shape[0], rgba.shape[1])

    eps = 1e-6  # Small value to avoid division by zero
    depth_map = torch.tensor(1.0 / (inv_depth_map + eps))
    depth_map = depth_map

    normal_map = get_normal_map(depth_map, key, cam_intrinsics, images_metas)
    normal_map = 0.5 * (normal_map + 1.0)

    plt.figure(figsize=(10, 8))
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0) # rm window padding
    plt.imshow(normal_map, interpolation='nearest')
    plt.show()