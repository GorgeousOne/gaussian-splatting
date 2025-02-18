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
def compute_normals(depths, points, view_dirs):
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
        torch.roll(points, shifts=(-2, 0), dims=(0, 1)),
        torch.roll(points, shifts=(1, 0), dims=(0, 1)),
        torch.roll(points, shifts=(2, 0), dims=(0, 1)),
        torch.roll(points, shifts=(0, -1), dims=(0, 1)),
        torch.roll(points, shifts=(0, -2), dims=(0, 1)),
        torch.roll(points, shifts=(0, 1), dims=(0, 1)),
        torch.roll(points, shifts=(0, 2), dims=(0, 1)),
    ]

    # make extrapolation of center depth from each pair of neighbors
    leftExtrap = torch.abs(neighbor_depths[0] - neighbor_depths[1])
    rightExtrap = torch.abs(neighbor_depths[2] - neighbor_depths[3])
    upExtrap = torch.abs(neighbor_depths[4] - neighbor_depths[5])
    downExtrap = torch.abs(neighbor_depths[6] - neighbor_depths[7])

    # mask which extrapolation is more accurate to original depth (left vs right, up vs down)
    horzMask = leftExtrap < rightExtrap
    vertMask = upExtrap < downExtrap

    # broadcast masks to the same shape as neighbors
    horzMask = horzMask.unsqueeze(-1).expand(-1, -1, 3)
    vertMask = vertMask.unsqueeze(-1).expand(-1, -1, 3)

    # choose more accurate extrapolation for each direction
    horzDeriv = torch.where(horzMask, neighbors[0] - neighbors[1], neighbors[2] - neighbors[3])
    vertDeriv = torch.where(vertMask, neighbors[4] - neighbors[5], neighbors[6] - neighbors[7])

    # create normalized normals from cross product of derivatives
    normal = torch.cross(vertDeriv, horzDeriv)
    normal = F.normalize(normal, dim=-1)

    # account for different cross product directions, flip normals
    normal[~horzMask] *= -1
    normal[~vertMask] *= -1
    return normal


def get_normal_map(depth_map, key, cameras, images: List[rwm.Image]):
    rgba = cv2.imread("/home/mighty/repos/datasets/hah/esszimmer_small/test_depth.png", cv2.IMREAD_UNCHANGED)
    inv_depth_map = rgba.view(np.float32).reshape(rgba.shape[0], rgba.shape[1])
    eps = 1e-6  # Small value to avoid division by zero
    depth_map2 = torch.tensor(1.0 / (inv_depth_map + eps))

    invmonodepthmap = cv2.imread("/home/mighty/repos/datasets/tandt/train/depths/00241.png", cv2.IMREAD_UNCHANGED)
    invmonodepthmap = invmonodepthmap.astype(np.float32) / (2**16)
    if invmonodepthmap.ndim != 2:
        invmonodepthmap = invmonodepthmap[..., 0]
    invmonodepthmap[invmonodepthmap < 1e-3] = np.nan
    depth_map =  torch.tensor(1.0 / (invmonodepthmap + 1e-6))
    # breakpoint()


    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]
    depth_shape = depth_map.shape
    map_scale = depth_shape[0] / cam_intrinsic.height
    map_scale = 1

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

    normal_map = compute_normals(depth_map, points, dirs)
    return normal_map.permute(1, 2, 0)  # Convert back to (H, W, 3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default="../datasets/hah/esszimmer_small")
    parser.add_argument('--depths_dir', default="../datasets/hah/esszimmer_small/depth")
    parser.add_argument('--model_type', default="bin")
    args = parser.parse_args()

    cam_intrinsics, images_metas, points3d = rwm.read_model(os.path.join(args.base_dir, "sparse", "0"), ext=f".{args.model_type}")
    key = 83
    inv_depth_map = torch.tensor(cv2.imread("/home/mighty/repos/datasets/hah/esszimmer_small/depth/DSC00087_DxO_83.png", -1).astype(np.float32) / 512)
    depth_map = 1.0 / (inv_depth_map + 1e-6)


    normal_map = get_normal_map(depth_map, key, cam_intrinsics, images_metas)
    normal_map = 0.5 * normal_map + 0.5

    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0) # rm window padding
    plt.imshow(normal_map.permute(2, 0, 1), cmap='plasma', interpolation='nearest')
    plt.show()