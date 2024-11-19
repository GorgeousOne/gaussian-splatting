"""Helper to visualize point clouds for debugging.
I can only get it to run with 'python -m depth_pruning.render_helper' because of weird relative package imports.
"""

import os

import numpy as np
import pyvista as pv
import scene.dataset_readers as dr
import utils.graphics_utils as gu
import utils.read_write_model as rwm
from depth_pruning.depth2pointcloud import cam2worldmat

def norm_vec(v):
    mag = np.linalg.norm(v)
    return v / mag

def render_pcd(plotter, pcd:gu.BasicPointCloud):
    point_cloud = pv.PolyData(pcd.points)
    if pcd.colors is None:
        plotter.add_points(point_cloud, point_size=3)
    else:
        point_cloud['colors'] = pcd.colors
        plotter.add_points(point_cloud, scalars='colors', rgb=True, point_size=3)


def render_cam(plotter, key, images, cameras, color='blue'):
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]
    
    # get pinhole camera focal length
    f_x = cam_intrinsic.params[0]

    cam_pos = image_meta.tvec
    cam_rot = rwm.qvec2rotmat(image_meta.qvec)

    w = cam_intrinsic.width
    h = cam_intrinsic.height
    vertices = np.array([
        cam_pos,
        cam_pos + cam_rot @ norm_vec(np.array([-w/2, -h/2, -f_x])),
        cam_pos + cam_rot @ norm_vec(np.array([w/2, -h/2, -f_x])),
        cam_pos + cam_rot @ norm_vec(np.array([-w/2, h/2, -f_x])),
        cam_pos + cam_rot @ norm_vec(np.array([w/2, h/2, -f_x])),
    ])
    # define connected points as padded connectivity array
    indices = np.hstack([
        [3, 1, 0, 2,
         3, 3, 0, 4,
         5, 1, 2, 4, 3  , 1],
    ])
    # plot camera cone
    cam_cone = pv.PolyData(vertices)
    cam_cone.lines = indices
    plotter.add_mesh(cam_cone, color=color, line_width=2)
    # plot camera up vector
    arrow = pv.Arrow(start=cam_pos, direction=cam_rot @ np.array([0, 1, 0]))
    plotter.add_mesh(arrow, color='green', render_points_as_spheres=False)


def render_cloud(ply_path):
    plotter = pv.Plotter()
    pcd = dr.fetchPly(ply_path)
    render_pcd(plotter, pcd)

    cam_intrinsics, images_metas, points3d = rwm.read_model(os.path.join("/home/mighty/repos/datasets/tandt/train/sparse/0"), ext=".bin")
    render_cam(plotter, 1, images_metas, cam_intrinsics, 'red')
    render_cam(plotter, 13, images_metas, cam_intrinsics, (255, 128, 0))
    render_cam(plotter, 29, images_metas, cam_intrinsics, (255, 255, 0))

    plotter.show_axes()
    plotter.show_grid(
        grid=True,
        location='outer',
        color='black'
    )

    plotter.view_xz()
    plotter.show()


if __name__ == "__main__":
    # render_cloud("/home/mighty/repos/datasets/tandt/train/sparse/0/points3D.ply")
    render_cloud("/home/mighty/repos/datasets/tandt/train/pcds/00001.ply")