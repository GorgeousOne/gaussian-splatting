"""Helper to visualize point clouds for debugging.
I can only get it to run with 'python -m depth_pruning.render_helper' because of weird relative package imports.
"""

import os
import colorsys

import numpy as np
import pyvista as pv
import scene.dataset_readers as dr
import utils.graphics_utils as gu
import utils.read_write_model as rwm

def norm_vec(v, scale=1):
    mag = np.linalg.norm(v)
    return v / mag * scale

def c2w_vec(v_cam, c2w_rot, c2w_t):
    return c2w_rot @ v_cam + c2w_t

def render_pcd(plotter, ply_path):
    pcd = dr.fetchPly(ply_path)
    point_cloud = pv.PolyData(pcd.points)
    if pcd.colors is None:
        plotter.add_points(point_cloud, point_size=3)
    else:
        point_cloud['colors'] = pcd.colors
        plotter.add_points(point_cloud, scalars='colors', rgb=True, point_size=1, )


def render_cam(plotter, key, images, cameras, color='blue', scale=1, show_up=True):
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]
    
    # get pinhole camera focal length
    f_x = cam_intrinsic.params[0]

    c2w_rot = rwm.qvec2rotmat(image_meta.qvec).T
    c2w_t = c2w_rot @ -image_meta.tvec

    w = cam_intrinsic.width
    h = cam_intrinsic.height
    vertices = np.array([
        c2w_t,
        c2w_rot @ norm_vec(np.array([-w/2, -h/2, f_x]), scale) + c2w_t,
        c2w_rot @ norm_vec(np.array([w/2, -h/2, f_x]), scale) + c2w_t,
        c2w_rot @ norm_vec(np.array([-w/2, h/2, f_x]), scale) + c2w_t,
        c2w_rot @ norm_vec(np.array([w/2, h/2, f_x]), scale) + c2w_t,
    ])
    # define connected points as padded connectivity array
    indices = np.hstack([
        [3, 1, 0, 2,
         3, 3, 0, 4,
         5, 1, 2, 4, 3, 1],
    ])
    # plot camera cone
    cam_cone = pv.PolyData(vertices)
    cam_cone.lines = indices
    plotter.add_mesh(cam_cone, color=color, point_size=0)
    # plot camera up vector
    if show_up:
        cam_up = pv.Arrow(start=c2w_t, direction=c2w_rot @ np.array([0, -1, 0]), shaft_radius=0.025, tip_radius=0.05)
        plotter.add_mesh(cam_up, color='green', point_size=0)


def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))


if __name__ == "__main__":
    plotter = pv.Plotter(window_size=[1920, 1080])
    render_pcd(plotter, "/home/mighty/repos/datasets/db/playroom/sparse/0/points3D.ply")
    render_pcd(plotter, "/home/mighty/repos/datasets/db/playroom/pcds/DSC05580.ply")
    #render_pcd(plotter, "/home/mighty/repos/datasets/tandt/train/pcds/00029.ply")

    # image_metas consists of cam extrinsics and image info
    cam_intrinsics, images_metas, points3d = rwm.read_model(os.path.join("/home/mighty/repos/datasets/db/playroom/sparse/0"), ext=".bin")

    for i in [9]: #images_metas.keys():
        rainbow_color = hsv2rgb(i / len(images_metas) * 0.8, 1, 1)
        render_cam(plotter, i, images_metas, cam_intrinsics, color=rainbow_color, scale=0.3, show_up=False)
        

    plotter.show_axes()
    plotter.show_grid(
        grid=True,
        location='outer',
        color='black'
    )
    plotter.view_zx()
    plotter.renderer.camera.is_set = True
    plotter.camera.position = (0, 0, 0)
    # plotter.camera.distance = 5
    plotter.show()
