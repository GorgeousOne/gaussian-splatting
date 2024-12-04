"""Helper to visualize point clouds for debugging.
I can only get it to run with
python -m depth_pruning.render_helper
because of weird relative package imports.
"""

import os
import colorsys
import warnings

import numpy as np
import pyvista as pv
import scene.dataset_readers as dr
import utils.graphics_utils as gu
import utils.read_write_model as rwm
import depth_pruning.occupancy_grid as oc

class CheckboxList():
    def __init__(self, plotter, min_x=10, min_y=10, spacing=60, text_size=12):
        self.pl = plotter
        self.min_x = min_x
        self.min_y = min_y
        self.spacing = spacing
        self.actors = []

    def add_checkbox(self, actor, label="", is_visible=False):
        pos_y = self.min_y + len(self.actors) * self.spacing
        plotter.add_checkbox_button_widget(lambda flag: actor.SetVisibility(flag), value=is_visible, color_on='white', position=(self.min_x, pos_y))
        plotter.add_text(label, position=(70, pos_y))

        actor.SetVisibility(is_visible)
        self.actors.append(actor)


def norm_vec(v, scale=1):
    mag = np.linalg.norm(v)
    return v / mag * scale


def c2w_vec(v_cam, c2w_rot, c2w_t):
    return c2w_rot @ v_cam + c2w_t


def fetchObj(path):
    with open(path) as file:
        vertices, colors, normals = [], [], []
        for line in file:
            parts = line.split()
            if parts[0] == 'v':
                vertices.append(list(map(float, parts[1:4])))
                colors.append(list(map(float, parts[4:7])))
            elif parts[0] == 'vn':
                normals.append(list(map(float, parts[1:4])))
    return gu.BasicPointCloud(
        points=np.array(vertices),
        colors=np.array(colors),
        normals=np.array(normals))


def render_pcd(plotter, file_path, add_checkbox=True):
    if file_path.endswith('.ply'):
        pcd = dr.fetchPly(file_path)
    else:
        pcd = fetchObj(file_path)

    point_cloud = pv.PolyData(pcd.points)
    if pcd.colors is None:
        actor = plotter.add_points(point_cloud, point_size=3)
    else:
        point_cloud['colors'] = pcd.colors
        actor = plotter.add_points(point_cloud, scalars='colors', rgb=True, point_size=1, )

    if add_checkbox:
        global checkboxes
        checkboxes.add_checkbox(actor, 'pcd', is_visible=True)


def render_cam(plotter, key, images, cameras, color='blue', scale=1, show_up=True):
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]

    # get pinhole camera focal length
    f_x = cam_intrinsic.params[0]

    c2w_rot = rwm.qvec2rotmat(image_meta.qvec).T
    c2w_t = c2w_rot @ -image_meta.tvec

    w = cam_intrinsic.width
    h = cam_intrinsic.height

    # vertices the camera frustum pyramid
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
    # plot frustum
    cam_cone = pv.PolyData(vertices)
    cam_cone.lines = indices
    plotter.add_mesh(cam_cone, color=color, point_size=0)
    # plot camera up vector
    if show_up:
        cam_up = pv.Arrow(start=c2w_t, direction=c2w_rot @ np.array([0, -1, 0]), shaft_radius=0.025, tip_radius=0.05)
        plotter.add_mesh(cam_up, color='green', point_size=0)


def render_voxels(plotter, grid=oc.VoxelGrid):
    min_point = grid.bound_min
    grid_shape = grid.voxels.shape
    voxel_size = grid.resolution
    occupancy_counts = grid.voxels

    # Generate the voxel grid points
    x = np.arange(grid_shape[0] + 1) * voxel_size + min_point[0]
    y = np.arange(grid_shape[1] + 1) * voxel_size + min_point[1]
    z = np.arange(grid_shape[2] + 1) * voxel_size + min_point[2]
    x, y, z = np.meshgrid(x, y, z, indexing="ij")

    grid = pv.StructuredGrid(x, y, z)
    # Assign cell data for occupancy
    #grid["occupancy"] = occupancy_counts.ravel(order="F")  # PyVista uses Fortran order
    grid["occupancy"] = np.sqrt(occupancy_counts.ravel(order="F"))  # Example: Square root scaling

    # Mask out cells with occupancy = 0
    masked_grid = grid.extract_cells(grid["occupancy"] > 0)

    actor = plotter.add_mesh(masked_grid, show_edges=True, cmap="viridis", scalars="occupancy")
    global checkboxes
    checkboxes.add_checkbox(actor, 'pcd occ grid')


def render_voxel_obj(plotter, obj_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh = pv.read(obj_path)

    # voxelized = pv.voxelize(mesh, density=0.1, check_surface=False)  # Adjust density as needed
    grid = oc.voxelize_mesh(mesh, 0.1)
    global checkboxes

    actor = plotter.add_mesh(grid, show_edges=True)
    checkboxes.add_checkbox(actor, 'mesh occ grid')

    actor = plotter.add_mesh(mesh)
    checkboxes.add_checkbox(actor, 'mesh')

    # def toggle_vis(flag):
    #     actor.SetVisibility(flag)

    # plotter.add_checkbox_button_widget(toggle_vis, value=True, color_on='white', position=(10, 70))
    # plotter.add_text('mesh occupancy grid', position=(70, 70))


def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))


def convert_bin2ply(bin_path):
    ply_path = bin_path[:-3] + 'ply'
    if os.path.exists(ply_path):
        return
    xyz, rgb, _ = dr.read_points3D_binary(bin_path)
    dr.storePly(ply_path, xyz, rgb)


if __name__ == "__main__":
    sparse_ply_path = '/home/mighty/repos/datasets/db/playroom/metashape_reco/sparse/0/points3D.ply'
    plotter = pv.Plotter(window_size=[1920, 1080])
    checkboxes = CheckboxList(plotter)

    # render_pcd(plotter, "/home/mighty/repos/datasets/db/playroom/sparse/0/points3D.ply")
    convert_bin2ply('/home/mighty/repos/datasets/db/playroom/metashape_reco/sparse/0/points3D.bin')
    render_pcd(plotter, sparse_ply_path)


    # image_metas consists of cam extrinsics and image info
    cam_intrinsics, images_metas, points3d = rwm.read_model(os.path.join('/home/mighty/repos/datasets/db/playroom/metashape_reco/sparse/0'), ext='.bin')
    pcds_dir = '/home/mighty/repos/datasets/db/playroom/pcds'
    for key in range(1, 20): #images_metas.keys():
        rainbow_color = hsv2rgb(key / len(images_metas) * 0.8, 1, 1)
        render_cam(plotter, key, images_metas, cam_intrinsics, color=rainbow_color, scale=0.3, show_up=False)

        img_name = images_metas[key].name.split('.')[0] + '.ply'
        # render_pcd(plotter, os.path.join(pcds_dir, img_name))


    voxels = oc.voxelize_pcd(dr.fetchPly(sparse_ply_path).points, 0.1)
    render_voxels(plotter, voxels)
    render_voxel_obj(plotter, '/home/mighty/repos/datasets/db/playroom/metashape_reco/mesh.obj')

    plotter.show_axes()
    plotter.show_grid(
        grid=True,
        location='outer',
        color='black'
    )
    plotter.view_xy()
    # plotter.enable_parallel_projection()
    plotter.show()
