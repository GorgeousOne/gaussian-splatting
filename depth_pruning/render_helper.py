"""Helper to visualize point clouds for debugging.
I can only get it to run with
python -m depth_pruning.render_helper
because of weird relative package imports.
"""

import os
import colorsys
from trimesh import voxel

from scene.dataset_readers import sceneLoadTypeCallbacks
from types import SimpleNamespace

import numpy as np
import pyvista as pv
import scene.dataset_readers as dr
import utils.graphics_utils as gu
import depth_pruning.make_occupancy as mo

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

def render_pcd(plotter, pcd:gu.BasicPointCloud, name):
    '''render a pointcloud in the viewer + visibility toggle checkbox'''
    point_cloud = pv.PolyData(pcd.points)
    if pcd.colors is None:
        actor = plotter.add_points(point_cloud, point_size=3)
    else:
        point_cloud['colors'] = pcd.colors
        actor = plotter.add_points(point_cloud, rgb=True, point_size=1)

    global checkboxes
    checkboxes.add_checkbox(actor, name)


from utils.graphics_utils import fov2focal


def render_scene_info_cam(plotter, cam:dr.CameraInfo, color='blue', scale=0.1, show_up=False):
    f_x = fov2focal(cam.FovX, cam.width)
    # no transpose because dataset_readers already does this
    c2w_rot = cam.R
    c2w_t = -c2w_rot @ cam.T
    render_cam(plotter, c2w_t, c2w_rot, cam.width, cam.height, f_x, color, scale, show_up)


def render_cam(plotter, c2w_t, c2w_rot, w, h, f_x, color='blue', scale=0.1, show_up=False):
    '''render wirenet camera cone in viewer
    '''
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


def render_trimesh_voxel(plotter, grid:voxel.VoxelGrid, name):
    render_voxels(plotter, grid.bounds[0], grid.matrix, grid.pitch[0], name)


def render_voxels(plotter, min_point:np.ndarray, voxels:np.ndarray, density:float, name):
    '''render an occupancy voxel grid in the viewer + visibility toggle checkbox
        @param density: side length of one voxel cube
    '''
    # Generate the voxel grid points
    shape = voxels.shape
    x = np.arange(shape[0] + 1) * density + min_point[0]
    y = np.arange(shape[1] + 1) * density + min_point[1]
    z = np.arange(shape[2] + 1) * density + min_point[2]
    x, y, z = np.meshgrid(x, y, z, indexing="ij")

    grid = pv.StructuredGrid(x, y, z)
    grid["occupancy"] = voxels.ravel(order="F")  # PyVista uses Fortran order
    # only display cells where occupancy actually >0
    masked_grid = grid.extract_cells(grid["occupancy"] > 0)

    actor = plotter.add_mesh(masked_grid, show_edges=True)
    global checkboxes
    checkboxes.add_checkbox(actor, name)


def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))


def convert_bin2ply(bin_path):
    '''load a .bin 3d point file and write it to .ply'''
    ply_path = bin_path[:-3] + 'ply'
    if os.path.exists(ply_path):
        return
    xyz, rgb, _ = dr.read_points3D_binary(bin_path)
    dr.storePly(ply_path, xyz, rgb)


if __name__ == "__main__":
    # # sparse_ply_path = '/home/mighty/repos/datasets/db/playroom/metashape_reco/sparse/0/points3D.ply'
    # sparse_ply_path = '/home/mighty/repos/datasets/hah/esszimmer_small/example.ply'
    # sparse_bin_path = '/home/mighty/repos/datasets/db/playroom/metashape_reco/sparse/0/points3D.bin'
    # mesh_path = '/home/mighty/repos/datasets/db/playroom/metashape_reco/mesh.obj'
    # pcds_dir = '/home/mighty/repos/datasets/db/playroom/metashape_reco/pcds'

    #testing bedroom sparse point cloud against camera positions
    sparse_ply_path = '/home/mighty/Documents/blender/bedroom3/points3d.ply'
    sparse_bin_path = '/home/mighty/repos/datasets/db/playroom/metashape_reco/sparse/0/points3D.bin'

    # obj mesh to display
    # mesh_path = '/home/mighty/repos/datasets/db/playroom/metashape_reco/mesh.obj'
    mesh_path = '/home/mighty/Documents/blender/bedroom2/occupancy_mesh.obj'

    pcds_dir = '/home/mighty/repos/datasets/db/playroom/metashape_reco/pcds'


    # set path for blender bedroom cameras
    # args = SimpleNamespace()
    # args.source_path = '/home/mighty/Documents/blender/bedroom4'
    # args.images = 'images'

    # set path playroom cameras
    args = SimpleNamespace()
    args.source_path = '/home/mighty/repos/datasets/db/playroom'
    args.images = ''

    # load scene info (kinda only camemera infos)
    if os.path.exists(os.path.join(args.source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, '', '', False, False)
    elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, False, '', '', False)
    else:
        assert False, "Could not recognize scene type!"


    # create plotter
    plotter = pv.Plotter(window_size=[1920, 1080])
    checkboxes = CheckboxList(plotter)

    # render pcd
    convert_bin2ply(sparse_bin_path)
    pcd = dr.fetchPly(sparse_ply_path)
    render_pcd(plotter, pcd, 'sparse PCD')


    for i in range(25, 76):
        rainbow_color = hsv2rgb(i / len(scene_info.train_cameras) * 0.8, 1, 1)
        render_scene_info_cam(plotter, scene_info.train_cameras[i], color=rainbow_color, scale=0.3, show_up=False)
    #     img_name = images_metas[key].name.split('.')[0] + '.ply'
    #     if key == 15:
    #         depth_pcd = dr.fetchPly(os.path.join(pcds_dir, img_name))
    #         render_pcd(plotter, depth_pcd, 'depth map proj ' + str(key))


    # render voxels of pcd
    sparse_voxels = mo.voxelize_pcd(pcd.points, 0.1)
    render_trimesh_voxel(plotter, sparse_voxels, 'sparse voxels')

    # render db/playroom occupancy
    # mesh_voxels = mo.voxelize_mesh(trimesh.load(mesh_path), 0.1)
    # mo.save_voxel(mesh_voxels, '/home/mighty/repos/datasets/db/playroom/metashape_reco/occupancy_grid.npz')
    # mesh_voxels = mo.load_voxel('/home/mighty/repos/datasets/db/playroom/metashape_reco/occupancy_grid.npz')

    # render voxels haus am horn
    # mesh_voxels2 = mo.load_voxel('/home/mighty/repos/datasets/hah/obj/hah_occupancy_thin.npz')
    # mesh_voxels3 = mo.load_voxel('/home/mighty/repos/datasets/hah/obj/hah_occupancy_thick.npz')
    # render_trimesh_voxel(plotter, mesh_voxels2, 'fine voxels')
    # render_trimesh_voxel(plotter, mesh_voxels3, 'coarse voxels')

    # render obj mesh
    mesh = pv.read(mesh_path)
    actor = plotter.add_mesh(mesh)
    checkboxes.add_checkbox(actor, 'mesh', True)

    # create grid
    plotter.show_axes()
    plotter.show_grid(
        grid=True,
        location='outer',
        color='black'
    )
    plotter.view_xy()

    # render orthographic
    # plotter.enable_parallel_projection()
    plotter.show()
