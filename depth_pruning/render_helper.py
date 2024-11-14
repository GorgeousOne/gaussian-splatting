import os

import numpy as np
import pyvista as pv

from plyfile import PlyData
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text

def fetch_ply(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return positions, colors


def render_points(plotter, points, colors=None):
    point_cloud = pv.PolyData(points)
    if colors is None:
        plotter.add_points(point_cloud, point_size=3)
    else:
        point_cloud['colors'] = colors
        plotter.add_points(point_cloud, scalars='colors', rgb=True, point_size=3)


def render_cams(plotter, cams):

    line = pv.Line(pointa=(0, 0, 0), pointb=(1, 0, 0))
    plotter.add_mesh(line, color='red', line_width=3)

def render_pcd(ply_path):
    plotter = pv.Plotter()

    points, colors = fetch_ply(ply_path)
    render_points(plotter, points, colors)


    vector_start = np.array([0, 0, 0])
    vector_end = np.array([0.5, 0.5, 0.5])
    arrow = pv.Arrow(start=vector_start, direction=vector_end - vector_start)
    plotter.add_mesh(arrow, color='green')

    plotter.show_axes()
    plotter.show_grid(
        grid=True,
        location='outer',
        color='black'
    )

    plotter.view_xz()
    plotter.show()


if __name__ == "__main__":
    render_pcd("/home/mighty/repos/datasets/tandt/train/sparse/0/points3D.ply")