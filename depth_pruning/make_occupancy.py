import numpy as np
from trimesh import grouping, voxel, base
from trimesh.voxel import encoding as enc
from trimesh import transformations as tr
import os

# from timesh.voxel.creation.py:voxelize_subdivide()
def voxelize_pcd(points3d:np.ndarray, density:float):
    # convert the vertices to their voxel grid position
    # still pondering why they used round instead of floor 🤔
    hit = np.round(points3d / density).astype(int)

    # remove duplicates
    unique, _inverse = grouping.unique_rows(hit)
    # get the voxel centers in model space
    occupied_index = hit[unique]
    origin_index = occupied_index.min(axis=0)
    origin_position = origin_index * density

    return voxel.VoxelGrid(
        enc.SparseBinaryEncoding(occupied_index - origin_index),
        transform=tr.scale_and_translate(scale=density, translate=origin_position),
    )

def voxelize_mesh(mesh:base.Trimesh, density:float):
    return mesh.voxelized(density)

def save_voxel(voxel_grid:voxel.VoxelGrid, npz_file:str):
    '''
    Save a trimesh voxel grid to a .npz file.
    '''
    output_dir = os.path.dirname(npz_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        np.savez(npz_file, transform=voxel_grid.transform, indices=voxel_grid.sparse_indices)
    except Exception as e:
        raise IOError(f'Failed to save voxel grid to {npz_file}: {e}')

def load_voxel(npz_file:str):
    '''
    Load a trimesh voxel grid from a .npz file.
    '''
    if not os.path.exists(npz_file):
        raise FileNotFoundError(f'The file {npz_file} does not exist.')

    try:
        data = np.load(npz_file)
    except Exception as e:
        raise IOError(f'Failed to load voxel grid from {npz_file}: {e}')

    if 'indices' not in data or 'transform' not in data:
        raise ValueError(f'The file {npz_file} is missing required voxel grid fields: "indices" or "transform".')

    return voxel.VoxelGrid(
        enc.SparseBinaryEncoding(data['indices']),
        data['transform'],
    )

if __name__ == '__main__':
    import trimesh
    import argparse

    parser = argparse.ArgumentParser(description='Converts a .obj 3D mesh to an occupancy voxel grid. Saves the voxel grid to a .npz file.')
    parser.add_argument('mesh_path', type=str, help='Path to .obj mesh file.')
    parser.add_argument('density', type=float, help='Size of each voxel cube in the grid.')
    parser.add_argument('--output', '-o', type=str, help='Path to save a .npz file. Defaults to mesh_path with "_occupancy.npz" suffix if omitted', default=None)
    args = parser.parse_args()

    if not os.path.exists(args.mesh_path):
        raise FileNotFoundError(f'The file {args.mesh_path} does not exist.')
    try:
        mesh = trimesh.load(args.mesh_path, force='mesh')
    except Exception as e:
        raise IOError(f'Failed to load mesh from {args.mesh_path}: {e}')

    voxel_grid = voxelize_mesh(mesh, args.density)

    if args.output is None:
        output_path = args.mesh_path[:-4] + '_occupancy.npz'
    else:
        output_path = args.output

    save_voxel(voxel_grid, output_path)
    print(f'Voxel grid saved to {output_path}.')
