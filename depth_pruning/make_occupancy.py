import numpy as np
from trimesh import grouping, voxel, base
from trimesh.voxel import encoding as enc
from trimesh import transformations as tr

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

# hehe
def voxelize_mesh(mesh:base.Trimesh, density:float):
    '''
    Returns a boolean voxel grid with all voxels set to true where the mesh intersects.
    AND THE WHOLE THING IS NOT AXIS ALIGNED!!! ITS OFFSET BY 0.5x OF THE DENSITY!!!
    '''
    return mesh.voxelized(density)