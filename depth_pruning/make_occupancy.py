import numpy as np
from trimesh.voxel import VoxelGrid

def get_voxel_bounds(points3d, density:float):
    min = points3d.min(axis=0)
    max = points3d.max(axis=0)
    # round grid bounds to nearest multiple of density
    # so multiple grids align with each other
    min_bound = np.floor(min / density) * density
    max_bound = np.ceil(max / density) * density
    return np.stack([min_bound, max_bound], axis=0)


def voxelize_pcd(points3d:np.ndarray, density:float):
    bounds = get_voxel_bounds(points3d, density)
    grid_shape = np.ceil((bounds[1] - bounds[0]) / density).astype(int)
    voxel_grid = np.zeros(grid_shape, dtype=bool)

    indices = ((points3d - bounds[0]) / density).astype(int)
    voxel_grid[tuple(indices.T)] = True
    return bounds, voxel_grid

# hehe
def voxelize_mesh(mesh:VoxelGrid, density=0.1):
    '''
    Returns a boolean voxel grid with all voxels set to true where the mesh intersects.
    AND THE WHOLE THING IS NOT AXIS ALIGNED!!! ITS OFFSET BY 0.5x OF THE DENSITY!!!
    '''
    return mesh.voxelized(density)