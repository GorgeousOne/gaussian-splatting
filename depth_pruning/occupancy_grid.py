import numpy as np
import pyvista as pv

class VoxelGrid:

    def __init__(self, resolution, bound_max, bound_min=np.zeros((3,))):
        self.resolution = resolution
        self.inv_res = 1 / resolution
        self.bound_max = np.array(bound_max)
        self.bound_min = np.array(bound_min)

        size = self.bound_max - self.bound_min
        shape = np.ceil(size * self.inv_res).astype(np.int64)
        self.voxels = np.zeros(shape, dtype=np.int64)

    def contains(self, points3d):
        return np.all((self.bound_min <= points3d) & (points3d <= self.bound_max), axis=1)

    #TODO maybe find a way to return -1 for values out of bounds?
    def __getitem__(self, indices):
        return self.voxels[tuple(indices.T)]

    def get_indices(self, points3d):
        return np.floor((points3d-self.bound_min) * self.inv_res).astype(np.int64)

    def get_occupancies(self, points3d):
        return self[self.get_indices(points3d)]

    def max_occ(self):
        return self.voxels.max()

    def add_points(self, points3d):
        indices = self.get_indices(points3d)
        np.add.at(self.voxels, tuple(indices.T), 1)


def voxelize_pcd(points3d=np.ndarray, resolution=float) -> VoxelGrid:
    min = points3d.min(axis=0)
    max = points3d.max(axis=0)
    # round grid bounds to nearest multiple of resolution
    # so multiple grids align with each other
    grid_min = np.floor(min / resolution) * resolution
    grid_max = np.ceil(max / resolution) * resolution

    grid = VoxelGrid(resolution, grid_max, grid_min)
    grid.add_points(points3d)
    return grid

def b_round(x, base=1):
    return round(x / base) * base

def voxelize_mesh(mesh, density):
    """pyvista's voxelize but round the grid bounds to nearest multiple of density"""
    # check and pre-process input mesh
    surface = mesh.extract_geometry()  # filter preserves topology

    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    x = np.arange(b_round(x_min, density), b_round(x_max, density), density)
    y = np.arange(b_round(y_min, density), b_round(y_max, density), density)
    z = np.arange(b_round(z_min, density), b_round(z_max, density), density)
    x, y, z = np.meshgrid(x, y, z)

    # Create unstructured grid from the structured grid
    grid = pv.StructuredGrid(x, y, z)
    ugrid = pv.UnstructuredGrid(grid)

    # get part of the mesh within the mesh's bounding surface.
    selection = ugrid.select_enclosed_points(surface, tolerance=0.0, check_surface=False)
    mask = selection.point_data['SelectedPoints'].view(np.bool_)

    # extract cells from point indices
    vox = ugrid.extract_points(mask)
    return vox


if __name__ == '__main__':
    #test 6 values in [0, 0.5) and 5 in [0.5, 1)
    points_low = np.random.uniform(0.0, 0.5, (5, 3))
    points_high = np.random.uniform(0.5, 1.0, (5, 3))
    points = np.vstack((np.array([0, 0, 0]), points_low, points_high))

    v = voxelize_pcd(points, .5)

    assert (2, 2, 2) == v.voxels.shape
    assert np.array_equal(np.array([[[6, 0], [0, 0]], [[0, 0], [0, 5]]]), v.voxels)
    assert 6 == v.max_occ()
    assert np.array_equal([6, 0, 5], v[np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1]])])
    assert np.array_equal([True, True, False], v.contains(np.array([[0, 0, 0], [.5, .5, .5], [1, 1, 1]])))
    # print(v.get_occupancies(np.array([[0, 0, 0], [.5, .5, .5], [1, 1, 1]])))
