import numpy as np

class VoxelGrid:

    def __init__(self, resolution, bound_max, bound_min=np.zeros((3,))):
        self.inv_res = 1 / resolution
        self.bound_max = np.array(bound_max)
        self.bound_min = np.array(bound_min)

        size = self.bound_max - self.bound_min
        shape = np.ceil(size * self.inv_res).astype(np.int64)
        self.voxels = np.zeros(shape, dtype=np.int64)

    def __contains__(self, points3d):
        inside = np.all((self.bound_min <= points3d) & (points3d <= self.bound_max), axis=1)
        return inside

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


def create_voxel_grid(points3d, resolution) -> VoxelGrid:
    min = points3d.min(axis=0)
    max = points3d.max(axis=0)
    grid = VoxelGrid(resolution, max, min)
    grid.add_points(points3d)
    return grid


if __name__ == '__main__':
    #test 6 values in [0, 0.5) and 5 in [0.5, 1)
    points_low = np.random.uniform(0.0, 0.5, (5, 3))
    points_high = np.random.uniform(0.5, 1.0, (5, 3))
    points = np.vstack((np.array([0, 0, 0]), points_low, points_high))

    v = create_voxel_grid(points, .5)

    assert (2, 2, 2) == v.voxels.shape
    assert np.array_equal(np.array([[[6, 0], [0, 0]], [[0, 0], [0, 5]]]), v.voxels)
    assert 6 == v.max_occ()
    assert np.array_equal(np.array([6, 0, 5]), v[np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1]])])
