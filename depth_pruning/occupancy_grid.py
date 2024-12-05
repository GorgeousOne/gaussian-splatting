import numpy as np
import depth_pruning.tri_aabb as ta
import tqdm

class VoxelGrid:

    def __init__(self, density, bound_min, bound_max):
        self.density = density
        self.inv_density = 1 / density
        self.bound_max = np.array(bound_max)
        self.bound_min = np.array(bound_min)

        size = self.bound_max - self.bound_min
        shape = np.ceil(size * self.inv_density).astype(np.int64)
        self.voxels = np.zeros(shape, dtype=np.int64)

    def contains(self, points3d):
        return np.all((self.bound_min <= points3d) & (points3d <= self.bound_max), axis=1)

    #TODO maybe find a way to return -1 for values out of bounds?
    def __getitem__(self, indices):
        return self.voxels[tuple(indices.T)]

    def get_indices(self, points3d):
        return np.floor((points3d-self.bound_min) * self.inv_density).astype(np.int64)

    def get_points(self, indices):
        return indices * self.density + self.bound_min

    def get_occupancies(self, points3d):
        return self[self.get_indices(points3d)]

    def max_occ(self):
        return self.voxels.max()

    def add_occupancy(self, indices):
        np.add.at(self.voxels, tuple(indices.T), 1)

    def add_points(self, points3d):
        indices = self.get_indices(points3d)
        np.add.at(self.voxels, tuple(indices.T), 1)

def get_voxel_bounds(points3d, density:float):
    min = points3d.min(axis=0)
    max = points3d.max(axis=0)
    # round grid bounds to nearest multiple of density
    # so multiple grids align with each other
    grid_min = np.floor(min / density) * density
    grid_max = np.ceil(max / density) * density
    return grid_min, grid_max


def voxelize_pcd(points3d:np.ndarray, density:float) -> VoxelGrid:
    grid = VoxelGrid(density, *get_voxel_bounds(points3d, density))
    grid.add_points(points3d)
    return grid

def voxelize_mesh(obj_path, density):
    verts, faces = read_obj_to_np(obj_path)
    grid = VoxelGrid(density, *get_voxel_bounds(verts, density))
    voxel_indices = grid.get_indices(verts)
    extents = np.array([density, density, density])

    for face in tqdm.tqdm(faces):
        indices = voxel_indices[face]
        i_min = np.min(indices, axis=0)
        i_max = np.max(indices, axis=0)

        for idx in np.ndindex(*(i_max - i_min + 1)):
            voxel = i_min + idx
            if ta.intersects_triangle_aabb(*verts[face], grid.get_points(voxel) + 0.5 * density, extents):
                grid.add_occupancy(voxel)
    return grid

def read_obj_to_np(obj_path):
    vertices = []
    faces = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):  # Vertex line
                parts = line.split()
                vertex = list(map(float, parts[1:4]))  # Convert x, y, z to float
                vertices.append(vertex)
            elif line.startswith('f '):  # Face line
                parts = line.split()
                # OBJ format: Faces are 1-based, adjust for 0-based indexing
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(face)
    return np.array(vertices), np.array(faces, dtype=int)

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
