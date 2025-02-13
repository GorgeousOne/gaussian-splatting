import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

def align_guassians_z(quaternions, scales):
    new_quaternions = np.copy(quaternions)
    mask_x = scales[:, 0] == np.min(scales, axis=1)
    mask_y = scales[:, 1] == np.min(scales, axis=1)
    mask_z = scales[:, 2] == np.min(scales, axis=1)

    new_quaternions[mask_z] = np.array([0.0, 0.0, 0.0, 1.0])
    new_quaternions[mask_x] = np.array([0.0, 0.7071068, 0.0, 0.7071068])
    new_quaternions[mask_y] = np.array([0.0, 0.0, 0.7071068, 0.7071068])

    new_scales = np.copy(scales)
    # this makes no sense, why does doubling the scale make the gaussian smaller?
    new_scales[mask_x, 0] *= 2
    new_scales[mask_y, 1] *= 2
    new_scales[mask_z, 2] *= 2

    return new_quaternions, new_scales


# def align_guassians_z(quaternions, scales):
#     new_quaternions = np.copy(quaternions)
#     r = Rotation.from_quat(quaternions)
#     mask_x = scales[:, 0] == np.min(scales, axis=1)
#     mask_y = scales[:, 1] == np.min(scales, axis=1)
#     mask_z = scales[:, 2] == np.min(scales, axis=1)

#     euler_x = r[mask_x].as_euler('xyz', degrees=False)
#     euler_y = r[mask_y].as_euler('yxz', degrees=False)
#     euler_z = r[mask_z].as_euler('zxy', degrees=False)

#     rotations_x = Rotation.from_euler('xyz', [euler_x[0], 0, 0], degrees=False)
#     rotations_y = Rotation.from_euler('yxz', [euler_y[1], 0, 0], degrees=False)
#     rotations_z = Rotation.from_euler('zxy', [euler_z[2], 0, 0], degrees=False)

#     new_quaternions[mask_x] = rotations_x.as_quat()
#     new_quaternions[mask_y] = rotations_y.as_quat()
#     new_quaternions[mask_z] = rotations_z.as_quat()
#     return new_quaternions


if __name__ == "__main__":
    input_file = "/home/mighty/Documents/point_cloud_original.ply"
    output_file = "/home/mighty/repos/gaussian-splatting/output/02-06_esszimmer_vlate/point_cloud/iteration_30000/point_cloud.ply"

    ply = PlyData.read(input_file)
    vertex = ply['vertex']
    positions = np.vstack((vertex['x'], vertex['y'], vertex['z'])).T
    quaternions = np.vstack((vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3'])).T
    scales = np.vstack((vertex['scale_0'], vertex['scale_1'], vertex['scale_2'])).T

    #filter all positions around y=16.1294 +- 0.1
    delta = 0.05
    z_plane = 10
    mask = np.logical_and(positions[:, 2] > z_plane - delta, positions[:, 2] < z_plane + delta)

    # TEST remove all vertices with the mask
    # vertex = ply['vertex'][~mask]

    quats, new_scales = align_guassians_z(quaternions[mask], scales[mask])
    quaternions[mask] = quats
    scales[mask] = new_scales
    positions[mask, 2] -= 0.9 * (positions[mask, 2] - z_plane)

    vertex['rot_0'] = quaternions[:, 0]
    vertex['rot_1'] = quaternions[:, 1]
    vertex['rot_2'] = quaternions[:, 2]
    vertex['rot_3'] = quaternions[:, 3]

    vertex['x'] = positions[:, 0]
    vertex['y'] = positions[:, 1]
    vertex['z'] = positions[:, 2]

    vertex['scale_0'] = scales[:, 0]
    vertex['scale_1'] = scales[:, 1]
    vertex['scale_2'] = scales[:, 2]

    ply_out = PlyData([vertex])
    ply_out.write(output_file)
