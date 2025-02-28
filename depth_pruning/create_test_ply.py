import numpy as np
from plyfile import PlyData, PlyElement
import os

class Gaussian:
	"""Represents a single 3D Gaussian primitive"""
	def __init__(self, position, scale, color, rotation=None, opacity=1.0):
		"""
		Args:
			position (array-like): xyz coordinates [x, y, z]
			scale (array-like): scale factors [sx, sy, sz]
			color (array-like): RGB color [r, g, b], values 0-1
			rotation (array-like, optional): quaternion [w, x, y, z]
			opacity (float, optional): opacity value 0-1
		"""
		self.position = np.array(position, dtype=np.float32)
		self.scale = np.array(scale, dtype=np.float32)
		self.color = np.array(color, dtype=np.float32)

		# Default rotation is identity quaternion
		if rotation is None:
			self.rotation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
		else:
			# Normalize quaternion
			rotation = np.array(rotation, dtype=np.float32)
			norm = np.linalg.norm(rotation)
			self.rotation = rotation / norm if norm > 0 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

		self.opacity = float(opacity)

		# Initialize empty SH coefficients (will be computed if needed)
		self.sh_coefficients = None


def create_line_gaussians(start, end, step, point_scale, color=None):
	"""
	Create Gaussians arranged in a line with color gradient

	Args:
		start: [x, y, z] start point
		end: [x, y, z] end point
		color: RGB color
	Returns:
		List of Gaussian objects
	"""
	gaussians = []

	# Generate random colors if not provided
	if color is None:
		color = np.random.uniform(0, 1, 3)

	start = np.array(start, dtype=np.float32)
	end = np.array(end, dtype=np.float32)
	dir = end - start
	length = np.linalg.norm(dir)
	dir /= length
	print(int(length / step), "line")
	for i in range(0, int(length / step)):
		# Create a Gaussian
		position = start + dir * i * step
		scale = point_scale
		gaussian = Gaussian(position, scale, color)
		gaussians.append(gaussian)
	return gaussians


def create_quad_gaussians(start, left, right, step, point_scale=[0.1, 0.1, 0.05], color=None):
	"""
	Create Gaussians arranged in a 3D grid
	Args:
		step: distance between guassians
		point_scale: scale of each Gaussian
		color: RGB color for all points or None for random colors

	Returns:
		List of Gaussian objects
	"""
	gaussians = []
	size_left = np.linalg.norm(left)
	size_right = np.linalg.norm(right)
	norm_left = left / size_left * step
	norm_right = right / size_right * step
	num_points_left = int(size_left / step)
	num_points_right = int(size_right / step)
	print(num_points_left, num_points_right, num_points_left * num_points_right, "grid")
	if color is None:
		color = np.random.uniform(0, 1, 3)
	for i in range(num_points_left):
		for j in range(num_points_right):
			# create gaussian
			position = start + i * norm_left + j * norm_right
			scale = point_scale
			gaussian = Gaussian(position, scale, color)
			gaussians.append(gaussian)

	return gaussians


def save_gaussians_to_ply(gaussians, path, include_sh_coefficients=False):
	"""
	Save a list of Gaussian objects to a PLY file
	Args:
		gaussians: List of Gaussian objects
		path: Output path for the PLY file
		include_sh_coefficients: Whether to include spherical harmonic coefficients
	"""
	# Create directory if it doesn't exist
	os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

	num_gaussians = len(gaussians)

	# Extract properties into arrays
	xyz = np.array([g.position for g in gaussians])
	normals = np.zeros_like(xyz)  # Typically not used
	colors = np.array([g.color for g in gaussians])
	opacities = np.array([[g.opacity] for g in gaussians])
	scales = np.array([g.scale for g in gaussians])
	rotations = np.array([g.rotation for g in gaussians])

	# Initialize arrays for SH coefficients if needed
	if include_sh_coefficients:
		# For this example, assume we use 3 DC components (RGB) and 12 rest components
		# (4 spherical harmonic bands × 3 color channels beyond the DC term)
		sh_rest_size = 12
		sh_rest = np.zeros((num_gaussians, sh_rest_size), dtype=np.float32)
	else:
		# If not using SH coefficients, just use the base colors as DC components
		sh_rest_size = 0

	# Construct list of attribute names
	attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz']

	# Add DC feature names (colors)
	for i in range(3):  # RGB
		attributes.append(f'f_dc_{i}')

	# Add rest feature names (SH coefficients)
	if include_sh_coefficients:
		for i in range(sh_rest_size):
			attributes.append(f'f_rest_{i}')

	# Add opacity
	attributes.append('opacity')

	# Add scaling factors
	for i in range(scales.shape[1]):
		attributes.append(f'scale_{i}')
	# Add rotation components
	for i in range(rotations.shape[1]):
		attributes.append(f'rot_{i}')

	# Create structured dtype for the PLY file
	dtype_full = [(attribute, 'f4') for attribute in attributes]

	# Concatenate all attributes into a single array
	if include_sh_coefficients:
		all_attributes = np.concatenate((xyz, normals, colors, sh_rest, opacities, scales, rotations), axis=1)
	else:
		all_attributes = np.concatenate((xyz, normals, colors, opacities, scales, rotations), axis=1)

	# Create the structured array
	elements = np.empty(num_gaussians, dtype=dtype_full)
	elements[:] = list(map(tuple, all_attributes))

	# Create the PLY element and data
	el = PlyElement.describe(elements, 'vertex')
	ply_data = PlyData([el])

	# Write to file
	ply_data.write(path)

	print(f"Saved {num_gaussians} Gaussians to: {path}")
	print(f"Format: {', '.join(attributes)}")

	return path

def create_wireframe_box(start, size, step=0.2, p=0.01):
	line_x00 = create_line_gaussians(
		start=start, end=start + np.array([size[0], 0, 0]),
		step=step, point_scale=[p/2, p, p], color=[1, 0, 0]
	)
	line_x10 = create_line_gaussians(
		start=start + np.array([0, size[1], 0]), end=start + np.array([size[0], size[1], 0]),
		step=step, point_scale=[p/2, p, p], color=[0, 1, 1]
	)
	line_x01 = create_line_gaussians(
		start=start + np.array([0, 0, size[2]]), end=start + np.array([size[0], 0, size[2]]),
		step=step, point_scale=[p/2, p, p], color=[0, 1, 1]
	)
	line_x11 = create_line_gaussians(
		start=start + np.array([0, size[1], size[2]]), end=start + np.array([size[0], size[1], size[2]]),
		step=step, point_scale=[p/2, p, p], color=[0, 1, 1]
	)

	line_y00 = create_line_gaussians(
		start=start, end=start + np.array([0, size[1], 0]),
		step=step, point_scale=[p, p/2, p], color=[0, 1, 0]
	)
	line_y10 = create_line_gaussians(
		start=start + np.array([size[0], 0, 0]), end=start + np.array([size[0], size[1], 0]),
		step=step, point_scale=[p, p/2, p], color=[1, 0, 1]
	)
	line_y01 = create_line_gaussians(
		start=start + np.array([0, 0, size[2]]), end=start + np.array([0, size[1], size[2]]),
		step=step, point_scale=[p, p/2, p], color=[1, 0, 1]
	)
	line_y11 = create_line_gaussians(
		start=start + np.array([size[0], 0, size[2]]), end=start + np.array([size[0], size[1], size[2]]),
		step=step, point_scale=[p, p/2, p], color=[1, 0, 1]
	)

	line_z00 = create_line_gaussians(
		start=start, end=start + np.array([0, 0, size[2]]),
		step=step, point_scale=[p, p, p/2], color=[0, 0, 1]
	)
	line_z10 = create_line_gaussians(
		start=start + np.array([size[0], 0, 0]), end=start + np.array([size[0], 0, size[2]]),
		step=step, point_scale=[p, p, p/2], color=[1, 1, 0]
	)
	line_z01 = create_line_gaussians(
		start=start + np.array([0, size[1], 0]), end=start + np.array([0, size[1], size[2]]),
		step=step, point_scale=[p, p, p/2], color=[1, 1, 0]
	)
	line_z11 = create_line_gaussians(
		start=start + np.array([size[0], size[1], 0]), end=start + np.array([size[0], size[1], size[2]]),
		step=step, point_scale=[p, p, p/2], color=[1, 1, 0]
	)
	return line_x00 + line_y00 + line_z00
	return \
		line_x00 + line_x10 + line_x01 + line_x11 + \
		line_y00 + line_y10 + line_y01 + line_y11 + \
		line_z00 + line_z10 + line_z01 + line_z11


def create_standing_1(start, s, step=1, p=0.01):
	dash_1 = create_line_gaussians(
		start=start,
		end=start + np.array([0, 0, s]),
		step=step,
		point_scale=[p/2, p, p],
		color=[1, 1, 1]
	)
	dash2 = create_line_gaussians(
		start=start + np.array([0, 0, s]),
		end=start + np.array([0, -0.4*s, 0.6*s]),
		step=step,
		point_scale=[p/2, p, p],
		color=[1, 1, 1]
	)
	return dash_1 + dash2

def create_lying_F(start, s, step=1, p=0.01):
	dash_1 = create_line_gaussians(
		start=start,
		end=start + np.array([s, 0, 0]),
		step=step,
		point_scale=[p, p, p/2],
		color=[1, 1, 1]
	)
	dash_2 = create_line_gaussians(
		start=start + np.array([0, 0, 0]),
		end=start + np.array([0, .6*s, 0]),
		step=step,
		point_scale=[p, p, p/2],
		color=[1, 1, 1]
	)
	dash_3 = create_line_gaussians(
		start=start + np.array([.4*s, 0, 0]),
		end=start + np.array([.4*s, .5*s, 0]),
		step=step,
		point_scale=[p, p, p/2],
		color=[1, 1, 1]
	)
	return dash_1 + dash_2 + dash_3

if __name__ == "__main__":
	# box_gaussians = create_box([10, 10, 10], step=0.2)
	wire = create_wireframe_box([-50, -50, -50], [100, 100, 100], step=5, p=0.01)
	shape_1 = create_standing_1([-50, 0, -30], s=60)
	shape_f = create_lying_F([-30, -20, -50], s=60)
	all_gaussians = wire + shape_1 + shape_f
	save_gaussians_to_ply(all_gaussians, "/home/mighty/repos/gaussian-splatting/output/02-06_esszimmer_vlate/point_cloud/iteration_30000/point_cloud.ply")