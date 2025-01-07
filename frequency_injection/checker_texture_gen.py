'''Script to replace textures in a MTL file with checker patterns of the same size.'''
import os
import re
import shutil
import numpy as np
from PIL import Image

# TODO downscale checker textures to 1x1 tile patterns if all all future textures are going to be multiple of 16x16
def generate_checker_texture(size:int, dst_dir:str):
	filename = f'checker_{size}.png'
	dst_path = os.path.join(dst_dir, filename)

	if os.path.exists(dst_path):
		print(f'Checker texture already exists "{dst_path}"')
		return dst_path

	# create PIL image
	mat = np.array([[((i + j) % 2) for j in range(size)] for i in range(size)])
	img = Image.fromarray(np.uint8(mat * 255), 'L').convert('1')
	img.save(dst_path)
	print(f'Generated  checker texture "{dst_path}"')
	return dst_path

def get_img_size(img_path):
	"""get the image size without loading it completely into memory"""
	if not os.path.exists(img_path):
		return -1, -1
	with Image.open(img_path) as img:
		return img.size

def replace_mtl_textures_with_checker(mtl_path):
	if not os.path.exists(mtl_path):
		print(f"Error: MTL file '{mtl_path}' not found.")
		return

	# backup mtl file before altering it
	mtl_dir = os.path.dirname(mtl_path)
	mtl_backup_path = mtl_path.replace('.', '_unmodified_backup.')
	shutil.copy(mtl_path, os.path.join(mtl_path, mtl_backup_path))

	with open(mtl_path, 'r') as file:
		lines = file.readlines()

	updated_lines = []
	texture_pattern = r"map_Kd\s+(.+)"  # Common texture maps

	for line in lines:
		match = re.match(texture_pattern, line.strip())
		if match:
			texture_path = match.group(1)
			size, _ = get_img_size(texture_path)
			size = size if size > 0 else 1600
			checker_path = generate_checker_texture(size // 16, mtl_dir)
			line = line.replace(texture_path, checker_path)
			print(f"Replacing texture: {texture_path} -> {checker_path}")

		updated_lines.append(line)

	with open(mtl_path, 'w') as file:
		file.writelines(updated_lines)

	print("Texture replacement complete.")

if __name__ == "__main__":
	pass
	mtl_file_path = "/home/mighty/Documents/blender/wavefront/bedroom.mtl"
	replace_mtl_textures_with_checker(mtl_file_path)