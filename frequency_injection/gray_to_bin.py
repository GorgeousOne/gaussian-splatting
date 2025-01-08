'''Script to convert blenders rendered images from RGB to binary masks.'''
from PIL import Image
import os
from tqdm import tqdm
import shutil

def convert_images_to_binary(input_dir, output_dir):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	for filename in tqdm(os.listdir(input_dir)):
		if filename.endswith('.png'):
			img_path = os.path.join(input_dir, filename)
			img = Image.open(img_path).convert('1')
			img.save(os.path.join(output_dir, filename))


if __name__ == '__main__':
	input_directory = '/home/mighty/Documents/blender/renders'
	output_directory = 'masks'
	if not os.path.exists(output_directory):
		os.makedirs(output_directory)
	# time converseion
	convert_images_to_binary(input_directory, output_directory)

	# delete input dir
	shutil.rmtree(input_directory)
