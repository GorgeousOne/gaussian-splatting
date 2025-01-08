'''Script to increase frequency of low frequency areas in an image using blender generated masks.'''
from PIL import Image
import numpy as np
import os

def apply_mask(image_path, mask_path, output_path):
	'''Apply an algorithm to the image at masked regions to reduce low frequency areas'''
	image = np.array(Image.open(image_path).convert('RGB'), dtype=np.uint8)
	mask_bool = np.array(Image.open(mask_path).convert('1'))

	# invert masked areas (bit heavy on the eyes)
	# image[mask_bool] = (256 - image[mask_bool])

	# slightly brighten masked areas
	image[mask_bool] = image[mask_bool] + 16
	Image.fromarray(image).save(output_path)

image_folder = 'test_imgs'
mask_folder = 'masks'
output_folder = 'test_masked_imgs'

if not os.path.exists(output_folder):
	os.makedirs(output_folder)

for image_path in os.listdir(image_folder):
	img_name = os.path.splitext(image_path)[0]
	if image_path.endswith('.jpg'):
		image_path = os.path.join(image_folder, image_path)
		mask_path = os.path.join(mask_folder, f'mask_{img_name}.png')
		output_path = os.path.join(output_folder, f'masked_{img_name}.jpg')
		apply_mask(image_path, mask_path, output_path)
