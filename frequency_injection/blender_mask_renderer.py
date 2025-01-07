'''Scipt to create and render a blender file with an obj model of a 3D reconstrction and colmap camera data.
With checker textures to create masks for exsisting images.'''
import bpy
# import read_write_model as rwm
import numpy as np
import mathutils
import os
from tqdm import tqdm

def load_obj(obj_path):
	bpy.ops.wm.obj_import(filepath=obj_path)
	update_materials()

def update_materials():
	for mat in bpy.data.materials:
		if not mat.node_tree:
			continue
		bsdf_node = None
		for node in mat.node_tree.nodes:
			if node.type == 'BSDF_PRINCIPLED':
				bsdf_node = node
			# disable texture interpolation for sharp edges
			if node.type == 'TEX_IMAGE':
				node.interpolation = 'Closest'
		links = mat.node_tree.links

		if not bsdf_node:
			continue
		# for i, o in enumerate(bsdf_node.inputs):
			# print(i, o)
		# plug the base color texture into emission instead
		for link in links:
			if link.to_node == bsdf_node and link.to_socket.name == 'Base Color':
				image_texture_node = link.from_node
				links.new(image_texture_node.outputs['Color'], bsdf_node.inputs['Emission Color'])
				bsdf_node.inputs['Emission Strength'].default_value = 20.0
				links.remove(link)
				break


def create_cam(scene, name:str, transformation_mat, lens:int=36):
	#TODO remove redundant camera models I think,
	cam = bpy.data.cameras.new(name)
	cam.lens = lens

	cam_obj = bpy.data.objects.new(name, cam)
	cam_obj.matrix_world = mathutils.Matrix(tuple(map(tuple, transformation_mat)))
	scene.collection.objects.link(cam_obj)


def colmap_to_blender_focal(f_colmap, pixel_w, sensor_w):
	'''
	Convert COLMAP focal length (in pixels) to Blender focal length (in mm).
	Args:
		f_colmap (float): Focal length from COLMAP (in pixels)
		pixel_w (int): Image width (in pixels)
		sensor_w (float): Sensor width (in mm, typically 36mm for full-frame cameras
	Returns:
		float: Equivalent Blender focal length (in mm)
	'''
	return f_colmap * (sensor_w / pixel_w)


def load_cameras(images, cameras, sensor_w=36):
	scene = bpy.context.scene

	for key, image_meta in images.items():
		cam_intrinsic = cameras[image_meta.camera_id]
		# get pinhole camera focal length
		f_x = cam_intrinsic.params[0]
		w = cam_intrinsic.width
		h = cam_intrinsic.height

		c2w_rot = rwm.qvec2rotmat(image_meta.qvec).T
		c2w_t = c2w_rot @ -image_meta.tvec
		c2w_mat = np.eye(4)
		c2w_mat[:3, :3] = c2w_rot
		c2w_mat[:3, 3] = c2w_t

		lens = colmap_to_blender_focal(f_x, w, sensor_w)
		img_name = image_meta.name.split('.')[0]
		create_cam(scene, img_name, c2w_mat, lens)

	scene.render.resolution_x = w
	scene.render.resolution_y = h

def set_render_settings():
	scene = bpy.context.scene
	scene.render.image_settings.color_mode = 'BW'
	scene.render.image_settings.color_depth = '8'
	scene.render.engine = 'BLENDER_EEVEE_NEXT'
	# render pixels in only black/white
	scene.eevee.taa_render_samples = 1


def render_cams(output_dir):
	scene = bpy.context.scene
	i = 0

	for cam in tqdm([obj for obj in scene.objects if obj.type == 'CAMERA']):
		scene.camera = cam
		bpy.context.scene.render.filepath = os.path.join(output_dir, f'mask_{cam.name}')
		bpy.ops.render.render(write_still=True)


def main(blender_filepath, obj_filepath, renders_dir='//renders', force=False):
	if os.path.exists(blender_filepath) and not force:
		bpy.ops.wm.open_mainfile(filepath=blender_filepath)
	else:
		bpy.ops.wm.read_factory_settings(use_empty=True)
		set_render_settings()
		load_obj(obj_filepath)

		# cam_intrinsics, images_metas, points3d = rwm.read_model('./playroom/sparse/0', ext='.bin')
		# load_cameras(images_metas, cam_intrinsics)
		bpy.ops.wm.save_as_mainfile(filepath=blender_filepath)
	# render_cams(renders_dir)

# blender -b --python
if __name__ == '__main__':
	blender_filepath = '/home/mighty/Documents/blender/wavefront/masking.blend'
	obj_filepath = '/home/mighty/Documents/blender/wavefront/bedroom.obj'
	main(blender_filepath, obj_filepath)
