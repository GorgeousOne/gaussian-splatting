'''Script for exporting a blender scene to obj with UVs scaled ~1:1 to the world scale.
This can be used to create a scene with mostly unfiformly sized checker pattern.'''

import bpy
import bmesh
import math
import os

def scale_uvs_to_match_geometry(mesh_obj):
	'''Scales the UV map so that the average UV island matches the world scale.'''
	# Get mesh data
	mesh = mesh_obj.data
	bm = bmesh.new()
	bm.from_mesh(mesh)

	# Ensure UV layer exists
	uv_layer = bm.loops.layers.uv.active
	if uv_layer is None:
		print(f'Skipping {mesh_obj.name}, no UV data found.')
		bm.free()
		return

	total_face_area = 0.0
	total_uv_area = 0.0

	# Iterate over faces to calculate areas
	for face in bm.faces:
		# 3D geometry area
		face_area = face.calc_area()
		total_face_area += face_area
		total_uv_area += get_uv_area(face, uv_layer)
	# print(total_face_area, total_uv_area)

	# Avoid division by zero
	if total_face_area == 0 or total_uv_area == 0:
		print(f'Skipping {mesh_obj.name}, invalid face or UV area.')
		bm.free()
		return

	# Compute scale factor
	scale_factor = math.sqrt(total_face_area / total_uv_area)

	# Scale UVs
	for face in bm.faces:
		for loop in face.loops:
			loop[uv_layer].uv = loop[uv_layer].uv * scale_factor

	# Apply changes
	bm.to_mesh(mesh)
	bm.free()

def apply_modifiers(obj):
	bpy.context.view_layer.objects.active = obj
	obj.select_set(True)
	bpy.ops.object.transform_apply()
	obj.select_set(False)


def get_uv_area(face, uv_layer):
	total_uv_area = 0.0
	# UV area (Shoelace formula)
	if len(face.loops) >= 3:
		uv_coords = [loop[uv_layer].uv for loop in face.loops]
		uv_area = 0.5 * abs(sum(
			uv_coords[i].x * uv_coords[i - 1].y - uv_coords[i - 1].x * uv_coords[i].y
			for i in range(len(uv_coords))
		))
		total_uv_area += uv_area
	return total_uv_area

def create_checker_mtl(mtl_filepath):
	'''creates a obj mtl file with a checker texture'''
	with open(mtl_filepath, 'w', encoding='utf-8') as f:
		f.write('newmtl placeholder\n')
		f.write('Ka 1.000 1.000 1.000\n')
		f.write('Kd 0.000 0.000 0.000\n')
		f.write('Ks 0.000 0.000 0.000\n')
		f.write('d 1.0\n')
		f.write('illum 2\n')
		f.write('map_Kd placeholder.png\n')

def add_checker_mtl_to_obj(obj_filepath, mtl_name):
	'''adds "usemtl placeholder" to every object in the obj file'''
	with open(obj_filepath, 'r', encoding='utf-8') as f:
		lines = f.readlines()
	with open(obj_filepath, 'w', encoding='utf-8') as f:
		for line in lines:
			f.write(line)
			if line.startswith('o '):
				f.write(f'usemtl {mtl_name}\n')

def export_checker_obj(blend_filepath, obj_filepath):
	bpy.ops.wm.open_mainfile(filepath=blend_filepath)

	for obj in bpy.data.objects:
		if obj.type == 'MESH':
			try:
				apply_modifiers(obj)
			except Exception as ignored:
				pass
			scale_uvs_to_match_geometry(obj)

	bpy.ops.wm.obj_export(
		filepath=obj_filepath,
		export_normals=False,
		export_materials=False)

	obj_name = os.path.basename(obj_filepath).split('.')[0]
	obj_dir = os.path.dirname(obj_filepath)
	create_checker_mtl(os.path.join(obj_dir, f'{obj_name}.mtl'))
	add_checker_mtl_to_obj(obj_filepath, 'placeholder')

if __name__ == "__main__":
	blend_filepath = '/home/mighty/Documents/blender/bedroom.blend'
	obj_filepath = '/home/mighty/Documents/blender/wavefront/bedroom.obj'
	export_checker_obj(blend_filepath, obj_filepath)
