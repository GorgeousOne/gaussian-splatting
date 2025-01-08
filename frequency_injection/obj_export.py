'''Script for exporting a blender scene to obj with UVs scaled ~1:1 to the world scale.
This can be used to create a scene with mostly unfiformly sized checker pattern.'''

import bpy
import bmesh
import math
import os

def unify_uv_scales():
	for obj in bpy.data.objects:
		if not obj.type == 'MESH':
			continue
		try:
			apply_modifiers(obj)
		except Exception as ignored:
			pass
		scale_uvs_to_match_geometry(obj)


def apply_modifiers(obj):
	bpy.context.view_layer.objects.active = obj
	obj.select_set(True)
	bpy.ops.object.transform_apply()
	obj.select_set(False)


def scale_uvs_to_match_geometry(mesh_obj):
	'''Scales the UV map so that the average UV island matches the world scale.'''
	# Get mesh data
	mesh = mesh_obj.data
	bm = bmesh.new()
	bm.from_mesh(mesh)
	# Ensure UV layer exists
	uv_layer = bm.loops.layers.uv.active
	if uv_layer is None:
		try:
			print(f'Unwrapping {mesh_obj.name}...')
			unwrap_mesh(mesh_obj)
		except Exception as e:
			print(f'Skipping {mesh_obj.name}, no UV could be generated: {e}')
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


def unwrap_mesh(mesh_obj):
	'''Unwraps the mesh object.'''
	bpy.context.view_layer.objects.active = mesh_obj
	mesh = mesh_obj.data
	bpy.ops.object.mode_set(mode='EDIT')
	bpy.ops.mesh.select_all(action='SELECT')
	bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)
	bpy.ops.object.mode_set(mode='OBJECT')

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


def replace_mats_with_checker(checker_filepath):
	if not os.path.exists(checker_filepath):
		raise FileNotFoundError(f"Checker texture not found at path '{checker_filepath}'.")

	for mat in bpy.data.materials:
		bpy.data.materials.remove(mat)

	new_mat = bpy.data.materials.new(name="CheckerMaterial")
	new_mat.use_nodes = True

	# Add an image texture node
	nodes = new_mat.node_tree.nodes
	bsdf = nodes.get("Principled BSDF")

	tex_node = nodes.new(type="ShaderNodeTexImage")
	tex_node.image = bpy.data.images.load(checker_filepath)
	# make texture interpolation sharp
	tex_node.interpolation = 'Closest'

	# Link texture to the BSDF node as emission
	new_mat.node_tree.links.new(tex_node.outputs['Color'], bsdf.inputs['Emission Color'])
	bsdf.inputs['Emission Strength'].default_value = 20.0
	bsdf.inputs['Base Color'].default_value = (0, 0, 0, 1)

	for obj in bpy.data.objects:
		if obj.type == "MESH":
			obj.data.materials.clear()
			obj.data.materials.append(new_mat)


def remove_lights():
	for obj in bpy.data.objects:
		if obj.type == 'LIGHT':
			bpy.data.objects.remove(obj)


def set_world_black():
	world = bpy.data.worlds.new(name="Black_World")
	bpy.context.scene.world = world
	world.use_nodes = True

	tree = world.node_tree
	tree.nodes.clear()

	bg_node = tree.nodes.new(type='ShaderNodeBackground')
	bg_node.inputs[0].default_value = (0, 0, 0, 1)

	output_node = tree.nodes.new(type='ShaderNodeOutputWorld')

	tree.links.new(bg_node.outputs[0], output_node.inputs[0])


if __name__ == "__main__":
	blend_filepath = '/home/mighty/Documents/blender/bedroom.blend'
	checker_filepath = '/home/mighty/Documents/blender/wavefront/checker_100.png'

	bpy.ops.wm.open_mainfile(filepath=blend_filepath)
	unify_uv_scales()
	replace_mats_with_checker(checker_filepath)
	remove_lights()
	set_world_black()

	# create new filename with _checker suffix
	output_file = blend_filepath.replace(".blend", "_masked.blend")
	bpy.ops.wm.save_as_mainfile(filepath=output_file)
