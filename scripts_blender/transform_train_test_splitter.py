import json
import os

def split_transforms(input_file, train_file, test_file, test_step):
	with open(input_file, 'r') as f:
		data = json.load(f)

	train_data = {"camera_angle_x": data["camera_angle_x"], "frames": []}
	test_data = {"camera_angle_x": data["camera_angle_x"], "frames": []}

	for i, frame in enumerate(data["frames"]):
		if (i + 1) % test_step == 0:
			test_data["frames"].append(frame)
		else:
			train_data["frames"].append(frame)

	with open(train_file, 'w') as f:
		json.dump(train_data, f, indent=4)

	with open(test_file, 'w') as f:
		json.dump(test_data, f, indent=4)

	print(f"Data split into {len(train_data['frames'])} train frames, {len(test_data['frames'])} test frames")


base_dir = '/home/mighty/Documents/blender/bedroom3'

# Example usage
split_transforms(
	os.path.join(base_dir, 'transforms.json'),
	os.path.join(base_dir, 'transforms_train.json'),
	os.path.join(base_dir, 'transforms_test.json'), 10)
