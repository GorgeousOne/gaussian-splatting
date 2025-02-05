
# python ./train.py -s /home/mighty/Documents/blender/bedroom3 -m ./output/debug_feb4 --eval --data_device=cpu --voxel /home/mighty/Documents/blender/bedroom2/occupancy_grid.npz --voxel_iterations $(seq 100 100 300)

python ./train.py -s /home/mighty/repos/datasets/hah/esszimmer_small/ -m ./output/02-05_18_00_esszimmer_v --eval --data_device=cpu --voxel /home/mighty/repos/datasets/hah/obj/hah_occupancy.npz --voxel_iterations $(seq 1000 1000 30001)
# python ./train.py -s /home/mighty/repos/datasets/hah_esszimmer2/ -m ./output/hah_01-28_18_00 --eval --data_device=cpu --voxel /home/mighty/repos/datasets/hah_esszimmer2/occupancy_grid_low.npz --voxel_iterations $(seq 5000 1000 24001)
# python ./train.py -s /home/mighty/Documents/blender/bedroom3 -m ./output/bedroom_01-29_15-00 --eval --data_device=cpu --voxel /home/mighty/Documents/blender/bedroom2/occupancy_grid.npz --voxel_iterations $(seq 5000 1000 25001)