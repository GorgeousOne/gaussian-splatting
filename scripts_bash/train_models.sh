#hah with one voxel prune in different iterations
python ./train.py -s /home/mighty/repos/datasets/hah/esszimmer_small/ -m ./output/02-06_esszimmer_vlate --data_device=cpu --voxel /home/mighty/repos/datasets/hah/obj/hah_occupancy_thin.npz --voxel_iterations 29000
# python ./train.py -s /home/mighty/repos/datasets/hah/esszimmer_small/ -m ./output/02-06_esszimmer_vmid --data_device=cpu --voxel /home/mighty/repos/datasets/hah/obj/hah_occupancy_thin.npz --voxel_iterations 15000
# python ./train.py -s /home/mighty/repos/datasets/hah/esszimmer_small/ -m ./output/02-06_esszimmer_vearly --data_device=cpu --voxel /home/mighty/repos/datasets/hah/obj/hah_occupancy_thin.npz --voxel_iterations 1000
# python ./train.py -s /home/mighty/repos/datasets/hah/esszimmer_small/ -m ./output/02-06_esszimmer_vnever --data_device=cpu


#test out before%after views
# python ./train.py -s /home/mighty/repos/datasets/hah/esszimmer_small/ -m ./output/02-06_15_00_esszimmer_vthick_test --data_device=cpu --voxel /home/mighty/repos/datasets/hah/obj/hah_occupancy_thick.npz --voxel_iterations $(seq 1000 1000 10001) --save_iterations 10000 --iterations 10000


#haus am horn with differnt voxel types
# python ./train.py -s /home/mighty/repos/datasets/hah/esszimmer_small/ -m ./output/02-05_19_00_esszimmer_vthin --data_device=cpu --voxel /home/mighty/repos/datasets/hah/obj/hah_occupancy_thin.npz --voxel_iterations $(seq 1000 1000 30001)
# python ./train.py -s /home/mighty/repos/datasets/hah/esszimmer_small/ -m ./output/02-05_20_00_esszimmer_vthick --data_device=cpu --voxel /home/mighty/repos/datasets/hah/obj/hah_occupancy_thick.npz --voxel_iterations $(seq 1000 1000 30001)

#synthetic bedrrom with differnt voxel types
# python ./train.py -s /home/mighty/Documents/blender/bedroom3 -m ./output/02-05_17-00_bedrrom_vthin --eval --data_device=cpu --voxel /home/mighty/Documents/blender/bedroom2/obj_occupancy_thin.npz --voxel_iterations $(seq 1000 1000 30001)