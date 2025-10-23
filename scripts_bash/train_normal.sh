
# generate checkpoint to be able to test faster
# python ./train.py -s /home/mighty/repos/datasets/hah/esszimmer_small/ -m ./output/02-28_debug_normal --data_device=cpu

# python ./train_single.py -s /home/mighty/repos/datasets/hah/esszimmer_small/ -n normals -m ./output/02-28_debug_normal --data_device=cpu -r2  --start_checkpoint=./output/02-28_debug_normal/chkpnt10000.pth --iterations=11000 --densify_until_iter=0 --position_lr_init=0 --position_lr_final=0 --scaling_lr=0 --feature_lr=0 --save_iterations 10100

# python ./train.py -s /home/mighty/repos/datasets/hah/esszimmer_small/ -n normals -m ./output/03-18_20-30_normal_r4_step10 --data_device=cpu -r4 --save_iteration 5_000 10_000 20_000 30_000 --checkpoint_iterations 5_000 10_000 20_000 --normal_interval 10
# python ./train.py -s /home/mighty/repos/datasets/hah/esszimmer_small/ -n normals -m ./output/03-18_21-00_normal_r4_step1 --data_device=cpu -r4 --save_iteration 5_000 10_000 20_000 30_000 --checkpoint_iterations 5_000 10_000 20_000

python ./train.py -s /home/mighty/repos/datasets/hah/esszimmer_small/ -m ./output/03-18_21-00_r4 --data_device=cpu -r4 --save_iteration 15_000 30_000 --checkpoint_iterations 15_000
