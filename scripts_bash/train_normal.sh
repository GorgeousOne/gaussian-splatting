
# generate checkpoint to be able to test faster
# python ./train.py -s /home/mighty/repos/datasets/hah/esszimmer_small/ -m ./output/02-28_debug_normal --data_device=cpu

python ./train_single.py -s /home/mighty/repos/datasets/hah/esszimmer_small/ -n normals -m ./output/02-28_debug_normal --data_device=cpu -r2  --start_checkpoint=./output/02-28_debug_normal/chkpnt10000.pth --iterations=11000 --densify_until_iter=0 --position_lr_init=0 --position_lr_final=0 --scaling_lr=0 --feature_lr=0 --save_iterations 10010