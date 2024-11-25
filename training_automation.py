import argparse
import os
import subprocess
import sys

from datetime import datetime


def list_eval_imgs(imgs_dir, output_file, eval_step):
    img_names = sorted(os.listdir(imgs_dir))
    eval_imgs = [img_names[i] for i in range(eval_step, len(img_names), eval_step)]
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(eval_imgs)


def gen_depths(imgs_dir, working_dir, output_dir):
    conda_env = 'depth_anything'
    sub_program = f'python ./Depth-Anything-V2/run.py --encoder vitl --pred-only --grayscale --img-path {imgs_dir} --outdir {output_dir}'
    # run depth anything via bash? to simulate interactive shell with different configurations? cant get it to work otherwise
    command = f'bash -i -c "conda run -n {conda_env} {sub_program}"'
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=working_dir,
            check=True,  # Raise exception if the command fails
            text=True,   # Capture output as text (not bytes)
            capture_output=True,
        )
        if result.stderr:
            print("Script errors:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        print(f"Output: {e.stdout}")
        print(f"Errors: {e.stderr}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='/home/mighty/repos/datasets/db/playroom')
    parser.add_argument('--eval_step', type=int, default=5, help='use every nth image as eval image')
    parser.add_argument('--depth_reg', action='store_true', default=False)
    parser.add_argument('--exp_comp', action='store_true', default=False)
    args = parser.parse_args()

    base_dir = args.base_dir
    model_name = base_dir.split('/')[-1]
    imgs_dir = os.path.join(base_dir, 'images')
    colmap_dir = os.path.join(base_dir, 'sparse/0') 
    depths_dir = os.path.join(base_dir, 'depths') 

    eval_file = os.path.join(colmap_dir, 'test.txt')
    training_args = [
        f'-s {base_dir}',
        '--eval'
    ]

    if not os.path.exists(eval_file):
        list_eval_imgs(base_dir, os.path.join(colmap_dir, 'test.txt'), args.eval_step)
    
    if args.depth_reg:
        training_args.append(f'-d {depths_dir}')
        model_name += '_depth'
        if not os.path.exists(depths_dir):
            gen_depths(imgs_dir, '/home/mighty/repos', depths_dir)
    
    if args.exp_comp:
        model_name += '_exposure'
        training_args.extend([
            '--exposure_lr_init 0.001',
            '--exposure_lr_final 0.0001',
            '--exposure_lr_delay_steps 5000',
            '--exposure_lr_delay_mult 0.001',
            '--train_test_exp'])
    
    model_name += datetime.now().strftime("_%m-%d_%H-%M")
    model_output_dir = f'./output/{model_name}'
    training_args.append(f'-m {model_output_dir}')

    if os.path.exists(model_output_dir):
        sys.exit(f"Error: The file '{model_output_dir}' already exists. Exiting to prevent overwriting.")
    
    command = f'bash -i -c "conda run -n gaussian_splatting python ./train.py {" ".join(training_args)}"'
    print('Start training: ' + command)

    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,  # Raise exception if the command fails
            text=True,   # Capture output as text (not bytes)
            stdout=subprocess.PIPE,  # Redirect stdout to a pipe
            stderr=subprocess.PIPE   # Redirect stderr to a pipe
        )
        if result.stderr:
            print("Script errors:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        print(f"Output: {e.stdout}")
        print(f"Errors: {e.stderr}")



