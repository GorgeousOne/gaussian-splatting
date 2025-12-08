'''render logs as graph'''
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv('./output/11-02_17-00_bedroom_normals_r1_just_debug/logs.csv')

# Metrics that exist for both test and train
base_metrics = ['l1_render', 'ssim_render', 'psnr_render', 'lpips_render', 'l1_normal']

fig, axes = plt.subplots(len(base_metrics), 1, figsize=(10, 12), sharex=True)

for ax, m in zip(axes, base_metrics):
    test_col = f'test_{m}'
    train_col = f'train_{m}'

    if test_col in df.columns and train_col in df.columns:
        ax.plot(df['iteration'], df[test_col], label=f'Test {m}', linewidth=1.5)
        ax.plot(df['iteration'], df[train_col], label=f'Train {m}', linestyle='--', linewidth=1.2)
        ax.set_ylabel(m)
        ax.grid(True)
        ax.legend(loc='upper right')

axes[-1].set_xlabel('Iteration')
fig.suptitle('Train vs Test Metrics Over Iterations', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
