import numpy as np
import cv2
import matplotlib.pyplot as plt

rgba = cv2.imread("/home/mighty/repos/datasets/hah/esszimmer_small/test_depth.png", cv2.IMREAD_UNCHANGED)
inv_depth_map = rgba.view(np.float32).reshape(rgba.shape[0], rgba.shape[1])

eps = 1e-6  # Small value to avoid division by zero
depth_map = 1.0 / (inv_depth_map + eps)

plt.figure(figsize=(10, 8))
plt.imshow(depth_map, cmap='plasma', interpolation='nearest')
plt.colorbar(label="Depth (meters)")
plt.title("Metric Depth Map Visualization")
plt.xlabel("Image Width (pixels)")
plt.ylabel("Image Height (pixels)")
plt.show()