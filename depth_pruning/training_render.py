import numpy as np
import cv2

def show_images_side_by_side(img1, img2, depth, window_name='Comparison'):
    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()
    # depth = depth.detach().cpu().numpy()

    # Assuming img1 and img2 are [C, H, W] and C=3
    img1 = np.transpose(img1, (1, 2, 0))
    img2 = np.transpose(img2, (1, 2, 0))
    # depth = np.transpose(depth, (1, 2, 0))

    # Convert to uint8
    img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img1 = img1[:, :, [2, 1, 0]] # RGB -> BGR conversion
    img2 = img2[:, :, [2, 1, 0]]


    # depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)
    # combined_image = np.concatenate((img1, img2, depth_color), axis=1)
    combined_image = np.concatenate((img1, img2), axis=1)
    cv2.imshow(window_name, combined_image)
    cv2.waitKey(1)  # This is necessary to ensure the window responds, 1ms should not block