import numpy as np
import cv2
from enum import Enum

class RenderMode(Enum):
    IMAGE = 0
    DEPTH = 1
    NORMAL = 2

def show_images(img_mode_pairs, window_name='Images'):
    """
    Display images side by side based on (image, mode) pairs.

    Args:
        img_mode_pairs: List of tuples (image, mode) where mode is RenderMode
        window_name: Name for the display window
    """
    processed_images = []

    for img, mode in img_mode_pairs:
        # Convert from tensor if needed
        img = img.detach().cpu().numpy()
        # Handle [C, H, W] format
        img = np.transpose(img, (1, 2, 0))

        # Process based on mode
        if mode == RenderMode.NORMAL:
            img = (img * 128 + 128).astype(np.uint8)
        elif mode == RenderMode.DEPTH:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            img = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
        else:  # RenderMode.IMAGE
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # RGB to BGR for OpenCV
        # if len(img.shape) == 3 and img.shape[2] == 3:
        img = img[:, :, [2, 1, 0]]
        processed_images.append(img)

    # Concatenate horizontally
    combined_image = np.concatenate(processed_images, axis=1)
    # Display
    cv2.imshow(window_name, combined_image)
    return cv2.waitKey(1)
    # while True:
    #     # wait for esc key to close window
    #     if cv2.waitKey(1) & 0xFF == 27:
    #         break
    # cv2.destroyAllWindows()
