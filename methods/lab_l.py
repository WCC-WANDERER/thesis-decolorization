from skimage.color import rgb2lab
import numpy as np

def lab_l_gray(img):
    """
    L* channel from LAB color space
    LAB L is in range 0..100 so we rescale to 0..255
    img: numpy RGB
    """
    lab = rgb2lab(img)
    L = lab[:, :, 0]  # 0..100
    L_scaled = (L / 100 * 255).astype(np.uint8)
    return L_scaled
