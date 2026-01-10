import numpy as np

def average_gray(img):
    """
    Gray = (R + G + B) / 3
    img: numpy array in RGB format (H, W, 3)
    """
    return np.mean(img, axis=2).astype(np.uint8)
