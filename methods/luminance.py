import numpy as np

def luminance_gray(img):
    """
    Y = 0.299 R + 0.587 G + 0.114 B  (NTSC / Rec.601)
    img: numpy array in RGB format
    """
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    return Y.astype(np.uint8)
