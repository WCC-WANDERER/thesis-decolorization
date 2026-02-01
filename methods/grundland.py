import numpy as np
from .decolorize import decolorize   # import the GitHub implementation

def grundland_decolor(np_img):
    """
    Input: np_img = HxWx3 uint8
    Output: gray = HxW uint8 from Grundland & Dodgson
    """

    # decolorize() expects filename OR an RGB float image, therefore:
    # Temporarily save the image to float image in [0..1]
    img_float = np_img.astype(np.float32) / 255.0

    # Call the original algorithm
    G = decolorize(img_float)   # (H,W,3) float [0..1]

    # Convert to uint8 0..255
    Gray_uint8 = (G * 255).clip(0, 255).astype(np.uint8)

    return Gray_uint8
