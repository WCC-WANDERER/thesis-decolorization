import numpy as np
from skimage.filters import sobel
from skimage.feature import canny
from scipy.ndimage import gaussian_filter

def fsim(img1, img2):
    """
    FSIM (Feature Similarity Index)
    Simplified version suitable for grayscale comparisons.
    Images expected as float 0..1
    """

    # Sobel gradient magnitude
    G1 = sobel(img1)
    G2 = sobel(img2)

    # Phase congruency approximated via Canny edges (acceptable for FSIM-lite)
    PC1 = canny(img1, sigma=1)
    PC2 = canny(img2, sigma=1)

    # Combine gradient + phase congruency
    T1 = np.maximum(PC1, G1)
    T2 = np.maximum(PC2, G2)

    # FSIM similarity map
    R = np.minimum(T1, T2) / (np.maximum(T1, T2) + 1e-5)

    return np.mean(R)
