import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray

def c2g_ssim(color_img, gray_img):
    """
    C2G-SSIM: Color-to-Gray Structural SIMilarity
    color_img: H x W x 3 float (0..1)
    gray_img : H x W or H x W x 1 float (0..1)
    """

    # Convert to float and ensure shapes
    color = color_img.astype(np.float64)

    gray = gray_img.astype(np.float64)
    if gray.ndim == 3:
        gray = gray[:, :, 0]   # force 2D grayscale

    # Convert color to luminance using standard ITU-R BT.601
    lum = rgb2gray(color)

    # Constants from Zhao et al.
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)

    # Gaussian smoothing
    mu_l = gaussian_filter(lum, sigma=1)
    mu_g = gaussian_filter(gray, sigma=1)

    sigma_l = gaussian_filter(lum * lum, sigma=1) - mu_l * mu_l
    sigma_g = gaussian_filter(gray * gray, sigma=1) - mu_g * mu_g
    sigma_lg = gaussian_filter(lum * gray, sigma=1) - mu_l * mu_g

    # C2G-SSIM formula
    ssim_map = ((2 * mu_l * mu_g + C1) * (2 * sigma_lg + C2)) / \
               ((mu_l * mu_l + mu_g * mu_g + C1) *
                (sigma_l + sigma_g + C2))

    return float(np.clip(np.mean(ssim_map), 0, 1))
    #return np.clip(np.mean(ssim_map), 0, 1)
