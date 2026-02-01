import numpy as np
from skimage.filters import sobel_h, sobel_v
from skimage.color import rgb2gray

def grr(color_img, gray_img):
    """
    GRR: Gradient Recall Ratio
    color_img: H x W x 3 float (0..1)
    gray_img : H x W float (0..1)
    returns: float value of ratio of preserved gradient energy from color to grayscale
    """

    # Ensure float
    color = color_img.astype(np.float64)
    gray = gray_img.astype(np.float64)

    # Compute x- and y-gradients for grayscale image
    Yx = sobel_h(gray)
    Yy = sobel_v(gray)
    grad_gray_energy = np.sum(Yx**2 + Yy**2)

    # Compute x- and y-gradients for each channel of color image
    grad_color_energy = 0.0
    for c in range(3):
        Ic = color[:, :, c]
        Ix = sobel_h(Ic)
        Iy = sobel_v(Ic)
        grad_color_energy += np.sum(Ix**2 + Iy**2)

    # Avoid division by zero
    if grad_color_energy == 0:
        return 0.0

    return grad_gray_energy / grad_color_energy
