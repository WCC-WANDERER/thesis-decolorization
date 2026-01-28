# ccpr_ccfr.py
import numpy as np
from skimage.color import rgb2lab

def ccpr(color_img, gray_img, tau_gray=0.03, tau_lab=7.0, q=1/3.0):
    """
    Contrast-Preserving Pixel Ratio (CCPR)
    """
    lab = rgb2lab(color_img)
    gray = np.squeeze(gray_img)

    # Horizontal differences
    lab_diff_h = np.sqrt(
        q * (lab[:, :-1, 0] - lab[:, 1:, 0])**2 +
        (lab[:, :-1, 1] - lab[:, 1:, 1])**2 +
        (lab[:, :-1, 2] - lab[:, 1:, 2])**2
    )
    gray_diff_h = np.abs(gray[:, :-1] - gray[:, 1:])
    
    # Vertical differences
    lab_diff_v = np.sqrt(
        q * (lab[:-1, :, 0] - lab[1:, :, 0])**2 +
        (lab[:-1, :, 1] - lab[1:, :, 1])**2 +
        (lab[:-1, :, 2] - lab[1:, :, 2])**2
    )
    gray_diff_v = np.abs(gray[:-1, :] - gray[1:, :])

    # Horizontal CCPR
    mask_h = lab_diff_h >= tau_lab
    preserved_h = np.sum((gray_diff_h >= tau_gray) & mask_h)
    total_h = np.sum(mask_h)

    # Vertical CCPR
    mask_v = lab_diff_v >= tau_lab
    preserved_v = np.sum((gray_diff_v >= tau_gray) & mask_v)
    total_v = np.sum(mask_v)

    if total_h + total_v == 0:
        return 0.0
    return (preserved_h + preserved_v) / (total_h + total_v)


def ccfr(color_img, gray_img, tau_gray=0.03, tau_lab=7.0, q=1/3.0):
    """
    Color Content Fidelity Ratio (CCFR)
    """
    lab = rgb2lab(color_img)
    gray = np.squeeze(gray_img)

    # Calculate color differences (Lab) and gray differences
    # Horizontal
    lab_diff_h = np.sqrt(q * (lab[:, :-1, 0] - lab[:, 1:, 0])**2 + (lab[:, :-1, 1] - lab[:, 1:, 1])**2 + (lab[:, :-1, 2] - lab[:, 1:, 2])**2)
    gray_diff_h = np.abs(gray[:, :-1] - gray[:, 1:])
    # Vertical
    lab_diff_v = np.sqrt(q * (lab[:-1, :, 0] - lab[1:, :, 0])**2 + (lab[:-1, :, 1] - lab[1:, :, 1])**2 + (lab[:-1, :, 2] - lab[1:, :, 2])**2)
    gray_diff_v = np.abs(gray[:-1, :] - gray[1:, :])

    # Define Theta: Smooth pairs in the original color image
    is_smooth_h = lab_diff_h <= tau_lab
    is_smooth_v = lab_diff_v <= tau_lab

    # Identify Artifacts: Smooth in color but visible edge in gray (> tau)
    bad_h = np.sum(is_smooth_h & (gray_diff_h > tau_gray))
    bad_v = np.sum(is_smooth_v & (gray_diff_v > tau_gray))

    # Denominator: Total number of smooth pairs (Theta)
    total_theta = np.sum(is_smooth_h) + np.sum(is_smooth_v)

    if total_theta == 0:
        return 1.0
    
    # Final CCFR calculation
    return 1.0 - (bad_h + bad_v) / total_theta
