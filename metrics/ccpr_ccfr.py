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

    # Horizontal CCFR
    mask_h = gray_diff_h > tau_gray
    bad_h = np.sum((lab_diff_h <= tau_lab) & mask_h)
    total_h = np.sum(mask_h)

    # Vertical CCFR
    mask_v = gray_diff_v > tau_gray
    bad_v = np.sum((lab_diff_v <= tau_lab) & mask_v)
    total_v = np.sum(mask_v)

    if total_h + total_v == 0:
        return 1.0
    return 1.0 - (bad_h + bad_v) / (total_h + total_v)
