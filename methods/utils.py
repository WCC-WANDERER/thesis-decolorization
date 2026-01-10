import cv2
import os

def load_image(path):
    """Loads image as RGB numpy array."""
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def save_gray(img_gray, path):
    """Saves grayscale numpy array as PNG."""
    cv2.imwrite(path, img_gray)
