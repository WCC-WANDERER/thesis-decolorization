import numpy as np

def rms(gray):
    """
    RMS: Root Mean Square
    gray: H x W or H x W x 1 float (0..1)
    returns: float value of the standard deviation of pixel intensities
    """
    mean = np.mean(gray)
    return np.sqrt(np.mean((gray - mean)**2))