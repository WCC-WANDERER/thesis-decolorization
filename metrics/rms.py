import numpy as np

def rms(gray):
    mean = np.mean(gray)
    return np.sqrt(np.mean((gray - mean)**2))