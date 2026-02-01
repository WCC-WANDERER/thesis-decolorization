import numpy as np

def nrms(original_rgb, gray):
    """
    NRMS: Normalized Root Mean Square
    original_rgb: H x W x 3 float (0..1)
    gray: H x W or H x W x 1 float (0..1)
    returns: float value of average Frobenius norm difference across color channels
    """

    # Ensure float
    I = original_rgb.astype(np.float32)
    Y = gray.astype(np.float32)

    nrms_sum = 0.0

    for c in range(3):             # R,G,B
        Ic = I[:, :, c]
        diff = Ic - Y

        num = np.linalg.norm(diff)   # Frobenius
        den = np.linalg.norm(Ic)     # Frobenius

        if den == 0:
            continue

        nrms_sum += (num / den)

    return nrms_sum / 3.0
