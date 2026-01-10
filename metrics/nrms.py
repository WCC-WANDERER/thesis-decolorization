import numpy as np

def nrms(original_rgb, gray):
    """
    Correct NRMS implementation according to literature:
    
    NRMS = 1/3 * Σ_C ||I_C - Y|| / ||I_C||
    
    where:
      - I_C is channel R,G,B from original image
      - Y is grayscale image
      - ||·|| is Frobenius norm
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
