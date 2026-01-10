import numpy as np
import cv2
import math
from scipy.sparse.linalg import cg, LinearOperator
from numba import njit, prange

def color2gray_spcr(img, mu=3, npi=1, dpi=4, alpha=20):
    """
    Optimized SPCR (SIGGRAPH 2005) Color2Gray using Numba for speed.
    img: HxWx3 RGB uint8
    returns: HxW uint8 grayscale
    """

    # Convert RGB → BGR for OpenCV
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    height, width, _ = img.shape
    pixels = img_lab.astype(int)

    # Extract L,a,b channels
    ps = pixels.reshape((-1, 3))
    l = (ps[:, 0] * 100 / 255).astype(np.float64)
    a = (ps[:, 1] - 128).astype(np.float64)
    b = (ps[:, 2] - 128).astype(np.float64)

    l_avg = l.mean()
    pixels = np.stack([l, a, b], axis=1).reshape((height, width, 3))

    theta = npi * math.pi / dpi
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # --- Precompute neighborhood sizes ---
    nneighb = np.zeros((height, width), dtype=np.int32)
    for i in range(height):
        for j in range(width):
            top = max(i - mu, 0)
            bot = min(i + mu, height - 1)
            left = max(j - mu, 0)
            right = min(j + mu, width - 1)
            nneighb[i, j] = (bot - top + 1) * (right - left + 1) - 1

    N = height * width
    B = np.zeros(N, dtype=np.float64)

    # --- Numba functions ---
    @njit
    def delta(p_i, p_j):
        da, db = p_i[1] - p_j[1], p_i[2] - p_j[2]
        dl = p_i[0] - p_j[0]
        dist_c = math.sqrt(da**2 + db**2)
        crunch = alpha * math.tanh(dist_c / alpha)
        if abs(dl) > crunch:
            return dl
        return crunch if (da * cos_t + db * sin_t) >= 0 else -crunch

    @njit(parallel=True)
    def build_B(pixels, mu):
        h, w, _ = pixels.shape
        B = np.zeros(h * w, dtype=np.float64)
        for i in prange(h):
            for j in range(w):
                idx = i * w + j
                top = max(i - mu, 0)
                bot = min(i + mu, h - 1)
                left = max(j - mu, 0)
                right = min(j + mu, w - 1)
                for ni in range(top, bot + 1):
                    for nj in range(left, right + 1):
                        if ni == i and nj == j:
                            continue
                        dij = delta(pixels[i, j], pixels[ni, nj])
                        dji = delta(pixels[ni, nj], pixels[i, j])
                        B[idx] += (dij - dji)
        return B

    B = build_B(pixels, mu)
    diagA = 2 * nneighb.flatten()

    # --- Optimized matvec ---
    @njit
    def matvec(x):
        x2d = x.reshape((height, width))
        out = diagA.reshape((height, width)) * x2d
        for i in range(height):
            for j in range(width):
                top = max(i - mu, 0)
                bot = min(i + mu, height - 1)
                left = max(j - mu, 0)
                right = min(j + mu, width - 1)
                for ni in range(top, bot + 1):
                    for nj in range(left, right + 1):
                        if ni == i and nj == j:
                            continue
                        out[i, j] -= 2 * x2d[ni, nj]
        return out.flatten()

    A = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
    g0 = l.copy()
    g, info = cg(A, B, x0=g0, maxiter=200, rtol=1e-3, atol=1e-4)

    # Normalize result
    g = g + (l_avg - g.mean())
    g = (g * 255 / 100).clip(0, 255)
    return g.reshape((height, width)).astype(np.uint8)
