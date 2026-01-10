import numpy as np
from skimage.transform import resize

def corrc2g(img, r=256):
    # img: uint8 RGB (H,W,3)
    img = img.astype(np.float64) / 255.0
    h, w, _ = img.shape

    f = r / min(h, w)
    f = min(f, 1)

    if f < 1:
        img_small = resize(img, (int(h * f), int(w * f)), order=0, preserve_range=True)
    else:
        img_small = img

    Mu = img_small.mean(axis=2)

    x = img_small - Mu[..., None]
    Sigma = np.sqrt((np.abs(x) ** 2).sum(axis=2) / 2.0) / 0.5774

    # --- First contrast map Q1 ---
    Q1 = Mu * Sigma

    d1 = Q1 - Q1.mean()
    dr = img_small[..., 0] - img_small[..., 0].mean()
    dg = img_small[..., 1] - img_small[..., 1].mean()
    db = img_small[..., 2] - img_small[..., 2].mean()

    Rho1 = np.array([
        (d1 * dr).sum() / np.sqrt((d1**2).sum() * (dr**2).sum()),
        (d1 * dg).sum() / np.sqrt((d1**2).sum() * (dg**2).sum()),
        (d1 * db).sum() / np.sqrt((d1**2).sum() * (db**2).sum()),
    ])

    # --- Second contrast map Q2 ---
    Q2 = Mu * (1 - Sigma)
    d1b = Q2 - Q2.mean()

    Rho2 = np.array([
        (d1b * dr).sum() / np.sqrt((d1b**2).sum() * (dr**2).sum()),
        (d1b * dg).sum() / np.sqrt((d1b**2).sum() * (dg**2).sum()),
        (d1b * db).sum() / np.sqrt((d1b**2).sum() * (db**2).sum()),
    ])

    # normalize weights
    def compute_lambda(Rho):
        den = Rho.max() - Rho.min()
        Gamma = np.zeros_like(Rho) if den < 1e-12 else (Rho - Rho.min()) / den - 0.5
        #Gamma = (Rho - Rho.min()) / (Rho.max() - Rho.min()) - 0.5
        beta = np.abs(Rho)
        beta = beta / beta.sum()
        lam = beta + np.minimum(beta, Gamma)
        lam = np.abs(lam)
        lam = lam / lam.sum()
        return lam

    lambda1 = compute_lambda(Rho1)
    lambda2 = compute_lambda(Rho2)

    # compute two candidate grayscale images
    out1 = (img_small[...,0] * lambda1[0] +
            img_small[...,1] * lambda1[1] +
            img_small[...,2] * lambda1[2])

    out2 = (img_small[...,0] * lambda2[0] +
            img_small[...,1] * lambda2[1] +
            img_small[...,2] * lambda2[2])

    def mat2gray(x):
        mn, mx = x.min(), x.max()
        if mx > mn:
            return (x - mn) / (mx - mn)
        return np.zeros_like(x)

    out1 = mat2gray(out1)
    out2 = mat2gray(out2)

    # histogram selection (fast version)
    idx = Sigma != 0
    hist1, _ = np.histogram(out1[idx], bins=11, range=(0,1))
    hist2, _ = np.histogram(out2[idx], bins=11, range=(0,1))

    k1 = np.sum(hist1[2:9] > 1.05 * hist2[2:9])
    k2 = np.sum(hist2[2:9] > 1.05 * hist1[2:9])

    use1 = (k1 >= k2) or (hist1[2:9].sum() * k1 > 1.5 * hist2[2:9].sum() * k2)

    chosen_lambda = lambda1 if use1 else lambda2

    # ALWAYS compute final output in *original resolution*
    gray = (img[...,0] * chosen_lambda[0] +
            img[...,1] * chosen_lambda[1] +
            img[...,2] * chosen_lambda[2])

    # normalize to 0–255
    mn, mx = gray.min(), gray.max()
    gray = (gray - mn) / (mx - mn + 1e-12)
    gray = (gray * 255).astype(np.uint8)

    return gray
