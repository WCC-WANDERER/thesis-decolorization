import numpy as np

def decolorize(img, scale=None, effect=.5, noise=.001):
    """
    img: either filename (str) OR numpy array HxWx3 float [0..1]
    Returns: decolorized grayscale image HxW float [0..1]
    """
    import matplotlib.image as mpimg

    # if img is string, read it
    if isinstance(img, str):
        RGB = mpimg.imread(img)
        if RGB.dtype != 'float32':
            if np.min(RGB) >= 0 and np.max(RGB) <= 255:
                RGB = RGB.astype(float)/255.0
    else:
        RGB = img.copy()
    
    dims = RGB.shape
    
    # handle grayscale or alpha channel
    if len(dims) == 2:  # already grayscale
        return RGB
    if dims[2] == 4:  # remove alpha
        RGB = RGB[:, :, :3]
        dims = RGB.shape

    # --- original algorithm ---
    RGB_flat = RGB.transpose(2,1,0).reshape(dims[2], dims[0]*dims[1])
    Weights = np.array([
        [0.2989360212937753847527155, 0.5870430744511212909351327, 0.1140209042551033243121518],
        [.5, .5, -1.0],
        [1.0, -1.0, 0.0]
    ])
    YPQ = Weights.dot(RGB_flat)
    Y, P, Q = map(np.squeeze, np.split(YPQ, 3, axis=0))

    Lmax = 1
    Lscale = 0.6685679342408883
    Smax = 1.1180339887498948
    alter = effect*(Lmax/Smax)
    Ch = np.sqrt(P**2 + Q**2)

    mesh = np.meshgrid(range(dims[0]), range(dims[1]))
    mesh = np.dstack([mesh[0], mesh[1]]).reshape(-1, 2)
    if scale is None:
        scale = np.sqrt(2*min(dims[0:2]))
    displace = scale * np.sqrt(2/np.pi) * np.random.normal(size=[dims[0]*dims[1],2])
    look = np.round(mesh + displace)
    look[:,0] = np.clip(look[:,0], 0, dims[0]-1)
    look[:,1] = np.clip(look[:,1], 0, dims[1]-1)
    look = (look[:,0] + dims[0]*look[:,1]).astype(int)

    delta = YPQ - YPQ[:,look]
    contrast_change = np.abs(delta[0,:])
    contrast_dir = np.sign(delta[0,:])
    color_diff = np.sqrt(np.sum((RGB_flat - RGB_flat[:,look])**2, axis=0)) + np.finfo(float).eps
    w = 1 - contrast_change/Lscale / color_diff
    w[color_diff < 1e-14] = 0
    axis = w * contrast_dir
    axis = np.multiply(delta[1:3, :], np.array([axis, axis]))
    axis = np.sum(axis, axis=1)
    
    proj = YPQ[1,:]*axis[0] + YPQ[2,:]*axis[1]
    proj = proj / (np.quantile(np.abs(proj), 1-noise) + 1e-14)

    # final grayscale
    L = Y
    C = effect*proj
    G = L + C
    img_range = np.quantile(G, [noise, 1-noise])
    G = (G - img_range[0]) / (img_range[1]-img_range[0]+1e-14)
    tgt_range = effect*np.array([0,Lmax]) + (1-effect)*np.quantile(YPQ[0,:],[noise,1-noise])
    G = tgt_range[0] + G*(tgt_range[1]-tgt_range[0]+1e-14)
    G = np.clip(G, L - alter*Ch, L + alter*Ch)
    G = np.clip(G, 0, Lmax)

    # reshape to original HxW
    G = G.reshape(dims[1], dims[0]).T

    return G
