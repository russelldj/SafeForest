import numpy as np
from scipy import spatial


def convert_colors_to_indices(img: np.ndarray, palette: np.ndarray):
    """ 

    img: (h, w, 3|4) input image of color masks
    palette: (n, 3) Ordered colors in the colormap
    """
    img = img[..., :3]
    im_shape = img.shape
    img = img.reshape((-1, 3))
    dist = spatial.distance.cdist(img, palette)
    indices = np.argmin(dist, axis=1)
    label_image = indices.reshape(im_shape[:2]).astype(np.uint8)
    return label_image
