import numpy as np
from scipy import spatial
import cv2


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


def imwrite_ocv(filename, img):
    """Take an RGB or RGBA image and write it using OpenCV"""
    shape = img.shape
    filename = str(filename)
    if len(shape) == 3 and shape[2] == 3:
        # RGB
        img = np.flip(img, axis=2)
    elif len(shape) == 3 and shape[2] == 4:
        # RGBA
        img = np.stack((np.flip(img[..., :3], axis=2), img[..., 3:4]), axis=2)
    elif len(shape) == 2:
        pass
    else:
        raise ValueError()

    cv2.imwrite(filename, img)
