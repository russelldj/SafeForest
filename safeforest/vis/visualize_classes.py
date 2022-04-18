import numpy as np
import time


def visualize_with_palette(index_image, palette, ignore_ind=255):
    """
    index_image : np.ndarray
        The predicted semantic map with indices. (H,W)
    palette : np.ndarray
        The colors for each index. (N classes,3)
    """
    h, w = index_image.shape
    index_image = index_image.flatten()

    dont_ignore = index_image != ignore_ind
    output = np.zeros((index_image.shape[0], 3))
    colored_image = palette[index_image[dont_ignore]]
    output[dont_ignore] = colored_image
    colored_image = np.reshape(output, (h, w, 3))
    return colored_image.astype(np.uint8)


def blend_images(im1, im2, alpha=0.7):
    return (alpha * im1 + (1 - alpha) * im2).astype(np.uint8)


def blend_images_gray(im1, im2, alpha=0.7):
    """Blend two images with the first transformed to grayscale

    im1: img to be turned to gray
    im2: img kept as normal color
    alpha: contribution of first image
    """
    num_channels = im1.shape[2]
    im1 = np.mean(im1, axis=2)
    im1 = np.expand_dims(im1, axis=2)
    im1 = np.repeat(im1, repeats=num_channels, axis=2)
    return (alpha * im1 + (1 - alpha) * im2).astype(np.uint8)
