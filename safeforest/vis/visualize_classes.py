import numpy as np


def visualize_with_palette(index_image, palette):
    """
    index_image : np.ndarray
        The predicted semantic map with indices. (H,W)
    palette : np.ndarray
        The colors for each index. (N classes,3)
    """
    h, w = index_image.shape
    index_image = index_image.flatten()
    colored_image = palette[index_image]
    colored_image = np.reshape(colored_image, (h, w, 3))
    return colored_image.astype(np.uint8)


def blend_images(im1, im2, alpha=0.7):
    return (alpha * im1 + (1 - alpha) * im2).astype(np.uint8)


def blend_images_gray(im1, im2, alpha=0.7):
    num_channels = im1.shape[2]
    im1 = np.mean(im1, axis=2)
    im1 = np.expand_dims(im1, axis=2)
    im1 = np.repeat(im1, repeats=num_channels, axis=2)
    return (alpha * im1 + (1 - alpha) * im2).astype(np.uint8)
