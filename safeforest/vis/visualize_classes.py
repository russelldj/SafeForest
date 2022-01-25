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
