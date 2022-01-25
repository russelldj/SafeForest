from typing import Union

import numpy as np
import distinctipy
from skimage import measure

BLACK = np.array([0, 0, 0])


def find_unique_colors(segmentation_im: np.array):
    """Get the unique colors in an image.

    Args:
        segmentation_mask: (H, W, 3) np.array where colors encode the classes
    
    Returns:
        A (H, 3) np.array where each row is a unique color.

    """
    if len(segmentation_im.shape) != 3:
        raise ValueError(
            f"Segmentation mask should be dimension 3 but was {len(segmentation_im.shape)}"
        )
    if segmentation_im.shape[2] != 3:
        raise ValueError(
            f"Segmentation mask should have 3 channels but had {segmentation_im.shape[2]}"
        )
    segmentation_im = np.reshape(segmentation_im, (-1, 3))
    unique_colors = np.unique(segmentation_im, axis=0)
    return unique_colors


def create_instance_mask(
    segmentation_im: np.array, num_colors: Union[int, None] = None
):
    """Creates uniquely colored instances from segmentation.

    For each class label, each disjoint region gets its own color label.

    Args:
        segmentation_mask: (H, W, 3) np.array where colors encode the classes

    Returns:
        A (H, W, 3) np.array encoding the distinct regions with different colors.
    """
    segmentation_im_shape = segmentation_im.shape
    unique_colors = find_unique_colors(segmentation_im)

    # Ignore the background by setting it to label 0
    ID_counter = 0
    ID_array = np.zeros(segmentation_im_shape[:2], dtype=np.uint32)

    for color in unique_colors:
        # Exclude the background class
        if np.all(color == BLACK):
            continue

        color = np.array([[color]])
        matching = color == segmentation_im
        matching = np.alltrue(matching, axis=2)
        labels, num_regions = measure.label(matching, connectivity=1, return_num=True)
        ID_array[matching] = labels[matching] + ID_counter
        ID_counter += num_regions

    distinct_colors = np.load("vis/unique_colors.npy")
    # distinct_colors = np.array(distinctipy.get_colors(ID_counter + 1))

    if np.any(np.all(np.array([BLACK]) == distinct_colors, axis=1)):
        # Switch black to the first row if present
        black_row = np.where(np.all(np.array([BLACK]) == distinct_colors, axis=1))[0][0]
        first_color = distinct_colors[0]
        distinct_colors[0] = BLACK
        distinct_colors[black_row] = first_color
    else:
        distinct_colors[0] = BLACK

    # This might be really slow
    remapped = distinct_colors[ID_array]

    return remapped
