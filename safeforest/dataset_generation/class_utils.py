import numpy as np


def combine_classes(first_image, second_image, remap):
    """
    remap : np.ndarray
        An integer array where the element at the ith, jth location represents the class
        when a pixel in the first image has value i and the pixel in the second image has
        value j.
    """
    img_shape = first_image.shape
    first_image, second_image = [x.flatten() for x in (first_image, second_image)]
    remapped = remap[first_image, second_image]
    remapped = np.reshape(remapped, img_shape)
    return remapped
