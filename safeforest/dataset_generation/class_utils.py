import numpy as np
import time


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


def remap_classes_bool_indexing(
    input_classes: np.array, remap: np.array, background_value: int = 7
):
    """Change indices based on input

    https://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array
    """
    output = np.ones_like(input_classes) * background_value
    for i, v in enumerate(remap):
        mask = input_classes == i
        output[mask] = v
    return output


def remap_classes(input_classes: np.array, remap: np.array):
    """Change indices based on input

    https://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array
    """
    print("This function is slow, use remap_classes_bool_indexing")
    from_values = np.arange(len(remap))
    d = dict(zip(from_values, remap))

    input_shape = input_classes.shape
    input_classes = input_classes.flatten()

    out = [d[i] for i in input_classes]
    out = np.reshape(out, input_shape)
    return out
