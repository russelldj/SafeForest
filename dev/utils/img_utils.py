import numpy

KERNEL = numpy.array([-1, 2, -1])


def lap2_focus_measure(image, mask):
    """
    Arguments:
        image: (N, M) greyscale, floating point from 0-255
        mask: (N, M) boolean array, True where we want to extract values

    Code written by Eric Schneider in the CalibrADE project:
    https://github.com/russelldj/CalibrADE/blob/main/calibrade/prune.py

    """
    lx = numpy.convolve(image.flatten(), KERNEL, mode="same").reshape(image.shape)
    ly = numpy.convolve(image.T.flatten(), KERNEL, mode="same").reshape(image.T.shape)
    raw = numpy.sum(numpy.abs(lx[mask]) + numpy.abs(ly[mask.T]))

    # vvvv Diverges from the paper vvvv

    # Because we are doing lap2 over a mask, we need to normalize by the size
    # of the mask! Otherwise large blurry things will have higher lap2 than
    # small sharp things
    number_norm = raw / numpy.sum(mask)
    # This feels kind of hand-wavy, but I noticed that dark images have a less
    # pronounced scores even if they look sharper to the human eye. When I
    # passed image/2 into the function, the score I got was 1/2 as much. Let's
    # scale by average brightness
    bright_norm = number_norm * (255 / numpy.average(image[mask]))
    # Remove nans
    no_nan = numpy.nan_to_num(bright_norm, nan=0.0)

    return no_nan
