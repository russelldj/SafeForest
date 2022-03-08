import numpy as np


def stretch_to_unit(data):
    """Rescale to be in the range (0, 1)
    """
    data_min = np.min(data)
    data_max = np.max(data)
    extent = data_max - data_min
    data = (data - data_min) / extent
    return data
