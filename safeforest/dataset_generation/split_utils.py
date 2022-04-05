import numpy as np


def get_is_train_array(num_total: int, num_train: int, *, seed=None, shift: int = 0):
    """
    """
    if seed is not None:
        print(f"Warning: setting numpy random seed to {seed}")
        np.random.seed(seed)
    values = np.random.permutation(num_total)
    values = np.concatenate((values[shift:], values[:shift]))

    is_train_array = values < num_train
    return is_train_array
