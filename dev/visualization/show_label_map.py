import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from safeforest.config import PALETTE_MAP, CLASS_MAP


def show_label_colors(names, palette, title=None, savepath=None):
    if title is None:
        title == ""

    assert len(names) == len(palette)
    num_cols = int(np.ceil(len(names) / 2))
    fig, axs = plt.subplots(2, num_cols)
    for i in range(len(names)):
        name = names[i]
        color = palette[i]
        color = np.expand_dims(np.expand_dims(color, axis=0), axis=1)
        axs[i // num_cols, i % num_cols].imshow(color)
        axs[i // num_cols, i % num_cols].set_title(name, fontsize=20)
    fig.suptitle(title, fontsize=20)
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--title")
    parser.add_argument("--savepath", type=Path)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    classes = CLASS_MAP[args.dataset]
    palette = PALETTE_MAP[args.dataset]
    show_label_colors(classes, palette, args.title, args.savepath)
