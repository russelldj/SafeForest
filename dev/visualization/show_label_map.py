import numpy as np
import matplotlib.pyplot as plt
from config import LABELS_INFO

from config import (
    RUI_CLASSES,
    RUI_PALETTE,
    YAMAHA_CLASSES,
    YAMAHA_PALETTE,
    RUI_YAMAHA_CLASSES,
    RUI_YAMAHA_PALETTE,
)

np.random.seed(123)
palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)


def show_label_colors(names, palette, title, savepath=None):
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


ids = [x["trainId"] for x in LABELS_INFO]
names = [x["name"] for x in LABELS_INFO]

names = [
    "trunks",
    "soil",
    "trails",
    "fuel",
    "tree canopy",
    "sky",
    "stumps",
]


names = RUI_CLASSES
palette = RUI_PALETTE

show_label_colors(RUI_CLASSES, RUI_PALETTE, "Rui labelmaps", "vis/rui_labelmap.png")
show_label_colors(
    YAMAHA_CLASSES, YAMAHA_PALETTE, "Yamaha labelmaps", "vis/yamaha_labelmap.png"
)
show_label_colors(
    RUI_YAMAHA_CLASSES,
    RUI_YAMAHA_PALETTE,
    "Rui-Yamaha merged labelmaps",
    "vis/rui_yamaha_labelmap.png",
)
