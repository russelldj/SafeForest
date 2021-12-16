import os
from pathlib import Path
import numpy as np

try:
    from local_config import DATA_REPO
except ImportError:
    DATA_REPO = Path("../Safe")
    print("Could not find local_config")

YAMAHA_PALETTE = np.array(
    [
        [0, 160, 0],
        [1, 88, 255],
        [40, 80, 0],
        [128, 255, 0],
        [156, 76, 30],
        [178, 176, 153],
        [255, 0, 0],
        [255, 255, 255],
    ]
)

YAMAHA_CLASSES = (
    "non-traversable low vegetation",
    "sky",
    "high vegetation",
    "traversable grass",
    "rough trail",
    "smooth trail",
    "obstacle",
    "truck",
)

RUI_PALETTE = np.array(
    [
        [107, 190, 118],
        [137, 249, 83],
        [235, 244, 247],
        [113, 44, 216],
        [214, 27, 52],
        [60, 206, 22],
        [18, 5, 149],
    ]
)

RUI_CLASSES = ("sky", "soil", "trails", "canopy", "fuel", "trunks", "stumps")
