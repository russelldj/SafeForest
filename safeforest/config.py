import os
from pathlib import Path
import numpy as np

try:
    from local_config import DATA_REPO
except ImportError:
    DATA_REPO = Path("../Safe")
    print("Could not find local_config")

LABELS_INFO = [
    {
        "hasInstances": False,
        "category": "sky",
        "catid": 0,
        "name": "sky",
        "ignoreInEval": False,
        "id": 0,
        "color": [0, 0, 0],
        "trainId": 0,
    },
    {
        "hasInstances": False,
        "category": "ground",
        "catid": 1,
        "name": "soil",
        "ignoreInEval": False,
        "id": 1,
        "color": [111, 74, 0],
        "trainId": 1,
    },
    {
        "hasInstances": False,
        "category": "ground",
        "catid": 1,
        "name": "trails",
        "ignoreInEval": False,
        "id": 2,
        "color": [81, 0, 81],
        "trainId": 2,
    },
    {
        "hasInstances": False,
        "category": "vegatation",
        "catid": 2,
        "name": "tree canopy",
        "ignoreInEval": False,
        "id": 3,
        "color": [128, 64, 128],
        "trainId": 3,
    },
    {
        "hasInstances": False,
        "category": "vegatation",
        "catid": 2,
        "name": "fuel",
        "ignoreInEval": False,
        "id": 4,
        "color": [244, 35, 232],
        "trainId": 4,
    },
    {
        "hasInstances": False,
        "category": "vegatation",
        "catid": 2,
        "name": "trunks",
        "ignoreInEval": False,
        "id": 5,
        "color": [250, 170, 160],
        "trainId": 5,
    },
    {
        "hasInstances": False,
        "category": "vegatation",
        "catid": 2,
        "name": "stumps",
        "ignoreInEval": False,
        "id": 6,
        "color": [0, 170, 160],
        "trainId": 6,
    },
]
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
        [0, 0, 255],
        [150, 75, 0],
        [140, 146, 172],
        [0, 255, 0],
        [255, 0, 0],
        [255, 0, 255],
        [255, 0, 128],
    ]
)

RUI_CLASSES = ("sky", "soil", "trails", "canopy", "fuel", "trunks", "stumps")

RUI_YAMAHA_CLASSES = (
    "sky",
    "traversable_ground",
    "traversable_vegetation",
    "untraversable_vegetation",
    "obstacle",
    "trunk",
)
RUI_YAMAHA_PALETTE = np.array(
    [
        [42, 125, 209],
        [128, 129, 131],
        [52, 209, 183],
        [13, 53, 26],
        [250, 50, 83],
        [184, 61, 245],
    ]
)


RGB_EXT = "_rgb"
SEG_EXT = "_seg"
IMG_DIR = "img_dir"
ANN_DIR = "ann_dir"
TRAIN_DIR = "train"
VAL_DIR = "val"
