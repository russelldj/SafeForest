import os
from pathlib import Path

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
