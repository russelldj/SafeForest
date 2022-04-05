import json
from pathlib import Path
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from safeforest.config import SUPER_ANNOTATE_PALETTE, REMAP_SUPER_ANNOTATE_TO_SEMFIRE
from safeforest.dataset_generation.img_utils import convert_colors_to_indices
from safeforest.dataset_generation.file_utils import write_cityscapes_file
from safeforest.dataset_generation.class_utils import (
    remap_classes,
    remap_classes_bool_indexing,
)

print(SUPER_ANNOTATE_PALETTE)


def hex_to_rgb(hex):
    return tuple(int(hex[i + 1 : i + 3], 16) for i in (0, 2, 4))


FILE = Path("/home/frc-ag-1/Downloads/UAV_Semantics/bag_4_7_0001.jpg___pixel.json")
FOLDER = Path("/home/frc-ag-1/Downloads/UAV_semantics_paired")
OUTPUT_FOLDER = Path(
    "/home/frc-ag-1/data/SafeForestData/datasets/portugal_UAV_12_21/derived/safe_forest_2/superannotate_training"
)
classes_file = Path(FOLDER, "classes", "classes.json")
class_info = json.load(open(classes_file, "r"))
print(class_info)

class_color_map = {}
for info in class_info:
    class_color_map[hex_to_rgb(info["color"])] = info["name"]

print(class_color_map.keys())
print(class_color_map.values())

files = sorted(FOLDER.glob("bag_4_7*"))

fig, axs = plt.subplots(1, 2)
index = 0
for i in range(0, len(files), 4):
    img = imread(files[i])
    label = imread(files[i + 1])
    if np.sum(label[..., :3]) == 0:
        continue
    axs[0].imshow(img)

    indices = convert_colors_to_indices(label, SUPER_ANNOTATE_PALETTE)
    remapped_indices = remap_classes_bool_indexing(
        indices, REMAP_SUPER_ANNOTATE_TO_SEMFIRE
    )

    # Don't include an image that's just background classes
    if np.all(remapped_indices == 7):
        continue

    axs[1].imshow(
        remapped_indices, vmin=0, vmax=SUPER_ANNOTATE_PALETTE.shape[0],
    )
    plt.pause(1)
    is_train = index >= 3
    write_cityscapes_file(
        remapped_indices, OUTPUT_FOLDER, index, is_ann=True, is_train=is_train
    )
    write_cityscapes_file(img, OUTPUT_FOLDER, index, is_ann=False, is_train=is_train)
    index += 1

