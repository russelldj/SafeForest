import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from safeforest.config import REMAP_SUPER_ANNOTATE_TO_SEMFIRE, SUPER_ANNOTATE_PALETTE
from safeforest.dataset_generation.class_utils import (
    remap_classes,
    remap_classes_bool_indexing,
)
from safeforest.dataset_generation.file_utils import (
    ensure_dir_normal_bits,
    write_cityscapes_file,
)
from safeforest.dataset_generation.img_utils import convert_colors_to_indices
from safeforest.vis.visualize_classes import visualize_with_palette
from tqdm import tqdm

print(SUPER_ANNOTATE_PALETTE)

FOLDER = Path("/home/frc-ag-1/Downloads/Portugal_4_22_updated")
OUTPUT_FOLDER = Path(
    "/home/frc-ag-1/data/SafeForestData/datasets/portugal_UAV_4_22/derived/training_all_classes"
)
OUTPUT_FOLDER = Path(
    "/home/frc-ag-1/data/SafeForestData/datasets/portugal_UAV_4_22/derived/training_all_classes"
)

PATTERN = "data_mapping_velodyne_*"


def hex_to_rgb(hex):
    return tuple(int(hex[i + 1 : i + 3], 16) for i in (0, 2, 4))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", default=FOLDER, type=Path)
    parser.add_argument("--output-folder", default=OUTPUT_FOLDER, type=Path)
    parser.add_argument("--pattern", default="*")
    args = parser.parse_args()
    return args


# FILE = Path("/home/frc-ag-1/Downloads/UAV_Semantics/bag_4_7_0001.jpg___pixel.json")
# FOLDER = Path("/home/frc-ag-1/Downloads/Portugal_4_22")
# classes_file = Path(FOLDER, "classes", "classes.json")
# class_info = json.load(open(classes_file, "r"))
# print(class_info)
#
# class_color_map = {}
# for info in class_info:
#    class_color_map[hex_to_rgb(info["color"])] = info["name"]
#
# print(class_color_map.keys())
# print(class_color_map.values())
def superannotate_to_cityscapes(
    input_folder, output_folder, pattern, remap=False, exclude_class=False
):
    ensure_dir_normal_bits(output_folder)
    files = sorted(input_folder.glob(pattern))

    fig, axs = plt.subplots(1, 2)
    index = 0
    for i in tqdm(range(0, len(files), 4)):
        img = imread(files[i])
        label = imread(files[i + 1])
        if np.sum(label[..., :3]) == 0:
            continue
        axs[0].imshow(img)

        indices = convert_colors_to_indices(label, SUPER_ANNOTATE_PALETTE)
        if remap:
            remapped_indices = remap_classes_bool_indexing(
                indices, REMAP_SUPER_ANNOTATE_TO_SEMFIRE
            )
        else:
            unlabeled_indices = indices == 0
            indices[unlabeled_indices] = 255
            remapped_indices = indices

        # Don't include an image that's just background classes
        if exclude_class and np.all(remapped_indices == 7):
            continue

        axs[1].imshow(
            visualize_with_palette(remapped_indices, SUPER_ANNOTATE_PALETTE),
            vmin=0,
            vmax=SUPER_ANNOTATE_PALETTE.shape[0],
        )
        plt.pause(1)
        is_train = index >= 3
        write_cityscapes_file(
            remapped_indices, output_folder, index, is_ann=True, is_train=is_train
        )
        write_cityscapes_file(
            img, output_folder, index, is_ann=False, is_train=is_train
        )
        index += 1


if __name__ == "__main__":
    args = parse_args()
    superannotate_to_cityscapes(args.input_folder, args.output_folder, args.pattern)
