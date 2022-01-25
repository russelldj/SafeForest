import argparse
from pathlib import Path

import numpy as np
from imageio import imread
from safeforest.dataset_generation.file_utils import (
    get_files,
    link_cityscapes_file,
    write_cityscapes_file,
)

INPUT_FOLDER = Path(
    "/home/frc-ag-1/data/SafeForestData/datasets/semfire_segmentation/original/2021_sete_fontes_forest"
)
OUTPUT_FOLDER = Path(
    "/home/frc-ag-1/data/SafeForestData/datasets/semfire_segmentation/derived/training/2021_sete_fontes_forest"
)

IMG_SUBFOLDER = "img"
LABEL_SUBFOLDER = "lbl"
TRAIN_FRAC = 0.9


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", default=INPUT_FOLDER, type=Path)
    parser.add_argument("--output-folder", default=OUTPUT_FOLDER, type=Path)
    args = parser.parse_args()
    return args


def main(input_folder: Path, output_folder: Path, write_RG_only=True):
    """
    input_folder:
        The folder containing img, lbl, etc.
    output_folder:
        Where to write the simlinked output structure 
    """
    img_folder = Path(input_folder, IMG_SUBFOLDER)
    label_folder = Path(input_folder, LABEL_SUBFOLDER)

    img_files, label_files = [get_files(x, "*") for x in (img_folder, label_folder)]

    num_train = int(TRAIN_FRAC * len(img_files))
    for img_file, label_file in zip(img_files, label_files):
        index = int(img_file.stem)

        img = imread(img_file)

        if write_RG_only:
            # TODO consider trying to optimize this
            img[..., 2] = 0

        write_cityscapes_file(
            img, output_folder, index, is_ann=False, num_train=num_train
        )
        link_cityscapes_file(
            label_file, output_folder, index, is_ann=True, num_train=num_train
        )


if __name__ == "__main__":
    args = parse_args()
    main(args.input_folder, args.output_folder)
