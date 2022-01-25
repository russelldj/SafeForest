import argparse
import itertools

import cv2
import numpy as np
from dev.dataset_creation.merge_classes import compute_nearest_class
from safeforest.config import PALETTE_MAP, REMAP_MAP
from safeforest.dataset_generation.file_utils import (
    ensure_dir_normal_bits,
    write_cityscapes_file,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--palette-name")
    parser.add_argument("--remap-name")
    args = parser.parse_args()
    return args


def main(input: str, output, palette_name, train_frac=0.85, remap_name=None):
    """
    input:
        The input video file
    output: 
        The output folder
    palette_name:
        What palette to use
    train_frac:
        What fraction of the images to put in the train folder
    remap:
        The name of a map to change the IDs
    """
    palette = PALETTE_MAP[palette_name]
    if remap_name is not None:
        remap = REMAP_MAP[remap_name]
    else:
        remap = np.arange(palette.shape[0])

    cap = cv2.VideoCapture(input)
    ensure_dir_normal_bits(output)

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_train = int(train_frac * num_frames)
    for i in itertools.count():
        ret, img = cap.read()
        if not ret:
            break

        classes = compute_nearest_class(img, palette)
        remapped_classes = remap[classes]
        write_cityscapes_file(remapped_classes, output, i, True, num_train)


if __name__ == "__main__":
    args = parse_args()
    main(args.input, args.output, args.palette_name)
