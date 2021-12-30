import argparse
from pathlib import Path

import cv2
import numpy as np
from config import RUI_PALETTE, YAMAHA_PALETTE
from scipy import spatial

"""
Take the predictions on two videos and merge them
"""

# REMAP = np.asarray([[], [], [], [], [], [], [], []], dtype=np.uint8)
REMAP = np.zeros((7, 8), dtype=np.uint8)


def compute_nearest_class(pred_image, palette):
    """
    pred_image : np.ndarray
        The predicted colors
    palette : np.ndarray
        The colors of each class
    """
    img_shape = pred_image.shape[:2]
    pred_image = pred_image.reshape((-1, 3))

    dists = spatial.distance.cdist(pred_image, palette)
    pred_ids = np.argmin(dists, axis=1)
    pred_image = np.reshape(pred_ids, img_shape)
    return pred_image


def combine_classes(first_image, second_image, remap):
    """
    remap : np.ndarray 
        An integer array where the element at the ith, jth location represents the class
        when a pixel in the first image has value i and the pixel in the second image has
        value j.
    """
    img_shape = first_image.shape
    first_image, second_image = [x.flatten() for x in (first_image, second_image)]
    remapped = remap[first_image, second_image]
    remapped = np.reshape(remapped, img_shape)
    return remapped


def main(
    raw_video,
    first_video,
    second_video,
    output_video,
    first_palette=RUI_PALETTE,
    second_palette=YAMAHA_PALETTE,
):
    raw_cap = cv2.VideoCapture(str(raw_video))
    first_cap = cv2.VideoCapture(str(first_video))
    second_cap = cv2.VideoCapture(str(second_video))
    while True:
        raw_ret, raw_pred = first_cap.read()
        first_ret, first_pred = first_cap.read()
        second_ret, second_pred = second_cap.read()
        if False in (raw_ret, first_ret, second_ret):
            break

        first_classes = compute_nearest_class(first_pred, first_palette)
        second_classes = compute_nearest_class(second_pred, second_palette)
        output = combine_classes(first_classes, second_classes, REMAP)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-video", type=Path)
    parser.add_argument("--first-video", type=Path)
    parser.add_argument("--second-video", type=Path)
    parser.add_argument("--output-video", type=Path)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args.raw_video, args.first_video, args.second_video, args.output_video)
