import argparse
from pathlib import Path
import itertools

from imageio import imread, imwrite
import matplotlib.pyplot as plt
import cv2
import numpy as np
from ubelt import ensuredir
from config import (
    RUI_PALETTE,
    YAMAHA_PALETTE,
    RUI_YAMAHA_PALETTE,
    RGB_EXT,
    SEG_EXT,
    IMG_DIR,
    ANN_DIR,
    TRAIN_DIR,
    VAL_DIR,
)


from show_seg_video import blend_images_gray
from scipy import spatial

"""
Take the predictions on two videos and merge them
"""

# Yamaha classes are i, Rui classes are j
REMAP = np.asarray(
    [
        [3, 3, 3, 3, 3, 5, 5],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 3, 3, 3, 3, 5, 5],
        [2, 2, 2, 2, 2, 2, 2],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [4, 4, 4, 4, 4, 4, 4],
        [4, 4, 4, 4, 4, 4, 4],
    ],
    dtype=np.uint8,
)


def visualize_with_palette(index_image, palette):
    """
    index_image : np.ndarray
        The predicted semantic map with indices. (H,W)
    palette : np.ndarray
        The colors for each index. (N classes,3)
    """
    h, w = index_image.shape
    index_image = index_image.flatten()
    colored_image = palette[index_image]
    colored_image = np.reshape(colored_image, (h, w, 3))
    return colored_image.astype(np.uint8)


def compute_nearest_class(pred_image, palette):
    """
    pred_image : np.ndarray
        The predicted colors, assumed to be BGR
    palette : np.ndarray
        The colors of each class
    """
    pred_image = np.flip(pred_image, axis=2)
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


def write_cityscapes_file(img, output_folder, index, is_ann, num_train):
    output_sub_folder = Path(
        output_folder,
        ANN_DIR if is_ann else IMG_DIR,
        TRAIN_DIR if index < num_train else VAL_DIR,
    )
    ensuredir(output_sub_folder, mode=0o0755)
    filename = f"{index:06d}{SEG_EXT if is_ann else RGB_EXT}.png"
    output_filepath = Path(output_sub_folder, filename)

    imwrite(output_filepath, img)


def main(
    raw_video,
    first_video,
    second_video,
    output,
    first_palette=YAMAHA_PALETTE,
    second_palette=RUI_PALETTE,
    write_video=False,
):
    raw_cap = cv2.VideoCapture(str(raw_video))
    first_cap = cv2.VideoCapture(str(first_video))
    second_cap = cv2.VideoCapture(str(second_video))

    num_frames = int(raw_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_train = int(num_frames * 0.85)

    video_writer = None

    for i in itertools.count(start=0):
        raw_ret, raw_img = raw_cap.read()
        first_ret, first_pred = first_cap.read()
        second_ret, second_pred = second_cap.read()
        if False in (raw_ret, first_ret, second_ret):
            break

        if video_writer is None and write_video:
            frame_height, frame_width, _ = raw_img.shape
            video_writer = cv2.VideoWriter(
                str(output),
                cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                15,
                (frame_width, frame_height),
            )

        first_classes = compute_nearest_class(first_pred, first_palette)
        second_classes = compute_nearest_class(second_pred, second_palette)
        combined_classes = combine_classes(first_classes, second_classes, REMAP)
        if write_video:
            visualized_combined_classes = visualize_with_palette(
                combined_classes, RUI_YAMAHA_PALETTE
            )

            overlay = blend_images_gray(
                raw_img, np.flip(visualized_combined_classes, axis=2)
            )

            video_writer.write(overlay)
        else:
            write_cityscapes_file(
                np.flip(raw_img, axis=2), output, i, is_ann=False, num_train=num_train
            )
            write_cityscapes_file(
                combined_classes, output, i, is_ann=True, num_train=num_train
            )
    if write_video:
        video_writer.release()
        # TODO write out the data


def parse_args():
    parser = argparse.ArgumentParser(
        "Combine classes from two different labeled videos"
    )
    parser.add_argument("--raw-video", type=Path, help="Path to color video")
    parser.add_argument(
        "--first-video", type=Path, help="Path to first input class video"
    )
    parser.add_argument(
        "--second-video", type=Path, help="Path to second input class video"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Either folder to write the dataset or video to write to",
    )
    parser.add_argument("--write-video", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(
        args.raw_video,
        args.first_video,
        args.second_video,
        args.output_video,
        args.write_video,
    )
