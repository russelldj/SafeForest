import argparse
import itertools
from pathlib import Path

import cv2
from cv2 import PARAM_UNSIGNED_INT
import matplotlib.pyplot as plt
import numpy as np
from dev.utils.img_utils import lap2_focus_measure
from safeforest.vis.visualize_classes import blend_images_gray
from safeforest.config import RUI_PALETTE, RUI_YAMAHA_PALETTE, YAMAHA_PALETTE
from safeforest.dataset_generation.class_utils import combine_classes
from safeforest.dataset_generation.file_utils import write_cityscapes_file
from safeforest.vis.visualize_classes import visualize_with_palette
from scipy import spatial
from tqdm import tqdm

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


def compute_nearest_class_1D(pred_pixels, palette):
    """

    Args:
        pred_pixels: (n, 3) pixels, assumed to be RGB
    """
    dists = spatial.distance.cdist(pred_pixels, palette)
    pred_ids = np.argmin(dists, axis=1)
    return pred_ids


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
    pred_ids = compute_nearest_class_1D(pred_image,)
    pred_image = np.reshape(pred_ids, img_shape)
    return pred_image


def main(
    raw_video,
    first_video,
    second_video,
    output,
    first_palette=YAMAHA_PALETTE,
    second_palette=RUI_PALETTE,
    write_video=False,
    sharpness_threshold=1000,
):
    """
    sharpness_theshold: float | None
        Discard images that are not this sharp. If None, keep all images.
    """
    raw_cap = cv2.VideoCapture(str(raw_video))
    first_cap = cv2.VideoCapture(str(first_video))
    second_cap = cv2.VideoCapture(str(second_video))

    num_frames = int(raw_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_train = int(num_frames * 0.85)

    video_writer = None
    sharps = []

    for i in tqdm(itertools.count(start=0)):
        raw_ret, raw_img = raw_cap.read()
        first_ret, first_pred = first_cap.read()
        second_ret, second_pred = second_cap.read()
        if False in (raw_ret, first_ret, second_ret):
            break

        if sharpness_threshold is not None:
            sharpness = lap2_focus_measure(raw_img, np.ones_like(raw_img).astype(bool))
            sharps.append(sharpness)
            if sharpness < sharpness_threshold:
                continue

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
    xs = np.arange(len(sharps))
    xs = xs / 15.0
    plt.style.use("./dev/report.mplstyle")
    plt.plot(xs, sharps)
    plt.xlabel("Seconds")
    plt.ylabel("LAP2 sharpness metric")
    plt.savefig("vis/sharpness")
    plt.show()

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
        args.output,
        args.write_video,
    )
