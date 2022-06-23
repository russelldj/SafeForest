'''
This tool runs inference on a series of images.
    1) Loads the *.pth file in workdir/latest.pth
    2) Creates a workdir/predicted/ dir
    3) Runs inference on all test_dir/img_dir/*png images and saves them
'''

import argparse
import cv2
import numpy
from pathlib import Path
from shutil import rmtree

from mmseg.apis import inference_segmentor, init_segmentor
import mmcv


def main(config_file, test_dir, workdir):
    checkpoint_file = workdir.joinpath("latest.pth")
    predicted_dir = workdir.joinpath("predicted")
    rmtree(predicted_dir, ignore_errors=True)
    predicted_dir.mkdir()

    model = init_segmentor(
        str(config_file),
        str(checkpoint_file),
        device="cuda:0",
    )

    for image in test_dir.joinpath("img_dir").glob("*png"):
        results = inference_segmentor(model, image)
        cv2.imwrite(str(predicted_dir.joinpath(image.name)), results[0])


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "config_file",
        help="Path to the mmsegmentation model config file.",
        type=Path,
    )
    parser.add_argument(
        "test_dir",
        help="Path to dir where test images are stored in test_dir/img_dir/",
        type=Path,
    )
    parser.add_argument(
        "workdir",
        help="Path to dir where the model weights are stored and the images"
             " will be saved",
        type=Path,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args.config_file, args.test_dir, args.workdir)
