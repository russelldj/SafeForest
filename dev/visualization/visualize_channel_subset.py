import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import ubelt as ub
from dev.utils.video_utils import SmartVideoWriter
from imageio import imread, imwrite
from tqdm import tqdm


def show_RG(folder, extension="png"):
    files = sorted(folder.glob("*." + extension))
    for f in files:
        img = imread(f)
        img[..., 2] = 0
        plt.imshow(img)
        plt.show()


def write_RG_files(input_folder, output_folder, extension="png"):
    files = sorted(input_folder.glob("*." + str(extension)))
    ub.ensuredir(output_folder, mode=0o0755)
    for f in tqdm(files):
        img = imread(f)
        img = np.concatenate(
            (img[..., :2], np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)),
            axis=2,
        )
        output_file = Path(output_folder, f.name)
        imwrite(output_file, img)


def write_RG_video(input_folder, output_file, extension="png"):
    writer = SmartVideoWriter(output_file)
    files = sorted(input_folder.glob("*." + str(extension)))
    ub.ensuredir(output_file.parent, mode=0o0755)
    for f in tqdm(files):
        img = imread(f)
        img = np.concatenate(
            (img[..., :2], np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)),
            axis=2,
        )
        img = np.flip(img, axis=2)
        writer.write(img)
    writer.release()


def write_RG_video_from_video(input_file, output_file, extension="png"):
    cap = cv2.VideoCapture(str(input_file))
    writer = SmartVideoWriter(output_file)
    ub.ensuredir(output_file.parent, mode=0o0755)
    while True:
        ret, img = cap.read()

        if not ret:
            break
        img = np.concatenate(
            (img[..., :2], np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)),
            axis=2,
        )
        img = np.flip(img, axis=2)
        writer.write(img)
    writer.release()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--write-video", action="store_true")
    parser.add_argument("--read-video", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.write_video:
        if args.read_video:
            write_RG_video_from_video(args.input, args.output, args.extension)
        else:
            write_RG_video(args.input, args.output, args.extension)
    else:
        write_RG_files(args.input, args.output, args.extension)
