import argparse
import cv2
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt


class VideoWriter:
    def __init__(self, filename, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=10):
        self.writer = None
        self.fourcc = fourcc
        self.filename = str(filename)
        self.fps = fps

    def write(self, img):
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = cv2.VideoWriter(self.filename, self.fourcc, self.fps, (w, h))
        self.writer.write(img)

    def release(self):
        if self.writer is not None:
            self.writer.release()


def blend_images(im1, im2, alpha=0.7):
    return (alpha * im1 + (1 - alpha) * im2).astype(np.uint8)


def blend_images_gray(im1, im2, alpha=0.7):
    num_channels = im1.shape[2]
    im1 = np.mean(im1, axis=2)
    im1 = np.expand_dims(im1, axis=2)
    im1 = np.repeat(im1, repeats=num_channels, axis=2)
    return (alpha * im1 + (1 - alpha) * im2).astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--label-path", type=Path)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()
    return args


def main(video_path, label_path, output_path):
    video_cap = cv2.VideoCapture(str(video_path))
    label_cap = cv2.VideoCapture(str(label_path))
    output_writer = VideoWriter(output_path)

    while True:
        video_ret, img = video_cap.read()
        label_ret, label = label_cap.read()
        if not video_ret or not label_ret:
            break
        blended = blend_images_gray(img, label)
        output_writer.write(blended)
    output_writer.release()


if __name__ == "__main__":
    args = parse_args()
    main(args.video_path, args.label_path, args.output_path)
