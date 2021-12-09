import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-video")
    parser.add_argument("--gt-folder")
    parser.add_argument("--test-file")
    args = parser.parse_args()
    return args


def main(pred_video, gt_folder, test_file):
    video_cap = cv2.VideoCapture(pred_video)
    with open(test_file, "r") as infile_h:
        for line in infile_h:
            filename = line.split(",")[1].strip()
            gt_path = os.path.join(gt_folder, filename)
            gt_image = cv2.imread(gt_path)
            ret, pred_image = video_cap.read()
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(gt_image * 20)
            axs[1].imshow(np.flip(pred_image, axis=2))
            plt.show()
            print(gt_path)


if __name__ == "__main__":
    args = parse_args()
    main(args.pred_video, args.gt_folder, args.test_file)
