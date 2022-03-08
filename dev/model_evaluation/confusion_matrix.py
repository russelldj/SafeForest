import argparse
import os

import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy import spatial

from safeforest.config import LABELS_INFO, RUI_PALETTE
from safeforest.vis.cf_matrix import make_confusion_matrix
from safeforest.model_evaluation.accuracy_computation import accumulate_confusion_matrix

np.random.seed(123)
# PALETTE = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
PALETTE = RUI_PALETTE
PALETTE = np.flip(PALETTE, axis=1)
names = [x["name"] for x in LABELS_INFO]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-video")
    parser.add_argument("--gt-folder")
    parser.add_argument("--test-file")
    args = parser.parse_args()
    return args


def main(pred_video, gt_folder, test_file):
    video_cap = cv2.VideoCapture(pred_video)
    confusion = np.zeros((7, 7))
    with open(test_file, "r") as infile_h:
        for line in infile_h:
            filename = line.split(",")[1].strip()
            gt_path = os.path.join(gt_folder, filename)
            gt_image = cv2.imread(gt_path)[..., 0]
            ret, pred_image = video_cap.read()

            pred_image = pred_image.reshape((-1, 3))

            dists = spatial.distance.cdist(pred_image, PALETTE[:7])
            pred_ids = np.argmin(dists, axis=1)
            pred_image = np.reshape(pred_ids, gt_image.shape[:2])

            # fig, axs = plt.subplots(1, 2)
            # axs[0].imshow(gt_image * 20)
            # axs[1].imshow(pred_image * 20)
            # plt.show()

            gts = gt_image.flatten()
            preds = pred_image.flatten()

            confusion = accumulate_confusion_matrix(preds, gts, confusion)
            print(gt_path)
    np.save("vis/confusion.npy", confusion)
    vis()


def vis(lognorm=False):
    confusion = np.load("vis/confusion.npy")

    confusion = confusion / (np.sum(confusion) / 100.0)
    make_confusion_matrix(
        confusion, categories=names, count=False, cmap="Blues", norm=None
    )
    plt.show()
    plt.xticks(np.arange(7), names, fontsize=12)
    plt.yticks(np.arange(7), names, fontsize=12)
    plt.xlabel("Predicted", fontsize=20)
    plt.ylabel("True", fontsize=20)
    if lognorm:
        plt.imshow(confusion, norm=LogNorm())
    else:
        plt.imshow(confusion)
    plt.colorbar()
    plt.title("Confusion matrix on synthetic test set", fontsize=20)
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    # vis()
    main(args.pred_video, args.gt_folder, args.test_file)
