import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from imageio import imread, imwrite
from mmseg.apis import inference_segmentor, init_segmentor
from sacred import Experiment
from sacred.observers import MongoObserver
from safeforest.config import PALETTE_MAP, REMAP_MAP, SEMFIRE_CLASSES
from safeforest.dataset_generation.file_utils import get_files
from safeforest.model_evaluation.accuracy_computation import accumulate_confusion_matrix
from safeforest.vis.cf_matrix import make_confusion_matrix
from tqdm import tqdm

ex = Experiment("evaluate_model")
ex.observers.append(MongoObserver(url="localhost:27017", db_name="mmseg"))

VALID_CLASSES = np.array([True, True, True, False, False, True, False])
QUALITATIVE_FILE = "vis/qualatative_{:06d}.png"
SACRED = True

CFG_PATH = Path(
    "/home/frc-ag-1/dev/SafeForestSuperepo/data/models/segformer_mit-b5_512x512_160k_portugal_UAV_12_21_safe_forest_2_pred_labels_sete/segformer_mit-b5_512x512_160k_portugal_UAV_12_21_safe_forest_2_pred_labels_sete.py"
)
MODEL_PATH = Path(
    "/home/frc-ag-1/dev/SafeForestSuperepo/data/models/segformer_mit-b5_512x512_160k_portugal_UAV_12_21_safe_forest_2_pred_labels_sete/iter_160000.pth"
)
IMAGES_DIR = Path(
    "/home/frc-ag-1/data/SafeForestData/datasets/semfire_segmentation/derived/training/train_different_percentages/train_80_percent/img_dir/val"
)
GROUNDTRUTH_DIR = Path(
    "/home/frc-ag-1/data/SafeForestData/datasets/semfire_segmentation/derived/training/train_different_percentages/train_80_percent/ann_dir/val"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", default=CFG_PATH)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--images-dir", default=IMAGES_DIR)
    parser.add_argument("--groundtruth-dir", default=GROUNDTRUTH_DIR)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--remap")
    parser.add_argument("--palette", options=list(PALETTE_MAP.keys()))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    return args


@ex.config
def config():
    cfg_path = CFG_PATH
    model_path = None  # MODEL_PATH
    images_dir = IMAGES_DIR
    groundtruth_dir = None  # GROUNDTRUTH_DIR
    num_classes = 7
    remap = None
    palette = None
    verbose = False
    log_preds = False
    sample_freq = 1  # At what freqency to take frames

    if model_path is None:
        model_path = None

    # Infer groundtruth dir
    if groundtruth_dir is None:
        if str(images_dir).count("img_dir") == 1:
            groundtruth_dir = str(images_dir).replace("img_dir", "ann_dir")


def confusion_union(matrix, index):
    # Row + column - double count
    return np.sum(matrix[index, :]) + np.sum(matrix[:, index]) - matrix[index, index]


def compute_mIoU(confusion_matrix: np.array):
    """

    confusion_matrix: (n, n) true label in i, pred in j
    """
    num_classes = confusion_matrix.shape[0]
    IoUs = []
    for i in range(num_classes):
        intersection = confusion_matrix[i, i]
        union = confusion_union(confusion_matrix, i)
        IoU = intersection / union
        IoUs.append(IoU)

    mIoU = np.mean(IoUs)
    return mIoU, IoUs


@ex.automain
def main(
    cfg_path,
    model_path,
    images_dir,
    groundtruth_dir,
    num_classes,
    verbose=False,
    remap=None,
    palette=None,
    log_preds=False,
    _run=None,
    sample_freq=1,
):
    model = init_segmentor(str(cfg_path), str(model_path))

    img_files = get_files(images_dir, "*")
    if groundtruth_dir is None:
        label_files = [None] * len(img_files)
    else:
        label_files = get_files(groundtruth_dir, "*")

    confusion = np.zeros((num_classes, num_classes))

    if palette in PALETTE_MAP.keys():
        palette = PALETTE_MAP[palette]
    else:
        print(f"Warning {palette} not in {PALETTE_MAP.keys()}")
        palette = None

    if remap in REMAP_MAP.keys():
        # Maps the index to the new label
        remap = REMAP_MAP[remap]
    else:
        print(f"Warning {remap} not in {REMAP_MAP.keys()}")
        remap = None

    _, axs = plt.subplots(1, 2)

    for i, (img_file, label_file) in tqdm(enumerate(zip(img_files, label_files))):
        if (i % sample_freq) != 0:
            continue

        img = imread(img_file)
        pred = inference_segmentor(model, img)[0]
        if remap is not None:
            # Remap the labels
            # TODO this is slow
            pred = remap[pred]

        if label_file is not None:
            label = imread(label_file)

            confusion = accumulate_confusion_matrix(
                pred.flatten(),
                label.flatten(),
                current_confusion=confusion,
                n_classes=len(SEMFIRE_CLASSES),
            )

        if log_preds:
            if palette is not None:

                pred = palette[pred]
                if label_file is not None:
                    label = palette[label]
                    output_img = np.concatenate((img, pred, label), axis=1).astype(
                        np.uint8
                    )
                else:
                    output_img = np.concatenate((img, pred), axis=1).astype(np.uint8)

                imwrite(QUALITATIVE_FILE.format(i), output_img)
            else:
                axs[0].imshow(img)
                if label_file is not None:
                    axs[1].imshow(np.concatenate((pred, label), axis=1))
                axs[1].imshow(np.concatenate((pred), axis=1))
                plt.savefig(QUALITATIVE_FILE.format(i))

            if SACRED:
                _run.add_artifact(QUALITATIVE_FILE.format(i))
            if verbose:
                plt.show()

    confusion = confusion[np.ix_(VALID_CLASSES, VALID_CLASSES)]
    valid_semfire_classes = np.array(SEMFIRE_CLASSES)[VALID_CLASSES]

    accuracy = np.sum(confusion.trace()) / np.sum(confusion)
    miou, ious = compute_mIoU(confusion)
    print(f"mIoU {miou}")
    print(f"IoUs {ious}")
    print(f"Accuracy {accuracy}")
    plt.close()
    extra_artists = make_confusion_matrix(
        confusion,
        categories=valid_semfire_classes,
        count=False,
        cmap="Blues",
        norm=None,
        xyplotlabels=True,
    )
    plt.xlabel("Predicted", fontsize=20)
    plt.ylabel("True", fontsize=20)
    plt.savefig(
        "vis/confusion_matrix.png",
        bbox_extra_artists=extra_artists,
        bbox_inches="tight",
    )
    np.save("res/confusion_matrix.npy", confusion)
    if SACRED:
        _run.add_artifact("vis/confusion_matrix.png")
        _run.add_artifact("res/confusion_matrix.npy")
    if verbose:
        plt.show()
    return accuracy


if __name__ == "__main__":
    if not SACRED:
        args = parse_args()
        main(
            args.cfg_path,
            args.model_path,
            args.images_dir,
            args.groundtruth_dir,
            args.num_classes,
            args.verbose,
            remap=args.remap,
            palette=args.pallete,
        )

