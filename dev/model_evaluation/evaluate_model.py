import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from mmseg.apis import inference_segmentor, init_segmentor
from safeforest.dataset_generation.file_utils import get_files
from safeforest.model_evaluation.accuracy_computation import accumulate_confusion_matrix
from safeforest.vis.cf_matrix import make_confusion_matrix
from safeforest.config import SEMFIRE_CLASSES, REMAP_MAP
from tqdm import tqdm

CFG_PATH = Path(
    "/home/frc-ag-1/dev/SafeForestSuperepo/data/models/segformer_mit-b5_512x512_160k_portugal_UAV_12_21_safe_forest_2_pred_labels_sete/segformer_mit-b5_512x512_160k_portugal_UAV_12_21_safe_forest_2_pred_labels_sete.py"
)
MODEL_PATH = Path(
    "/home/frc-ag-1/dev/SafeForestSuperepo/data/models/segformer_mit-b5_512x512_160k_portugal_UAV_12_21_safe_forest_2_pred_labels_sete/iter_160000.pth"
)
IMAGES_DIR = Path(
    "/home/frc-ag-1/data/SafeForestData/datasets/semfire_segmentation/derived/sete_fonte_rgb/rgb_subset"
)
GROUNDTRUTH_DIR = Path(
    "/home/frc-ag-1/data/SafeForestData/datasets/semfire_segmentation/derived/sete_fonte_lbl_padded_names"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", default=CFG_PATH)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--images-dir", default=IMAGES_DIR)
    parser.add_argument("--groundtruth-dir", default=GROUNDTRUTH_DIR)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--remap")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    return args


def main(
    cfg_path,
    model_path,
    images_dir,
    groundtruth_dir,
    num_classes,
    verbose=False,
    remap=None,
):
    model = init_segmentor(str(cfg_path), str(model_path))
    img_files, label_files = [get_files(x, "*") for x in (images_dir, groundtruth_dir)]
    confusion = np.zeros((num_classes, num_classes))

    if remap in REMAP_MAP.keys():
        # Maps the index to the new label
        remap = REMAP_MAP[remap]
    else:
        print(f"Warning {remap} not in {REMAP_MAP.keys()}")
        remap = None

    for img_file, label_file in tqdm(zip(img_files, label_files)):
        img = imread(img_file)
        pred = inference_segmentor(model, img)[0]
        if remap is not None:
            # Remap the labels
            # TODO this is slow
            pred = remap[pred]
        label = imread(label_file)

        confusion = accumulate_confusion_matrix(
            pred.flatten(),
            label.flatten(),
            current_confusion=confusion,
            n_classes=len(SEMFIRE_CLASSES),
        )
        if verbose:
            _, axs = plt.subplots(1, 2)
            axs[0].imshow(img)
            axs[1].imshow(np.concatenate((pred, label), axis=1))
            plt.show()

    print(f"Accuracy {np.sum(confusion.trace()) / np.sum(confusion)}")
    make_confusion_matrix(
        confusion,
        categories=SEMFIRE_CLASSES,
        count=False,
        cmap="Blues",
        norm=None,
        xyplotlabels=True,
    )
    # plt.xticks(np.arange(len(SEMFIRE_CLASSES)), SEMFIRE, fontsize=12)
    # plt.yticks(np.arange(len(SEMFIRE_CLASSES)), SEMFIRE, fontsize=12)
    plt.xlabel("Predicted", fontsize=20)
    plt.ylabel("True", fontsize=20)
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.cfg_path,
        args.model_path,
        args.images_dir,
        args.groundtruth_dir,
        args.num_classes,
        args.verbose,
        args.remap,
    )

