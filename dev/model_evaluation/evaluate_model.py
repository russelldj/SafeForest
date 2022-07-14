import argparse
import logging
from pathlib import Path
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import numpy as np
from imageio import imread, imwrite
from sacred import Experiment
from sacred.observers import MongoObserver
from safeforest.config import PALETTE_MAP, REMAP_MAP, SEMFIRE_CLASSES
from safeforest.dataset_generation.file_utils import get_files
from safeforest.model_evaluation.accuracy_computation import (
    accumulate_confusion_matrix,
    compute_mIoU,
)
from safeforest.vis.cf_matrix import make_confusion_matrix
from tqdm import tqdm

ex = Experiment("evaluate_model")
ex.observers.append(MongoObserver(url="localhost:27017", db_name="mmseg"))

VALID_CLASSES = np.array([True, True, True, False, False, True, False])
QUALITATIVE_FILE = "vis/qualitative_{:06d}.png"

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
    parser.add_argument("--palette", choices=list(PALETTE_MAP.keys()))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-preds", action="store_true")
    parser.add_argument("--sacred", action="store_true")
    args = parser.parse_args()

    # Infer groundtruth dir
    if args.groundtruth_dir is None:
        if str(args.images_dir).count("img_dir") == 1:
            args.groundtruth_dir = str(args.images_dir).replace("img_dir", "ann_dir")

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


def predict(img_files, model):
    from mmseg.apis import inference_segmentor
    for img_file in img_files:
        yield inference_segmentor(model, imread(img_file))[0]


def sample_for_confusion(img_files, pred_files, label_files, sample_freq,
                         num_classes, remap=None, log_preds=False, palette=None,
                         sacred=False, verbose=False, _run=None,
                         save_file=QUALITATIVE_FILE):

    confusion = np.zeros((num_classes, num_classes))
    _, axs = plt.subplots(2, 2, figsize=(10, 9))

    for i, (img_file, pred, label_file) in tqdm(enumerate(zip(img_files,
                                                              pred_files,
                                                              label_files))):

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
                n_classes=num_classes,
            )

        if log_preds:

            # Sample the confusion matrix every image, just subsample the
            # qualitative images
            if (i % sample_freq) != 0:
                continue

            img = imread(img_file)
            save_name = save_file.format(i)

            if palette is not None:
                pred = palette[pred]
                white_bar = np.ones((img.shape[0], 10, 3)) * 255
                if label_file is not None:
                    label = palette[label]
                    output_img = np.concatenate(
                        (img, white_bar, pred, white_bar, label), axis=1
                    ).astype(np.uint8)
                else:
                    output_img = np.concatenate((img, white_bar, pred), axis=1).astype(
                        np.uint8
                    )

                imwrite(save_name, output_img)
            else:
                axs[0, 0].imshow(img)
                axs[0, 0].set_title("Original image")
                imshow_with_default(axs[0, 1], pred, num_classes)
                axs[0, 1].set_title("Predicted classes")
                if label_file is not None:
                    imshow_with_default(axs[1, 1], label, num_classes)
                    axs[1, 1].set_title("True classes")
                    axs[1, 0].imshow((label - pred) != 0)
                    axs[1, 0].set_title("Wrong labels")
                plt.tight_layout()
                plt.savefig(save_name)

            if sacred:
                _run.add_artifact(save_name)
            if verbose:
                plt.show()
    return confusion


def imshow_with_default(axis, image, num_classes):
    '''Do imshow with corner values set for consistent coloring.'''
    image[0, 0] = 0
    image[0, 1] = num_classes - 1
    axis.imshow(image)


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
    sacred=True,
):
    from mmseg.apis import init_segmentor
    model = init_segmentor(str(cfg_path), str(model_path))

    img_files = get_files(images_dir, "*")
    if groundtruth_dir is None:
        label_files = [None] * len(img_files)
    else:
        label_files = get_files(groundtruth_dir, "*")

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

    confusion = sample_for_confusion(
        img_files=img_files,
        pred_files=predict(img_files, model),
        label_files=label_files,
        sample_freq=sample_freq,
        num_classes=num_classes,
        log_preds=log_preds,
        palette=palette,
        sacred=sacred,
        verbose=verbose,
        _run=_run,
        save_file=QUALITATIVE_FILE,
    )
    plt.close()

    confusion = confusion[np.ix_(VALID_CLASSES, VALID_CLASSES)]
    valid_semfire_classes = np.array(SEMFIRE_CLASSES)[VALID_CLASSES]

    return calc_metrics(confusion=confusion,
                        classes=valid_semfire_classes,
                        save_dir=Path("vis/"),
                        sacred=sacred,
                        _run=_run)


def calc_metrics(confusion, classes, save_dir, sacred=False, _run=None,
                 verbose=False, include_report=False):

    graph_path = save_dir.joinpath("confusion_matrix.png")
    array_path = save_dir.joinpath("confusion_matrix.npy")

    accuracy = np.sum(confusion.trace()) / np.sum(confusion)
    miou, ious = compute_mIoU(confusion)
    print(f"mIoU {miou}")
    print(f"IoUs {ious}")
    print(f"Accuracy {accuracy}")
    extra_artists = make_confusion_matrix(
        confusion,
        categories=classes,
        count=False,
        cmap="Blues",
        norm=None,
        xyplotlabels=True,
    )
    plt.xlabel("Predicted", fontsize=20)
    plt.ylabel("True", fontsize=20)
    plt.savefig(
        str(graph_path),
        bbox_extra_artists=extra_artists,
        bbox_inches="tight",
    )
    np.save(array_path, confusion)

    # This will take a long time for images, which is why it is disableable
    report = None
    if include_report:

    if sacred:
        _run.add_artifact(graph_path)
        _run.add_artifact(array_path)
        _run.log_scalar("mIoU", miou)
        for class_name, class_iou in zip(classes, ious):
            _run.log_scalar(f"IoUs_{class_name}", class_iou)
        if include_report:
            for stat in ("precision", "recall", "f1-score"):
                for class_name in classes:
                    _run.log_scalar(f"{stat}_{class_name}",
                                    report[class_name][stat])
                for combo in ("macro avg", "weighted avg"):
                    _run.log_scalar(f"{stat}_{combo.replace(' ', '_')}",
                                    report[combo][stat])
        _run.log_scalar("Accuracy", accuracy)
    if verbose:
        plt.show()

    return accuracy, ious, confusion


# Note that report generation may take a long time because classification_report
# takes in the full dataset as a vector and this causes huse vectors when we're
# dealing with images. If this causes issues, try to refactor.
def confusion_to_class_report(confusion_matrix, class_names):

    # This should always be true, just play it safe because of the shape[]
    # stuff below
    assert len(confusion_matrix.shape) == 2

    # Recreate true/predicted data using the matrix
    true_list = []
    pred_list = []

    y_true = numpy.array([])
    y_pred = numpy.array([])
    # Over the rows (true class)
    for i in range(confusion_matrix.shape[0]):
        # Over the columns (predicted class)
        for j in range(confusion_matrix.shape[1]):
            number = int(confusion_matrix[i, j])
            y_true = numpy.hstack((y_true, [i] * number))
            y_pred = numpy.hstack((y_pred, [j] * number))

    return classification_report(
        y_true=y_true,
        y_pred=y_pred,
        labels=[_ for _ in range(len(class_names))],
        target_names=class_names,
        output_dict=True,
    )


# if __name__ == "__main__":
#    args = parse_args()
#    main(
#        args.cfg_path,
#        args.model_path,
#        args.images_dir,
#        args.groundtruth_dir,
#        args.num_classes,
#        args.verbose,
#        remap=args.remap,
#        palette=args.palette,
#        log_preds=args.log_preds,
#        sacred=args.sacred,
#    )
