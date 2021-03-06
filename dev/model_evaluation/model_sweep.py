import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from safeforest.vis.cf_matrix import make_confusion_matrix
from safeforest.model_evaluation.accuracy_computation import compute_mIoU

plt.style.use("./dev/report.mplstyle")
RUI_MIOU = 0.353  # 0.29298355970960777

SIZES = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80]
SHIFTS = [0, 30, 60, 90, 120]
PRETRAINS = ["synthetic_pretrain_", ""]

CUT_SIZES = list(reversed([0.40, 0.60, 0.80]))
CUT_SHIFTS = [0]
CUT_PRETRAINS = ["CUT_pretrain_"]

MODEL_PATH = "/jet/home/russelld/data/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_{}sete_finetune_train_000{:03d}_{:.2f}0000/latest.pth"
CONFIG_PATH = "/jet/home/russelld/data/dev/mmsegmentation/work_dirs/segformer_mit-b5_512x512_10k_{}sete_finetune_train_000{:03d}_{:.2f}0000/segformer_mit-b5_512x512_10k_{}sete_finetune_train_000{:03d}_{:.2f}0000.py"
IMAGES_DIR = "/jet/home/russelld/data/SafeForestData/datasets/semfire_segmentation/derived/training/5_fold_datasets/train_000{:03d}_0.800000/img_dir/val"
ANN_DIR = "/jet/home/russelld/data/SafeForestData/datasets/semfire_segmentation/derived/training/5_fold_datasets/train_000{:03d}_0.800000/ann_dir/val"
LOG_PREDS = False
SWEEP_FILE = "res/CUT_sweep.pkl"
RUN = False

accuracies = []
all_ious = []
confusions = []
all_sizes = []
all_shifts = []

results = {}
if RUN:
    from evaluate_model import main

    for pretrain in PRETRAINS:
        results[pretrain] = {}
        for shift in SHIFTS:
            results[pretrain][shift] = []
            for size in SIZES:
                model_path = MODEL_PATH.format(pretrain, shift, size)
                config_path = CONFIG_PATH.format(
                    pretrain, shift, size, pretrain, shift, size
                )
                images_dir = IMAGES_DIR.format(shift)
                ann_dir = ANN_DIR.format(shift)
                print(
                    f"model_path: {model_path}\n config path: {config_path}\n images_dir: {images_dir}\n ann_dir: {ann_dir}\n"
                )
                accuracy, ious, confusion = main(
                    cfg_path=config_path,
                    model_path=model_path,
                    images_dir=images_dir,
                    groundtruth_dir=ann_dir,
                    num_classes=7,
                    palette="semfire",
                    log_preds=LOG_PREDS,
                )
                current_res = {
                    "accuracy": accuracy,
                    "ious": ious,
                    "accuracy": accuracy,
                    "confusion": confusion,
                    "size": size,
                }
                results[pretrain][shift].append(current_res)
                plt.close()
                with open(SWEEP_FILE, "wb") as f:
                    pickle.dump(results, f)


def produce_mious(res_dict: dict, sizes: list = SIZES):
    output_dict = {}
    for k, v in res_dict.items():
        output_dict[k] = {"mious": [], "ious": [], "sizes": []}
        for i, thing in enumerate(v):
            output_dict[k]["sizes"].append(sizes[i])
            output_dict[k]["mious"].append(np.mean(thing["ious"]))
            output_dict[k]["ious"].append(thing["ious"])

    return output_dict


def plot_mious(mious_dict, label, c=None):
    sizes = list(mious_dict.values())[0]["sizes"]
    sizes = [151 * size for size in sizes]

    mious_array = np.stack([v["mious"] for _, v in mious_dict.items()], axis=0)
    mean_mious = np.mean(mious_array, axis=0)
    min_mious = np.min(mious_array, axis=0)
    max_mious = np.max(mious_array, axis=0)

    plus_error = max_mious - mean_mious
    minus_error = mean_mious - min_mious
    plus_minus_error = np.stack((minus_error, plus_error), axis=0)
    plt.errorbar(sizes, mean_mious, yerr=plus_minus_error, label=label, c=c)
    plt.scatter(sizes, mean_mious, c=c)


def plot_table(mious_dict, which_size_index=-1):
    ious = []
    for k, v in mious_dict.items():
        ious.append(v["ious"][which_size_index])
    ious_array = np.stack(ious, axis=0)
    mean_across_samples = np.mean(ious_array, axis=0)
    sdev_across_samples = np.std(ious_array, axis=0)
    print(mean_across_samples)
    print(sdev_across_samples)
    for m, s in zip(mean_across_samples, sdev_across_samples):
        print(f"${100*m:.1f}\% \pm {100*s:.1f}\%$")


def plot_aggregate_confusion_matrix(dataset, size=0.8):
    confusions = []
    for k, v in dataset.items():
        for thing in v:
            if thing["size"] == size:
                confusions.append(thing["confusion"])
    confusion = np.stack(confusions, axis=2)
    confusion = np.sum(confusion, axis=2)

    extra_artists = make_confusion_matrix(
        confusion,
        categories=("Background", "Fuel", "Trunks", "Canopies"),
        count=False,
        cmap="Blues",
        norm=None,
        xyplotlabels=True,
        cbar=False,
    )
    plt.xlabel("Predicted", fontsize=20)
    plt.ylabel("True", fontsize=20)
    plt.savefig(
        "vis/confusion_matrix.pdf",
        bbox_extra_artists=extra_artists,
        bbox_inches="tight",
    )
    plt.savefig(
        "vis/confusion_matrix.png",
        bbox_extra_artists=extra_artists,
        bbox_inches="tight",
    )

    fuel_confusion = np.zeros((2, 2))
    fuel_confusion[0, 0] = confusion[1, 1]
    fuel_confusion[1, 0] = confusion[0, 1] + np.sum(confusion[2:, 1])
    fuel_confusion[0, 1] = confusion[1, 0] + np.sum(confusion[1, 2:])
    fuel_confusion[1, 1] = (
        confusion[0, 0]
        + np.sum(confusion[2:, 2:])
        + np.sum(confusion[2:, 0])
        + np.sum(confusion[0, 2:])
    )
    fuel_miou = compute_mIoU(fuel_confusion)
    total_miou = compute_mIoU(confusion)
    print(fuel_miou)

    make_confusion_matrix(
        fuel_confusion,
        categories=("Fuel", "Not fuel"),
        count=False,
        cmap="Blues",
        norm=None,
        xyplotlabels=True,
        percent=True,
    )
    plt.savefig("vis/fuel_confusion.png")


with open(SWEEP_FILE, "rb") as infile:
    cut_results = pickle.load(infile)

with open("res/test.pkl", "rb") as infile:
    results = pickle.load(infile)

cut = cut_results["CUT_pretrain_"]
synthetic = results["synthetic_pretrain_"]
setes_fonte = results[""]

cut_ious_dict = produce_mious(cut, CUT_SIZES)
synthetic_ious_dict = produce_mious(synthetic)
setes_fonte_ious_dict = produce_mious(setes_fonte)

plot_mious(setes_fonte_ious_dict, "Only Setes Fontes", c="C1")
plot_mious(synthetic_ious_dict, "Synthetic pretraining", c="C0")
plot_mious(cut_ious_dict, "CUT pretrained", c="C2")

plt.scatter(0, RUI_MIOU, label="Only synthetic", c="C0")
plt.xlabel("Number of Setes Fontes training images")
plt.ylabel("Test mIoU")
plt.legend(prop={"size": 15})
plt.savefig("vis/synthetic_experiments_mious.pdf")
plt.savefig("vis/synthetic_experiments_mious.png")
plt.show()


plot_table(setes_fonte_ious_dict)
plot_table(synthetic_ious_dict)


plot_aggregate_confusion_matrix(setes_fonte)
