import argparse
from imageio import imread
import numpy as np
from tqdm import tqdm

from safeforest.dataset_generation.file_utils import get_files
from safeforest.dataset_generation.split_utils import get_is_train_array


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inside mmsegmentation/configs/_base_/datasets there are"
                    " dataset config files, which include dataset stats like"
                    " img_norm_cfg = dict(mean=[123.6, 116.2, 103.5],"
                    " std=[58.3, 57.1, 57.3]). This function helps gets those"
                    " stats for new datasets."
    )
    parser.add_argument("--images-dir")
    parser.add_argument("--num-files", default=500, type=int)
    args = parser.parse_args()
    return args


def main(images, num_files):
    files = get_files(images, "*")

    # This seems like a bad thing - why is it splitting into training arrays?
    # Shouldn't it just use the files you point it towards?
    files_to_use = get_is_train_array(len(files), num_files)
    files = np.array(files)[files_to_use]

    # Calculate the weighted average of these values image by image, doing it
    # with all images at a time is not sustainable with medium numbers of
    # images
    means = []
    stds = []
    weights = []
    for x in tqdm(files):
        img = imread(x)
        means.append(np.mean(img, axis=(0, 1)))
        stds.append(np.std(img, axis=(0, 1)))
        # If the images were the same size you wouldn't need weights, but don't
        # assume that
        weights.append(img.shape[0] * img.shape[1])
    means = np.array(means)
    stds = np.array(stds)
    weights = np.array(weights) / np.max(weights)

    mean = np.average(means, axis=0, weights=weights)
    std = np.average(stds, axis=0, weights=weights)
    np.set_printoptions(precision=3)
    print(f"mean: {mean}, stdev: {std}")


if __name__ == "__main__":
    args = parse_args()
    main(args.images_dir, args.num_files)

