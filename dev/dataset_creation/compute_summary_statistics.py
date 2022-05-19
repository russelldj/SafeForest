import argparse
from imageio import imread
import numpy as np
from tqdm import tqdm

from safeforest.dataset_generation.file_utils import get_files
from safeforest.dataset_generation.split_utils import get_is_train_array


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir")
    parser.add_argument("--num-files", default=500, type=int)
    args = parser.parse_args()
    return args


def main(images, num_files):
    files = get_files(images, "*")

    files_to_use = get_is_train_array(len(files), num_files)
    files = np.array(files)[files_to_use]

    imgs = [imread(x) for x in tqdm(files)]
    imgs = np.concatenate(imgs, axis=0)  # tile vertically
    mean = np.mean(imgs, axis=(0, 1))
    std = np.std(imgs, axis=(0, 1))
    print(f"mean: {mean}, stdev: {std}")


if __name__ == "__main__":
    args = parse_args()
    main(args.images_dir, args.num_files)

