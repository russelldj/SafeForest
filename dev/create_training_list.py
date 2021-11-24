import argparse
from pathlib import Path
import numpy as np
from functools import reduce
import os

from video_utils import write_imagelist_to_video


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-dir", type=Path)
    parser.add_argument("--seg-dir", type=Path)
    parser.add_argument("--output-train-file", type=Path)
    parser.add_argument("--output-val-file", type=Path)
    parser.add_argument("--output-test-file", type=Path)
    parser.add_argument(
        "--test-frac", type=float, default=0.2, help="Fraction to use for testing"
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Fraction of non-test data to use for training. The rest is validation.",
    )
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()
    return args


def describe_max(seg_files):
    import cv2

    current_max = 0
    for sf in seg_files:
        data = cv2.imread(str(sf))
        current_max = max(current_max, np.amax(data))
    print(f"The max label is {current_max}")


def main(
    rgb_dir,
    seg_dir,
    train_file,
    val_file,
    test_file,
    train_frac,
    test_frac,
    random_seed,
    write_test_video=True,
):
    np.random.seed(random_seed)

    rgb_files = np.asarray(sorted(rgb_dir.glob("*png")))
    seg_files = np.asarray(sorted(seg_dir.glob("*png")))

    all_files = rgb_files.tolist() + seg_files.tolist()
    common_root = os.path.commonpath(all_files)

    rgb_files = np.asarray([x.relative_to(common_root) for x in rgb_files])
    seg_files = np.asarray([x.relative_to(common_root) for x in seg_files])
    num_train = int((1 - test_frac) * rgb_files.shape[0])
    rgb_files_test = rgb_files[num_train:]
    seg_files_test = seg_files[num_train:]

    rgb_files_train_val = rgb_files[:num_train]
    seg_files_train_val = seg_files[:num_train]

    random_vals = np.random.uniform(size=(num_train,))
    train_ids = random_vals < train_frac
    val_ids = np.logical_not(train_ids)

    with open(train_file, "w") as outfile_h:
        for r, s in zip(rgb_files_train_val[train_ids], seg_files_train_val[train_ids]):
            outfile_h.write(f"{r},{s}\n")

    with open(val_file, "w") as outfile_h:
        for r, s in zip(rgb_files_train_val[val_ids], seg_files_train_val[val_ids]):
            outfile_h.write(f"{r},{s}\n")

    with open(test_file, "w") as outfile_h:
        for r, s in zip(rgb_files_test, seg_files_test):
            outfile_h.write(f"{r},{s}\n")


    if write_test_video:
        breakpoint()
        rgb_files_test = [Path(common_root, f) for f in rgb_files_test]
        test_video_filename = str(test_file.with_suffix(".mp4"))
        write_imagelist_to_video(rgb_files_test, test_video_filename)

if __name__ == "__main__":
    args = parse_args()
    main(
        args.rgb_dir,
        args.seg_dir,
        args.output_train_file,
        args.output_val_file,
        args.output_test_file,
        args.train_frac,
        args.test_frac,
        args.random_seed,
    )
