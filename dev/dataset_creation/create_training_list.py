import argparse
from pathlib import Path
import numpy as np
from functools import reduce
import os

import ubelt as ub
from safeforest.dataset_generation.file_utils import get_train_val_test

from dev.utils.video_utils import write_imagelist_to_video

TRAIN_VAL_TEST = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-dir", type=Path)
    parser.add_argument("--seg-dir", type=Path)
    parser.add_argument("--output-train-file", type=Path)
    parser.add_argument("--output-val-file", type=Path)
    parser.add_argument("--output-test-file", type=Path)
    parser.add_argument(
        "--output-symlink-dir", type=Path, help="Where to write the symlinked files"
    )
    parser.add_argument(
        "--test-frac", type=float, default=0.2, help="Fraction to use for testing"
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Fraction of non-test data to use for training. The rest is validation.",
    )
    parser.add_argument("--symlink", action="store_true")
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


def get_train_val_test(
    input_rgb_dir,
    input_seg_dir,
    test_frac,
    train_frac,
    extension="*.png",
    shuffle_test=True,
):
    rgb_files = np.asarray(sorted(input_rgb_dir.glob(extension)))
    seg_files = np.asarray(sorted(input_seg_dir.glob(extension)))

    all_files = rgb_files.tolist() + seg_files.tolist()
    common_root = os.path.commonpath(all_files)

    rgb_files = np.asarray([x.relative_to(common_root) for x in rgb_files])
    seg_files = np.asarray([x.relative_to(common_root) for x in seg_files])
    num_train = int((1 - test_frac) * rgb_files.shape[0])
    if shuffle_test:
        train_val_inds = np.zeros((rgb_files.shape[0],), dtype=bool)
        train_val_locs = np.argsort(np.random.uniform(size=(len(rgb_files),)))[
            :num_train
        ]
        train_val_inds[train_val_locs] = True
        test_inds = np.logical_not(train_val_inds)
        rgb_files_test = rgb_files[test_inds]
        seg_files_test = seg_files[test_inds]
        rgb_files_train_val = rgb_files[train_val_inds]
        seg_files_train_val = seg_files[train_val_inds]
    else:
        rgb_files_test = rgb_files[num_train:]
        seg_files_test = seg_files[num_train:]
        rgb_files_train_val = rgb_files[:num_train]
        seg_files_train_val = seg_files[:num_train]

    random_vals = np.random.uniform(size=(num_train,))
    train_ids = random_vals < train_frac
    val_ids = np.logical_not(train_ids)
    return (
        (rgb_files_train_val[train_ids], rgb_files_train_val[val_ids], rgb_files_test),
        (seg_files_train_val[train_ids], seg_files_train_val[val_ids], seg_files_test),
        common_root,
    )


def symlink_mmseg_dataset(
    input_rgb_dir,
    input_seg_dir,
    output_dir,
    train_frac,
    test_frac,
    random_seed,
    write_test_video=True,
):
    """
    ├── my_dataset
    │   ├── img_dir
    │   │   ├── train
    │   │   │   ├── xxx{img_suffix}
    │   │   │   ├── yyy{img_suffix}
    │   │   │   ├── zzz{img_suffix}
    │   │   ├── val
    │   ├── ann_dir
    │   │   ├── train
    │   │   │   ├── xxx{seg_map_suffix}
    │   │   │   ├── yyy{seg_map_suffix}
    │   │   │   ├── zzz{seg_map_suffix}
    │   │   ├── val
    """

    # Consider taking in only an output directory
    np.random.seed(random_seed)
    # TODO try to reduct code reuse

    img_dir = Path(output_dir, "img_dir")
    ann_dir = Path(output_dir, "ann_dir")

    img_dir_subfolders = [Path(img_dir, x) for x in TRAIN_VAL_TEST]
    ann_dir_subfolders = [Path(ann_dir, x) for x in TRAIN_VAL_TEST]

    [ub.ensuredir(x, mode=0o0755) for x in img_dir_subfolders]
    [ub.ensuredir(x, mode=0o0755) for x in ann_dir_subfolders]

    rgb_files, seg_files, common_root = get_train_val_test(
        input_rgb_dir, input_seg_dir, test_frac, train_frac, extension="*.png"
    )

    rgb_output_paths = []

    # Do a lot of symlinking
    for folder, files in zip(img_dir_subfolders, rgb_files,):
        rgb_output_paths.append([])
        for f in files:
            output_path = Path(folder, f.name)
            info = ub.cmd(f"unlink '{output_path}'")
            output_path = str(output_path.with_suffix(""))
            output_path = output_path.replace("rgb ", "")
            output_path += "_rgb.png"
            input_path = Path(common_root, f)
            rgb_output_paths[-1].append(output_path)
            info = ub.cmd(f"ln -s '{input_path}' '{output_path}'")

    seg_output_paths = []

    # Do a lot of symlinking
    for folder, files in zip(ann_dir_subfolders, seg_files,):
        seg_output_paths.append([])
        for f in files:
            output_path = Path(folder, f.name)
            output_path = str(output_path.with_suffix(""))
            output_path = output_path.replace("segmentation ", "")
            output_path += "_segmentation.png"

            info = ub.cmd(f"unlink '{output_path}'")
            input_path = Path(common_root, f)
            seg_output_paths[-1].append(output_path)
            info = ub.cmd(f"ln -s '{input_path}' '{output_path}'")

    train_file, val_file, test_file = [
        Path(output_dir, f"{x}.txt") for x in TRAIN_VAL_TEST
    ]

    write_summary_files(
        rgb_output_paths,
        seg_output_paths,
        train_file,
        val_file,
        test_file,
        common_root,
        False,
    )

    return rgb_output_paths, seg_output_paths


def write_summary_files(
    all_rgb_files,
    all_seg_files,
    train_file,
    val_file,
    test_file,
    common_root,
    write_test_video,
):
    with open(train_file, "w") as outfile_h:
        for r, s in zip(all_rgb_files[0], all_seg_files[0]):
            outfile_h.write(f"{r},{s}\n")

    with open(val_file, "w") as outfile_h:
        for r, s in zip(all_rgb_files[1], all_seg_files[1]):
            outfile_h.write(f"{r},{s}\n")

    with open(test_file, "w") as outfile_h:
        for r, s in zip(all_rgb_files[2], all_seg_files[2]):
            outfile_h.write(f"{r},{s}\n")

    if write_test_video:
        breakpoint()
        rgb_files_test = [Path(common_root, f) for f in all_rgb_files[2]]
        test_video_filename = str(test_file.with_suffix(".mp4"))
        write_imagelist_to_video(rgb_files_test, test_video_filename)


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

    all_rgb_files, all_seg_files, common_root = get_train_val_test(
        rgb_dir, seg_dir, test_frac, train_frac
    )
    write_summary_files(
        all_rgb_files,
        all_seg_files,
        train_file,
        val_file,
        test_file,
        common_root,
        write_test_video,
    )


if __name__ == "__main__":
    args = parse_args()

    if args.symlink:
        symlink_mmseg_dataset(
            args.rgb_dir,
            args.seg_dir,
            args.output_symlink_dir,
            args.train_frac,
            args.test_frac,
            args.random_seed,
        )
    else:
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
