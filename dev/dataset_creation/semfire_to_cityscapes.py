import argparse
import os
from cProfile import label
from pathlib import Path

import numpy as np
import ubelt as ub
from imageio import imread
from safeforest.dataset_generation.file_utils import (
    get_files,
    make_cityscapes_file,
    write_cityscapes_file,
)
from safeforest.dataset_generation.split_utils import get_is_train_array
from sklearn.model_selection import ParameterGrid

INPUT_IMG_FOLDER = Path(
    "/home/frc-ag-1/data/SafeForestData/datasets/semfire_segmentation/original/2021_sete_fontes_forest/img"
)
INPUT_LABEL_FOLDER = Path(
    "/home/frc-ag-1/data/SafeForestData/datasets/semfire_segmentation/original/2021_sete_fontes_forest/lbl"
)
OUTPUT_FOLDER = Path(
    "/home/frc-ag-1/data/SafeForestData/datasets/semfire_segmentation/derived/training/2021_sete_fontes_forest"
)

IMG_SUBFOLDER = "img"
LABEL_SUBFOLDER = "lbl"
TRAIN_FRAC = 0.9


def parse_args():
    parser = argparse.ArgumentParser(
        description="Takes in folders of raw images and labeled images suitable"
                    " for semantic segmentation training, then stores them in"
                    " an output folder in the specific CityScapes directory"
                    " format. mmseg models are set up to read this format."
    )
    parser.add_argument("--input-img-folder", default=INPUT_IMG_FOLDER, type=Path)
    parser.add_argument("--input-lbl-folder", default=INPUT_LABEL_FOLDER, type=Path)
    parser.add_argument("--output-folder", default=OUTPUT_FOLDER, type=Path)
    parser.add_argument("--train-frac", default=[TRAIN_FRAC], type=float, nargs="+")
    parser.add_argument(
        "--img-prefix", default="", help="Prefix to strip from image files"
    )
    parser.add_argument(
        "--shift",
        help="number of indices to shift data before computing train/test split. The shift'th index will be the first one",
        type=int,
        default=[0],
        nargs="+",
    )
    parser.add_argument("--write-RG-only", action="store_true")
    parser.add_argument("--run_sweep", action="store_true")
    parser.add_argument(
        "--force-copy",
        help="Normally this tool symlinks files, if this flag is given"
             " they will instead be copied.",
        action="store_true",
    )
    args = parser.parse_args()
    return args


def main(
    img_folder: Path,
    label_folder: Path,
    output_folder: Path,
    *,
    img_prefix: str,
    write_RG_only: bool = False,
    force_copy: bool = False,
    use_filename_as_index: bool = False,
    verbose=True,
    seed=0,
    shift=0,
    train_frac=TRAIN_FRAC,
    remove_existing_output_dir=True,
):
    """
    input_folder:
        The folder containing img, lbl, etc.
    output_folder:
        Where to write the simlinked output structure 
    use_filename_as_index:
        Extract the filename as index instead of using position in list
    """
    if remove_existing_output_dir and os.path.exists(output_folder):
        ub.delete(output_folder)

    img_files, label_files = [get_files(x, "*") for x in (img_folder, label_folder)]

    if len(img_files) != len(label_files):
        breakpoint()
        raise ValueError("Different number of files")
    num_total = len(img_files)
    num_train = int(train_frac * num_total)
    is_train_array = get_is_train_array(num_total, num_train, seed=seed, shift=shift)

    for i, (img_file, label_file) in enumerate(zip(img_files, label_files)):
        if verbose:
            print(f"img_file: {img_file}, label_file: {label_file}")

        if use_filename_as_index:
            if img_prefix != "":
                stem = img_file.stem.replace(img_prefix, "")
            else:
                stem = img_file.stem

            index = int(stem)
        else:
            index = i

        is_train = is_train_array[index]

        cityscapes_kwargs = {"output_folder": output_folder,
                             "index": index,
                             "is_train": is_train}
        if write_RG_only:
            img = imread(img_file)
            # TODO consider trying to optimize this
            img[..., 2] = 0
            write_cityscapes_file(img, is_ann=False, **cityscapes_kwargs)
        else:
            make_cityscapes_file(img_file,
                                 is_ann=False,
                                 force_copy=force_copy,
                                 **cityscapes_kwargs)

        make_cityscapes_file(label_file,
                             is_ann=True,
                             force_copy=force_copy,
                             **cityscapes_kwargs)


if __name__ == "__main__":
    args = parse_args()
    if args.run_sweep:
        print(args.shift)
        shifts = (0, 30, 60, 90, 120)
        train_fracs = (0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8)
        param_dict = {"shift": shifts, "train_frac": train_fracs}
        param_grid = list(ParameterGrid(param_dict))
        for params in param_grid:
            output_folder = Path(
                args.output_folder,
                f"train_{params['shift']:06d}_{params['train_frac']:2f}",
            )
            print(output_folder)
            main(
                args.input_img_folder,
                args.input_lbl_folder,
                output_folder,
                img_prefix=args.img_prefix,
                write_RG_only=args.write_RG_only,
                force_copy=args.force_copy,
                **params,
            )
    else:
        main(
            args.input_img_folder,
            args.input_lbl_folder,
            args.output_folder,
            img_prefix=args.img_prefix,
            write_RG_only=args.write_RG_only,
            force_copy=args.force_copy,
            train_frac=args.train_frac[0],
            shift=args.shift[0],
        )
