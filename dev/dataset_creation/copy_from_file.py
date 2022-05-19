import argparse
import os
from pathlib import Path

import numpy as np
import shutil

from safeforest.dataset_generation.file_utils import ensure_dir_normal_bits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", type=Path)
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--output-folder", type=Path)
    args = parser.parse_args()
    return args


def main(input_folder, input_file, output_folder):
    """"""
    ensure_dir_normal_bits(output_folder)

    with open(input_file, "r") as infile_h:
        for l in infile_h:
            l = l.replace("sete_pontes", "rgb").strip()
            input_path = Path(input_folder, l)
            output_path = Path(output_folder, l)
            print(input_path, output_path, end="")
            shutil.copyfile(input_path, output_path)


if __name__ == "__main__":
    args = parse_args()
    main(args.input_folder, args.input_file, args.output_folder)
