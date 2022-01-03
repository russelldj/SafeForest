import argparse
from pathlib import Path

import numpy as np
import ubelt as ub
from imageio import imread, imwrite
from tqdm import tqdm

"""
remap classes
"""
# Taken from https://www.geeksforgeeks.org/python-key-value-pair-using-argparse/
# create a keyvalue class
class keyvalue(argparse.Action):
    # Constructor calling
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())

        for value in values:
            # split it into key and value
            key, value = value.split("=")
            # assign into dictionary
            getattr(namespace, self.dest)[int(key)] = int(value)


def remap_image(input_file, output_file, remap):
    img = imread(input_file)
    remapped_img = remap[img].astype(np.uint8)
    ub.ensuredir(output_file.parents[0], mode=0o0755)
    imwrite(output_file, remapped_img)


def main(annotation_dir, output_dir, remap):
    numpy_remap = np.ones((len(remap),), dtype=int) * -1

    for k, v in remap.items():
        # TODO think about whether this is right
        numpy_remap[k] = v

    assert not -1 in numpy_remap

    input_files = list(annotation_dir.glob("**/*.png"))
    output_files = [x.relative_to(annotation_dir) for x in input_files]
    output_files = [Path(output_dir, x) for x in output_files]
    [
        remap_image(i_f, o_f, numpy_remap)
        for i_f, o_f in tqdm(zip(input_files, output_files), total=len(input_files))
    ]

def merge_datasets(img_folders, ann_folders, output_folder):
    asdfasdf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        required=True,
        help="The top level directory containing all the annotations",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write the remapped data. The structure will be the same as the input",
    )
    parser.add_argument("--remap", nargs="*", action=keyvalue)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args.annotation_dir, args.output_dir, args.remap)
