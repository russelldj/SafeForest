import argparse
from dev.config import RUI_YAMAHA_CLASSES, RUI_YAMAHA_PALETTE
import numpy as np
from pathlib import Path
from tqdm import tqdm

import ubelt as ub
import matplotlib.pyplot as plt

from imageio import imread, imwrite
from scipy.ndimage import gaussian_filter1d

from show_seg_video import blend_images_gray
from merge_classes import visualize_with_palette
from config import RUI_YAMAHA_PALETTE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-dir", type=Path)
    parser.add_argument("--img-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    return args


def main(ann_dir, img_dir, output_dir, ext="png"):
    img_files = sorted(img_dir.glob(f"**/*{ext}"))
    ann_files = sorted(ann_dir.glob(f"**/*{ext}"))
    # num_pixels = imread(ann_files[0]).size
    # cnts = [np.sum(imread(x) == 5) for x in tqdm(files)]
    # fracs = [x / num_pixels for x in cnts]
    # np.save("vis/fracs.npy", fracs)
    fracs = np.load("vis/fracs.npy")
    filtered_fracs = gaussian_filter1d(fracs, 30)
    indices = np.where(fracs < (filtered_fracs - 0.05))

    ub.ensuredir(output_dir, mode=0o0755)

    for i in tqdm(indices[0]):
        ann_img = imread(ann_files[i])
        img_img = imread(img_files[i])
        label_img = visualize_with_palette(ann_img, RUI_YAMAHA_PALETTE)
        blended = blend_images_gray(img_img, label_img)
        name = img_files[i].name
        output_file = Path(output_dir, name)
        imwrite(output_file, blended)

    breakpoint()

    plt.plot(fracs)
    plt.plot(filtered_fracs)
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args.ann_dir, args.img_dir, args.output_dir)
