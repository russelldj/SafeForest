import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ubelt as ub
from dev.dataset_creation.merge_classes import visualize_with_palette
from safeforest.vis.visualize_classes import blend_images_gray
from imageio import imread, imwrite
from safeforest.config import RUI_YAMAHA_CLASSES, RUI_YAMAHA_PALETTE
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

plt.style.use("./dev/report.mplstyle")


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

    common_root = os.path.commonpath(img_files + ann_files)

    # for i in tqdm(indices[0]):
    #    ann_file = ann_files[i]
    #    img_file = img_files[i]
    #    output_ann_file = Path(output_dir, ann_file.relative_to(common_root))
    #    output_img_file = Path(output_dir, img_file.relative_to(common_root))
    #    ub.ensuredir(output_ann_file.parent, mode=0o0755)
    #    ub.ensuredir(output_img_file.parent, mode=0o0755)
    #    ub.cmd(f"cp {img_file} {output_img_file}")
    #    ub.cmd(f"cp {ann_file} {output_ann_file}")
    #
    #     label_img = visualize_with_palette(ann_img, RUI_YAMAHA_PALETTE)
    #     blended = blend_images_gray(img_img, label_img)
    #     name = img_files[i].name
    #     output_file = Path(output_dir, name)
    #     imwrite(output_file, blended)

    plt.plot(fracs, label="Per-frame fraction")
    plt.plot(filtered_fracs, label="Gaussian-smoothed fraction")
    plt.legend()
    plt.xlabel("Frame")
    plt.ylabel("Fraction of pixels labeled as trunk")
    plt.savefig("vis/trunk_pixel_fraction.png")
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args.ann_dir, args.img_dir, args.output_dir)
