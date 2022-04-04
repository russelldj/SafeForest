"""
Show the segmentation labels from TartanAir to see if they have 
Any semantic meaning
"""
import argparse
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from safeforest.config import PALETTE_MAP
from skimage import io
from tqdm import tqdm
from safeforest.dataset_generation.file_utils import ensure_dir_normal_bits
from safeforest.vis.visualize_classes import visualize_with_palette


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seg-dir", type=Path, required=True)
    parser.add_argument(
        "--palette", default="rui", choices=PALETTE_MAP.keys(), type=str
    )
    args = parser.parse_args()
    return args


def show_colormaps(seg_map, num_classes=7):
    square_size = int(np.ceil(np.sqrt(num_classes)))
    vis = np.zeros((square_size, square_size, 3))
    for index in range(num_classes):
        i = index // square_size
        j = index % square_size
        vis[i, j] = seg_map[index]
    vis = vis.astype(np.uint8)
    vis = np.repeat(np.repeat(vis, repeats=100, axis=0), repeats=100, axis=1)
    plt.imshow(vis)
    plt.show()
    breakpoint()


def load_png_npy(filename):
    if filename.suffix == ".npy":
        return np.load(filename)
    elif filename.suffix in (".png", ".jpg", ".jpeg"):
        return io.imread(filename)


def visualize(seg_dir, image_dir, output_dir, palette_name="rui"):
    palette = PALETTE_MAP[palette_name]
    ensure_dir_normal_bits(output_dir)
    seg_files = sorted(
        list(Path(seg_dir).glob("*.npy")) + list(Path(seg_dir).glob("*.png"))
    )
    image_files = sorted(Path(image_dir).glob("*.png"))
    if len(seg_files) != len(image_files):
        raise ValueError(
            f"Different length inputs, {len(seg_files)}, {len(image_files)}"
        )

    for seg_file, image_file in tqdm(zip(seg_files, image_files), total=len(seg_files)):
        seg = load_png_npy(seg_file)
        img = io.imread(image_file)
        # vis_seg = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        vis_seg = visualize_with_palette(seg, palette)
        # for i in range(seg_map.shape[0]):
        #    vis_seg[seg == i, :] = seg_map[i]

        concat = np.concatenate((img, vis_seg), axis=1)
        savepath = output_dir.joinpath(image_file.name)
        io.imsave(savepath, concat)


if __name__ == "__main__":
    args = parse_args()
    # seg_map = np.loadtxt(args.seg_map)
    # show_colormaps(seg_map)
    visualize(args.seg_dir, args.image_dir, args.output_dir, args.palette)
    # SEG_DIR = Path("data/P001/seg_left")
    # IMAGE_DIR = Path("data/P001/image_left")
    # OUTPUT_DIR = Path("vis/P001")
    # visualize(SEG_DIR, IMAGE_DIR, OUTPUT_DIR)

    # SEG_DIR = Path("data/P002/seg_left")
    # IMAGE_DIR = Path("data/P002/image_left")
    # OUTPUT_DIR = Path("vis/P002")
    # visualize(SEG_DIR, IMAGE_DIR, OUTPUT_DIR)
    #
    # SEG_DIR = Path("data/P006/seg_left")
    # IMAGE_DIR = Path("data/P006/image_left")
    # OUTPUT_DIR = Path("vis/P006")
    # visualize(SEG_DIR, IMAGE_DIR, OUTPUT_DIR)
