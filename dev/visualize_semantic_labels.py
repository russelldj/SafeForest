"""
Show the segmentation labels from TartanAir to see if they have 
Any semantic meaning
"""
import argparse
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from tqdm import tqdm
from ubelt.util_path import ensuredir

SEG_MAP = "dev/seg_rgbs.txt"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seg-dir", type=Path)
    parser.add_argument("--seg-map", default=SEG_MAP, type=Path)
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


def visualize(seg_dir, image_dir, output_dir, seg_map_file=SEG_MAP):
    seg_map = np.loadtxt(seg_map_file)
    ensuredir(output_dir)
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
        vis_seg = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        for i in range(256):
            vis_seg[seg == i, :] = seg_map[i]

        concat = np.concatenate((img, vis_seg), axis=1)
        savepath = output_dir.joinpath(image_file.name)
        io.imsave(savepath, concat)


if __name__ == "__main__":
    args = parse_args()
    seg_map = np.loadtxt(args.seg_map)
    show_colormaps(seg_map)
    visualize(args.seg_dir, args.image_dir, args.output_dir, args.seg_map)
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
