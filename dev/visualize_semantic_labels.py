"""
Show the segmentation labels from TartanAir to see if they have 
Any semantic meaning
"""

from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from tqdm import tqdm
from ubelt.util_path import ensuredir

SEG_MAP = np.loadtxt("dev/seg_rgbs.txt")


def visualize(seg_dir, image_dir, output_dir, seg_map=SEG_MAP):
    ensuredir(output_dir)
    seg_files = sorted(seg_dir.glob("*.npy"))
    image_files = sorted(image_dir.glob("*.png"))
    if len(seg_files) != len(image_files):
        raise ValueError("Different length inputs")

    for seg_file, image_file in tqdm(zip(seg_files, image_files), total=len(seg_files)):
        seg = np.load(seg_file)
        img = io.imread(image_file)
        vis_seg = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        for i in range(256):
            vis_seg[seg == i, :] = seg_map[i]

        concat = np.concatenate((img, vis_seg), axis=1)
        savepath = output_dir.joinpath(image_file.name)
        io.imsave(savepath, concat)


# SEG_DIR = Path("data/P001/seg_left")
# IMAGE_DIR = Path("data/P001/image_left")
# OUTPUT_DIR = Path("vis/P001")
# visualize(SEG_DIR, IMAGE_DIR, OUTPUT_DIR)

SEG_DIR = Path("data/P002/seg_left")
IMAGE_DIR = Path("data/P002/image_left")
OUTPUT_DIR = Path("vis/P002")
visualize(SEG_DIR, IMAGE_DIR, OUTPUT_DIR)

SEG_DIR = Path("data/P006/seg_left")
IMAGE_DIR = Path("data/P006/image_left")
OUTPUT_DIR = Path("vis/P006")
visualize(SEG_DIR, IMAGE_DIR, OUTPUT_DIR)
