import pyvista as pv
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

from safeforest.config import PALETTE_MAP
from dev.dataset_creation.merge_classes import compute_nearest_class_1D

DATA_FILE = "/home/frc-ag-1/experiments/octomap_outputs/time_reindexed_45_deg_2_processed.csv"  # "/home/frc-ag-1/experiments/octomap_outputs/example.txt"
OUTPUT_FILE = "/home/frc-ag-1/experiments/octomap_outputs/time_reindexed_45_deg_2_processed_indices.csv"  # "vis/semantic_points.csv"
RGB = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument(
        "--palette-name", choices=PALETTE_MAP.keys(), default="semfire-ros-w-ignore"
    )
    args = parser.parse_args()
    return args


def main(input_file, output_file, rgb, palette_name):
    df = pd.read_csv(input_file)
    header = df.columns.to_numpy().tolist() + ["semantic_ID"]
    xyz = df.iloc[:, :3].to_numpy()
    color = df.iloc[:, 3:6].to_numpy()
    semantic_color = df.iloc[:, 6:9].to_numpy()
    plotter = pv.Plotter()
    points = pv.PolyData(xyz)
    # ids = df["semantic_ID"]
    palette = PALETTE_MAP[palette_name]

    ids = compute_nearest_class_1D(semantic_color, palette)
    df["semantic_ID"] = ids
    if rgb:
        plotter.add_mesh(points, scalars=semantic_color / 255.0, rgb=True)
    else:
        plotter.add_mesh(points, scalars=ids)
    plotter.show()

    output = np.concatenate((xyz,), axis=1)
    if output_file is not None:
        df.to_csv(output_file, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
