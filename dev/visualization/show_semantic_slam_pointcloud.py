from matplotlib import offsetbox
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
    parser.add_argument(
        "--show-id",
        action="store_true",
        help="Show the class ID rather than the semantic color",
    )
    parser.add_argument(
        "--show-rgb-color",
        action="store_true",
        help="Show the RGB color rather than the semantic color",
    )
    parser.add_argument(
        "--palette-name", choices=PALETTE_MAP.keys(), default="semfire-ros-w-ignore"
    )
    parser.add_argument("--exclude-background-id", type=int)
    parser.add_argument("--screenshot-filename")
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = Path(str(args.input_file).replace(".csv", "_processed.csv"))

    if Path(args.screenshot_filename).is_dir():
        filename = Path(args.input_file).stem
        args.screenshot_filename = Path(args.screenshot_filename, filename + ".png")
    return args


def main(
    input_file,
    output_file,
    show_id,
    show_rgb_color,
    palette_name,
    exclude_background_id=None,
    screenshot_filename=None,
):
    # Read the data
    df = pd.read_csv(input_file)

    palette = PALETTE_MAP[palette_name]

    semantic_color = df.iloc[:, 6:9].to_numpy()
    ids = compute_nearest_class_1D(semantic_color, palette)
    df["semantic_ID"] = ids

    if exclude_background_id:
        not_background_mask = ids != exclude_background_id
        # not_background_inds = np.where(not_background_mask)[0]
        df = df.iloc[not_background_mask]
        semantic_color = semantic_color[not_background_mask]

    xyz = df.iloc[:, :3].to_numpy()
    if screenshot_filename is None:
        plotter = pv.Plotter()
    else:
        plotter = pv.Plotter(off_screen=True)

    points = pv.PolyData(xyz)

    if show_id:
        plotter.add_mesh(points, scalars=ids)
    else:
        if show_rgb_color:
            color = df.iloc[:, 3:6].to_numpy()
            plotter.add_mesh(points, scalars=color / 255.0, rgb=True)
        else:
            plotter.add_mesh(points, scalars=semantic_color / 255.0, rgb=True)

    if screenshot_filename is None:
        plotter.show()
    else:
        plotter.screenshot(filename=screenshot_filename)

    if output_file is not None:
        df.to_csv(output_file, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
