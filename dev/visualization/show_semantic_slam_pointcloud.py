import pyvista as pv
import pandas as pd
import numpy as np

from safeforest.config import SEMFIRE_ROS_PALETTE
from dev.dataset_creation.merge_classes import compute_nearest_class_1D

DATA_FILE = "/home/frc-ag-1/experiments/octomap_outputs/time_reindexed_45_deg_2_processed.csv"  # "/home/frc-ag-1/experiments/octomap_outputs/example.txt"
OUTPUT_FILE = "/home/frc-ag-1/experiments/octomap_outputs/time_reindexed_45_deg_2_processed_indices.csv"  # "vis/semantic_points.csv"
RGB = True


df = pd.read_csv(DATA_FILE)
header = df.columns.to_numpy().tolist() + ["semantic_ID"]
xyz = df.iloc[:, :3].to_numpy()
color = df.iloc[:, 3:6].to_numpy()
semantic_color = df.iloc[:, 6:9].to_numpy()
plotter = pv.Plotter()
points = pv.PolyData(xyz)
# ids = df["semantic_ID"]
ids = compute_nearest_class_1D(semantic_color, SEMFIRE_ROS_PALETTE)
df["semantic_ID"] = ids
if RGB:
    plotter.add_mesh(points, scalars=semantic_color / 255.0, rgb=True)
else:
    plotter.add_mesh(points, scalars=ids)
plotter.show()

output = np.concatenate((xyz,), axis=1)
df.to_csv(OUTPUT_FILE, index=False)
