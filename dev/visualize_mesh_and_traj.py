import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial.transform import Rotation as R

POINTCLOUD_FILE = Path(Path.home(), "Downloads/fullCloud_labeled.txt")
TRAJECTORY_FILE = Path(
    Path.home(),
    "data/SafeForestData/datasets/portugal_UAV_12_21/derived/safe_forest_2/slam_outputs/Odom_camera_left/2022-01-14-17-44/odom.txt",
)
COLOR_MAP = np.array([[128, 128, 128], [0, 255, 0], [150, 75, 0]])


def read_trajectory(file):
    labels = ("X", "Y", "Z", "q_x", "q_y", "q_z", "q_w", "timeStamp")
    data = pd.read_csv(file, names=labels)
    locs = data.iloc[:, :3].to_numpy()
    quats = data.iloc[:, 3:7].to_numpy()
    rots = [R.from_quat(q).as_matrix() for q in quats]
    rots = np.stack(rots, axis=2)

    return locs, rots


def read_pointcloud(file):
    data = pd.read_csv(file, names=("x", "y", "z", "labels"))
    xyzs = data.iloc[:, :3]
    labels = data.iloc[:, 3]
    xyzs = xyzs.to_numpy()
    labels = labels.to_numpy()
    return xyzs, labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointcloud-file", default=POINTCLOUD_FILE, type=Path)
    parser.add_argument("--trajectory-file", default=TRAJECTORY_FILE, type=Path)
    args = parser.parse_args()
    return args


def main(pointcloud_file, trajectory_file):
    locs, rots = read_trajectory(trajectory_file)
    xyzs, labels = read_pointcloud(pointcloud_file)
    colors = COLOR_MAP[labels]
    pc = pv.PolyData(xyzs)
    pc["class"] = colors

    traj = pv.PolyData(locs)
    traj["index"] = np.arange(locs.shape[0])

    plotter = pv.Plotter()
    plotter.add_mesh(pc, scalars="class", rgb=True)
    plotter.add_mesh(traj, scalars="index")
    plotter.show()


if __name__ == "__main__":
    args = parse_args()
    main(args.pointcloud_file, args.trajectory_file)
