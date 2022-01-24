import argparse
from pathlib import Path
from cv2 import imread, imwrite

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial.transform import Rotation as R
from sklearn.feature_extraction import image
from ubelt import timestamp
from show_seg_video import blend_images_gray

POINTCLOUD_FILE = Path(Path.home(), "Downloads/fullCloud_labeled.txt")
CAMERA_TRAJECTORY_FILE = Path(
    Path.home(),
    "data/SafeForestData/datasets/portugal_UAV_12_21/derived/safe_forest_2/slam_outputs/Odom_camera_left/2022-01-14-17-44/odom.txt",
)
LIDAR_TRAJECTORY_FILE = Path(
    Path.home(),
    "data/SafeForestData/datasets/portugal_UAV_12_21/derived/safe_forest_2/slam_outputs/Odom_lidar/2022-01-14-17-44/odom.txt",
)

IMAGE_FOLDER = Path(Path.home(), "data/SafeForestData/temp/all_portugal_2_images")
COLOR_MAP = np.array([[128, 128, 128], [13, 53, 26], [184, 61, 245]])
K = np.array([[719.4674, 0, 682.85536], [0, 719.4674, 555.98205], [0, 0, 1]])
IMSIZE = (1384, 1032)


def check_points(x, y, imsize):
    return np.all((x >= 0, x < imsize[0], y >= 0, y < imsize[1]))


def read_trajectory(file):
    labels = ("X", "Y", "Z", "q_x", "q_y", "q_z", "q_w", "timeStamp")
    data = pd.read_csv(file, names=labels)
    locs = data.iloc[:, :3].to_numpy()
    quats = data.iloc[:, 3:7].to_numpy()
    timestamps = data.iloc[:, 7].to_numpy()
    rots = [R.from_quat(q).as_matrix() for q in quats]
    rots = np.stack(rots, axis=2)

    return locs, rots, timestamps


def read_image(
    folder, timestamp,
):
    image_names = sorted(Path(folder).glob("*png"))
    stamps = [int(x.stem) for x in image_names]
    stamps = np.array(stamps)
    dists = np.abs(stamps - timestamp)
    best_match = np.argmin(dists)
    imname = str(image_names[best_match])
    img = imread(imname)
    return img


def read_pointcloud(file):
    data = pd.read_csv(file, names=("x", "y", "z", "labels"))
    xyzs = data.iloc[:, :3]
    labels = data.iloc[:, 3]
    xyzs = xyzs.to_numpy()
    labels = labels.to_numpy()
    return xyzs, labels


def project_points(points, colors, loc, rot, K, imsize):
    """
    imsize: tuple 
        (W, H)
    """
    points = points - loc

    local_points = np.linalg.inv(rot) @ points.transpose()
    proj = K @ local_points
    im_space = proj[:2] / proj[2]
    accumulator = np.zeros((3, imsize[0], imsize[1]))
    # TODO Vectorized
    for i, (x, y) in enumerate(im_space.transpose()):
        if proj[0, i] > 0 and check_points(x, y, IMSIZE):
            accumulator[:, int(x), int(y)] = colors[i]
    accumulator = np.transpose(accumulator, (2, 1, 0))
    return accumulator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointcloud-file", default=POINTCLOUD_FILE, type=Path)
    parser.add_argument("--trajectory-file", default=CAMERA_TRAJECTORY_FILE, type=Path)
    args = parser.parse_args()
    return args


def main(pointcloud_file, trajectory_file):
    # Hack because only one contains valid timestamps
    locs, rots, _ = read_trajectory(trajectory_file)
    _, _, timestamps = read_trajectory(LIDAR_TRAJECTORY_FILE)
    xyzs, labels = read_pointcloud(pointcloud_file)
    colors = COLOR_MAP[labels]
    pc = pv.PolyData(xyzs)
    pc["class"] = colors

    traj = pv.PolyData(locs)
    traj["index"] = np.arange(locs.shape[0])

    for i in range(0, len(locs), 25):
        image = read_image(IMAGE_FOLDER, timestamps[i])
        projected = project_points(xyzs, colors, locs[i], rots[..., i], K, IMSIZE)
        blended = blend_images_gray(image, projected, alpha=0.4)
        filename = f"vis/projections/{i:06d}.png"
        imwrite(filename, blended)

    plotter = pv.Plotter()
    plotter.add_mesh(pc, scalars="class", rgb=True)
    plotter.add_mesh(traj, scalars="index")
    plotter.show()


if __name__ == "__main__":
    args = parse_args()
    main(args.pointcloud_file, args.trajectory_file)
