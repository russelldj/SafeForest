import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from safeforest.config import SAFEFOREST_DATA_FOLDER
from sklearn.mixture import GaussianMixture

DEFAULT_POINTCLOUD = Path(
    SAFEFOREST_DATA_FOLDER,
    "datasets/portugal_UAV_12_21/derived/safe_forest_2/slam_outputs/PointCloud/2022-01-14-17-44/map_64.txt",
)


def to_unit(data):
    """Rescale to be in the range (0, 1)
    """
    data_min = np.min(data)
    data_max = np.max(data)
    extent = data_max - data_min
    data = (data - data_min) / extent
    return data


def show_GMM_isocontours(
    xyz_data,
    n_components=5,
    show_cloud=False,
    show_volume=False,
    sampling_resolution=3,
    isocontour_quantiles=(0.9999, 0.999999),
    downsample_to_fraction=0.05,
):
    # Fit a GMM to the existings points
    num_points = xyz_data.shape[0]
    retain_indices = np.random.choice(
        num_points, int(num_points * downsample_to_fraction)
    )
    xyz_data = xyz_data[retain_indices]

    if show_cloud:
        cloud = pv.PolyData(xyz_data)
        cloud["elevation"] = xyz_data[:, 2]
        cloud.plot(eye_dome_lighting=True)

    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(xyz_data)

    min_extents = np.min(xyz_data, axis=0)
    max_extents = np.max(xyz_data, axis=0)

    dims = ((max_extents - min_extents) / sampling_resolution).astype(int)
    probs_volume = pv.UniformGrid(
        dims=dims,
        origin=min_extents,
        spacing=(sampling_resolution, sampling_resolution, sampling_resolution),
    )
    grids = np.stack((probs_volume.x, probs_volume.y, probs_volume.z), axis=1)

    probs = gmm.score_samples(grids)

    probs_volume.point_data["probabilities"] = probs
    probs_volume.set_active_scalars("probabilities")

    if show_volume:
        # TODO unitize
        probs_volume.plot(volume=True)
    isocontours = np.quantile(probs, isocontour_quantiles)
    probs_contours = probs_volume.contour(isocontours, scalars=probs)
    probs_contours.plot(opacity=0.5)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointcloud", type=Path, default=DEFAULT_POINTCLOUD)
    parser.add_argument("--n-comps", type=int, default=1)
    args = parser.parse_args()
    return args


def main(file, n_comps):
    """"""
    data = pd.read_csv(file, names=("x", "y", "z", "count", "unixtime"))
    xyz = data.iloc[:, :3]
    xyz = xyz.to_numpy()

    show_GMM_isocontours(
        xyz, n_components=800, downsample_to_fraction=0.01, show_cloud=True
    )


if __name__ == "__main__":
    args = parse_args()
    main(args.pointcloud, args.n_comps)
