import argparse
from pathlib import Path

import distinctipy
import numpy as np
import pandas as pd
import pyvista as pv
from sklearn.covariance import log_likelihood
from safeforest.config import SAFEFOREST_DATA_FOLDER
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

DEFAULT_POINTCLOUD = Path(
    SAFEFOREST_DATA_FOLDER,
    "datasets/portugal_UAV_12_21/derived/safe_forest_2/slam_outputs/PointCloud/2022-01-14-17-44/map_64.txt",
)


def score_grid(min_max_extents, sampling_resolution: float, gmm: GaussianMixture):
    """
    min_max_extents:
        (np.array, np.array) The boundaries of the axis aligned cubiod to compute the sampling over
    sampling_resolution:
        The size of each grid
    ggm:
        The mixture model to sample from
    """
    min_extents, max_extents = min_max_extents

    dims = ((max_extents - min_extents) / sampling_resolution).astype(int)
    probs_volume = pv.UniformGrid(
        dims=dims,
        origin=min_extents,
        spacing=(sampling_resolution, sampling_resolution, sampling_resolution),
    )
    grids = np.stack((probs_volume.x, probs_volume.y, probs_volume.z), axis=1)

    log_likelihood = gmm.score_samples(grids)
    return probs_volume, log_likelihood


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

    probs_volume, probs = score_grid(xyz_data, sampling_resolution, gmm)
    probs_volume.point_data["probabilities"] = probs
    probs_volume.set_active_scalars("probabilities")

    if show_volume:
        # TODO unitize
        probs_volume.plot(volume=True)
    isocontours = np.quantile(probs, isocontour_quantiles)
    probs_contours = probs_volume.contour(isocontours, scalars=probs)
    probs_contours.plot(opacity=0.5)


def windowed_GMM(
    points: np.array,
    n_clusters: int = 100,
    n_comps_per_clusters: int = 10,
    n_points_for_kmeans: int = 1000,
    sampling_resolution: float = 0.5,
    isosurface_thresh: float = 0.001,
    vis: bool = True,
):
    """
    Fit GMM to different subregions 

    points:
        Observed points, (n, 3)
    n_windows:
        How many fractions to split it into
    n_comps_per_window:
        How many GMM components per window
    sampling_resolution: 
        Grid to sample for iscontours
    """
    kmeans = KMeans(n_clusters=n_clusters)
    # Fit on a subset of points to speed up computation
    sampled_inds = np.random.choice(points.shape[0], (n_points_for_kmeans,))
    kmeans.fit(points[sampled_inds])
    # Predict the labels for all of them
    cluster_labels = kmeans.predict(points)

    if vis and False:
        colors = np.array(distinctipy.get_colors(n_clusters))
        color_per_point = colors[cluster_labels]
        pc = pv.PolyData(points)
        pc["color"] = color_per_point
        pc.plot(scalars="color", rgb=True)

    # Try to paraellize this for best practice
    accumulator = None

    min_max_extents = (np.min(points, axis=0), np.max(points, axis=0))

    weights = []
    means = []
    covariances = []
    for i in tqdm(range(n_clusters)):
        sampled_points = points[cluster_labels == i]
        gmm = GaussianMixture(n_components=n_comps_per_clusters)
        gmm.fit(sampled_points)
        weights.append(gmm.weights_)
        means.append(gmm.means_)
        covariances.append(gmm.covariances_)

    weights = np.concatenate(weights, axis=0)
    means = np.concatenate(means, axis=0)
    means = np.concatenate(means, axis=0)
    breakpoint()
    prob_volume, log_likelihood = score_grid(min_max_extents, sampling_resolution, gmm)
    likelihood = np.exp(log_likelihood)

    if accumulator is None:
        accumulator = likelihood
    else:
        accumulator += likelihood
    np.save("data/accumulator.npy", accumulator)

    if vis:
        breakpoint()
        plotter = pv.Plotter()

        probs_contours = prob_volume.contour((isosurface_thresh,), scalars=accumulator)

        plotter.add_mesh(probs_contours)
        plotter.add_mesh(pv.PolyData(points))
        plotter.show()
    return accumulator


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
    windowed_GMM(xyz)

    show_GMM_isocontours(
        xyz, n_components=800, downsample_to_fraction=0.01, show_cloud=True
    )


if __name__ == "__main__":
    args = parse_args()
    main(args.pointcloud, args.n_comps)
