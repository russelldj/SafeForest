import argparse
from pathlib import Path
from telnetlib import GA

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


def score_grid(
    min_max_extents,
    sampling_resolution: float,
    gmm: GaussianMixture,
    max_pred_elements=100000000,
    just_volume: bool = False,
):
    """
    Compute the log likelihood for samples on a grid

    min_max_extents:
        (np.array, np.array) The boundaries of the axis aligned cubiod to compute the sampling over
    sampling_resolution:
        The size of each grid
    ggm:
        The mixture model to sample from
    max_pred_elements:
        Chunk the prediction matrix if it is bigger than this size
    just_volume:
        Just return the volume that would be predicted over
    """
    min_extents, max_extents = min_max_extents

    dims = ((max_extents - min_extents) / sampling_resolution).astype(int)
    probs_volume = pv.UniformGrid(
        dims=dims,
        origin=min_extents,
        spacing=(sampling_resolution, sampling_resolution, sampling_resolution),
    )
    if just_volume:
        return probs_volume
    grids = np.stack((probs_volume.x, probs_volume.y, probs_volume.z), axis=1)

    # The number of samples we are trying to run inference on
    num_samples = grids.shape[0]
    # The number of GMM componets
    num_components = gmm.means_.shape[0]
    # The naive prediction would be n_samples x n_components. We need to figure out how many
    # samples can fit in each chunk without making this matrix too big.
    samples_per_chunk = int(max_pred_elements / num_components)

    preds = []
    # Do predictions on subsets of the data
    for i in tqdm(range(0, num_samples, samples_per_chunk)):
        preds.append(gmm.score_samples(grids[i : i + samples_per_chunk]))

    log_likelihood = np.concatenate(preds, axis=0)
    return probs_volume, log_likelihood


def windowed_GMM(
    points: np.array,
    n_clusters: int = 100,
    n_comps_per_clusters: int = 10,
    n_points_for_kmeans: int = 1000,
    sampling_resolution: float = 0.5,
    isosurface_log_likelihood_thresholds: tuple = (-11,),
    vis: bool = True,
    reload: bool = True,
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
    reload:
        Reload exisiting data
    """
    kmeans = KMeans(n_clusters=n_clusters)
    # Fit on a subset of points to speed up computation
    sampled_inds = np.random.choice(points.shape[0], (n_points_for_kmeans,))
    kmeans.fit(points[sampled_inds])
    # Predict the labels for all of them
    cluster_labels = kmeans.predict(points)

    if vis:
        colors = np.array(distinctipy.get_colors(n_clusters))
        color_per_point = colors[cluster_labels]
        pc = pv.PolyData(points)
        pc["color"] = color_per_point
        pc.plot(scalars="color", rgb=True)

    if reload:
        log_likelihood = np.load("data/log_likelihood.npy")
        min_max_extents = (np.min(points, axis=0), np.max(points, axis=0))
        prob_volume = score_grid(
            min_max_extents, sampling_resolution, None, just_volume=True
        )
    else:
        weights = []
        means = []
        precisions = []
        precisions_cholesky = []
        # Try to paraellize this for best practice
        for i in tqdm(range(n_clusters)):
            sampled_points = points[cluster_labels == i]
            gmm = GaussianMixture(n_components=n_comps_per_clusters)
            gmm.fit(sampled_points)
            weights.append(gmm.weights_)
            means.append(gmm.means_)
            precisions.append(gmm.precisions_)
            precisions_cholesky.append(gmm.precisions_cholesky_)

        weights = np.concatenate(weights, axis=0)
        # Normalize the weight to be a valid probability distribution
        weights = weights / np.sum(weights)
        means = np.concatenate(means, axis=0)
        precisions = np.concatenate(precisions, axis=0)
        precisions_cholesky = np.concatenate(precisions_cholesky, axis=0)

        # Create a concatenation of all the previously fit ones
        inference_GMM = GaussianMixture(
            n_components=n_comps_per_clusters * n_clusters,
            weights_init=weights,
            means_init=means,
            precisions_init=precisions,
        )
        inference_GMM.weights_ = weights
        inference_GMM.means_ = means
        inference_GMM.precisions_ = precisions
        inference_GMM.precisions_cholesky_ = precisions_cholesky

        min_max_extents = (np.min(points, axis=0), np.max(points, axis=0))
        prob_volume, log_likelihood = score_grid(
            min_max_extents, sampling_resolution, inference_GMM
        )

    if vis:
        plotter = pv.Plotter()
        prob_volume["log_likelihood"] = log_likelihood
        probs_contours = prob_volume.contour(
            isosurface_log_likelihood_thresholds, scalars=log_likelihood
        )

        z = probs_contours.points[:, -1]
        plotter.add_mesh(probs_contours, scalars=z)
        plotter.show()
    return log_likelihood


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointcloud", type=Path, default=DEFAULT_POINTCLOUD)
    parser.add_argument("--n-clusters", type=int, default=100)
    parser.add_argument("--n-comps-per-clusters", type=int, default=10)
    parser.add_argument("--n-points-for-kmeans", type=int, default=1000)
    parser.add_argument("--sampling-resolution", type=float, default=0.5)
    parser.add_argument(
        "--isosurface-log-likelihood-thresholds", nargs="+", default=(-11,), type=float
    )
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()
    return args


def main(
    file,
    n_clusters: int = 100,
    n_comps_per_clusters: int = 10,
    n_points_for_kmeans: int = 1000,
    sampling_resolution: float = 0.5,
    isosurface_log_likelihood_thresholds: tuple = (-11,),
    vis: bool = True,
):
    """"""
    data = pd.read_csv(file, names=("x", "y", "z", "count", "unixtime"))
    xyz = data.iloc[:, :3]
    xyz = xyz.to_numpy()
    windowed_GMM(
        xyz,
        n_clusters,
        n_comps_per_clusters,
        n_points_for_kmeans,
        sampling_resolution,
        isosurface_log_likelihood_thresholds,
        vis,
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.pointcloud,
        args.n_clusters,
        args.n_comps_per_clusters,
        args.n_points_for_kmeans,
        args.sampling_resolution,
        args.isosurface_log_likelihood_thresholds,
        args.vis,
    )
