from pathlib import Path
from turtle import down

import numpy as np
import pandas as pd
import pyvista as pv
from pyvista import examples
from sklearn.mixture import GaussianMixture


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
    isocontour_quantiles=(0.97, 0.99),
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

    min_extents = np.min(xyz, axis=0)
    max_extents = np.max(xyz, axis=0)

    dims = ((max_extents - min_extents) / sampling_resolution).astype(int)
    probs_volume = pv.UniformGrid(
        dims=dims,
        origin=min_extents,
        spacing=(sampling_resolution, sampling_resolution, sampling_resolution),
    )
    grids = np.stack((probs_volume.x, probs_volume.y, probs_volume.z), axis=1)

    probs = gmm.score_samples(grids)
    # probs = to_unit(probs)

    probs_volume.point_data["probabilities"] = probs
    probs_volume.set_active_scalars("probabilities")

    if show_volume:
        # TODO unitize
        probs_volume.plot(volume=True)
    isocontours = np.quantile(probs, isocontour_quantiles)
    probs_contours = probs_volume.contour(isocontours, scalars=probs)
    probs_contours.plot(opacity=0.5)


FILE = Path(Path.home(), "Downloads/map_156.txt")
OUTPUT_FILE = Path(Path.home(), "data/SafeForestData/temp/mesh.xyz")

data = pd.read_csv(FILE, names=("x", "y", "z", "count", "unixtime"))
xyz = data.iloc[:, :3]
xyz = xyz.to_numpy()

show_GMM_isocontours(
    xyz, n_components=800, downsample_to_fraction=0.01, show_cloud=True
)
