from pathlib import Path

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


FILE = Path(Path.home(), "Downloads/map_156.txt")
OUTPUT_FILE = Path(Path.home(), "data/SafeForestData/temp/mesh.xyz")

# Read the data and con
data = pd.read_csv(FILE, names=("x", "y", "z", "count", "unixtime"))
xyz = data.iloc[:, :3]
xyz = xyz.to_numpy()

# Fit a GMM to the existings points
gmm = GaussianMixture(n_components=1)
gmm.fit(xyz)

RESOLUTION = 3

# Create meshgrid to sample from

min_extents = np.min(xyz, axis=0)
max_extents = np.max(xyz, axis=0)

dims = ((max_extents - min_extents) / RESOLUTION).astype(int)
probs_volume = pv.UniformGrid(
    dims=dims, origin=min_extents, spacing=(RESOLUTION, RESOLUTION, RESOLUTION)
)
grids = np.stack((probs_volume.x, probs_volume.y, probs_volume.z), axis=1)
# samples = [np.arange(min_extents[i], max_extents[i], RESOLUTION) for i in range(3)]
# len_samples = [len(x) for x in samples]
# grids = np.meshgrid(*samples)
# grids = [x.flatten(order="C") for x in grids]
# grids = np.stack(grids, axis=1)
breakpoint()

probs = gmm.score_samples(grids)
probs = to_unit(probs)

probs_volume.point_data["probabilities"] = probs
probs_volume.set_active_scalars("probabilities")

probs_volume.plot(volume=True)

probs_contours = probs_volume.contour([0.5], scalars=probs)

# Create a plot
p = pv.Plotter()
p.add_mesh(probs_contours)
p.show()
