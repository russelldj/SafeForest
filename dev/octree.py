import pandas as pd
from pathlib import Path
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyrr
from timerit import Timerit


def intersect(octree, ray):
    # Hack so it's passed by reference
    best_node = [None, None]
    best_dist = np.inf

    # This is horribly redundant because
    def f_traverse(
        node, node_info, best_dist=best_dist, ray=ray, best_node=best_node,
    ):
        aabb = pyrr.aabb.create_from_points(
            np.asarray([node_info.origin, node_info.origin + node_info.size])
        )
        res = pyrr.geometric_tests.ray_intersect_aabb(ray, aabb)

        intersected = res is not None

        if intersected:
            dist = np.linalg.norm(res - ray[0])

            # Check the logic here. My thought was if it intersects farther away than the current best leaf,
            # it can't be right. Check whether we need an epsilon here to avoid floating point error.
            # I don't think so because leaf nodes are the only ones that produce a distance.
            if dist > best_dist:
                return True

            # The second part of this check should be redundant, down to issues with epsilon
            if isinstance(node, o3d.geometry.OctreeLeafNode) and dist < best_dist:
                best_dist = dist
                best_node[0] = node
                best_node[1] = node_info

        # The return is whether to NOT traverse the child nodes
        return not intersected

    octree.traverse(f_traverse)

    # This is set by reference within the traversal
    return best_node


# FILE = Path(Path.home(), "Downloads/map_156.txt")
FILE = Path(
    Path.home(),
    "data/SafeForestData/datasets/portugal_UAV_12_21/derived/safe_forest_2/slam_outputs/PointCloud/2022-01-14-17-44/map_64.txt",
)
OUTPUT_FILE = Path(Path.home(), "data/SafeForestData/temp/mesh.xyz")

data = pd.read_csv(FILE, names=("x", "y", "z", "count", "unixtime"))
xyz = data.iloc[:, :3]
xyz = xyz.to_numpy()

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(xyz)
# o3d.io.write_point_cloud(str(OUTPUT_FILE), cloud)

zs = xyz[:, 2]
print(zs.shape)
cmap = mpl.colormaps["viridis"]
norm = mpl.colors.Normalize()
colors = cmap(norm(zs))[:, :3]

# fit to unit cube
cloud.scale(
    1 / np.max(cloud.get_max_bound() - cloud.get_min_bound()), center=cloud.get_center()
)
cloud.colors = o3d.utility.Vector3dVector(colors)
# o3d.visualization.draw_geometries([cloud])

print()

sample_point = np.asarray(cloud.points)[100]

print("octree division")
octree = o3d.geometry.Octree(max_depth=10)
octree.convert_from_point_cloud(cloud, size_expand=0.01)
ray = np.array([[0, 0, 0], sample_point])

t1 = Timerit(1000)

for _ in t1:
    intersected_node = intersect(octree=octree, ray=ray)

print("t1.total_time = %r" % (t1.total_time,))
breakpoint()

o3d.visualization.draw_geometries([octree])

print(data)
