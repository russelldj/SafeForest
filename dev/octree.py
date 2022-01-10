import pandas as pd
from pathlib import Path
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyrr


def intersect(octree, ray):
    best_indices = None
    best_dist = np.inf

    def f_traverse(node, node_info, best_dist=best_dist, ray=ray):
        early_stop = False
        aabb = pyrr.aabb.create_from_points(
            np.asarray([node_info.origin, node_info.origin + node_info.size])
        )
        res = pyrr.geometric_tests.ray_intersect_aabb(ray, aabb)

        intersected = res is not None

        if intersected and isinstance(node, o3d.geometry.OctreeLeafNode):
            dist = np.linalg.norm(res - ray[0])
            if dist < best_dist:
                best_dist = dist
                best_indices = node.indices

        return not intersected

        if isinstance(node, o3d.geometry.OctreeInternalNode):
            if isinstance(node, o3d.geometry.OctreeInternalPointNode):
                n = 0
                for child in node.children:
                    if child is not None:
                        n += 1
                print(
                    "{}{}: Internal node at depth {} has {} children and {} points ({})".format(
                        "    " * node_info.depth,
                        node_info.child_index,
                        node_info.depth,
                        n,
                        len(node.indices),
                        node_info.origin,
                    )
                )

                # we only want to process nodes / spatial regions with enough points
                early_stop = len(node.indices) < 250
            if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
                print(
                    "{}{}: Leaf node at depth {} has {} points with origin {}".format(
                        "    " * node_info.depth,
                        node_info.child_index,
                        node_info.depth,
                        len(node.indices),
                        node_info.origin,
                    )
                )
        else:
            raise NotImplementedError("Node type not recognized!")

        # early stopping: if True, traversal of children of the current node will be skipped
        return early_stop

    octree.traverse(f_traverse)
    return ID


FILE = Path(Path.home(), "Downloads/map_156.txt")
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

print("octree division")
octree = o3d.geometry.Octree(max_depth=4)
octree.convert_from_point_cloud(cloud, size_expand=0.01)
ray = np.array([[0, 0, 0], [23.9061 + 0.5, -24.0909 + 0.5, 0.0428392 + 0.5]])
intersect(octree=octree, ray=ray)

scene = o3d.t.geometry.RaycastingScene()
breakpoint()

o3d.visualization.draw_geometries([octree])

print(data)
