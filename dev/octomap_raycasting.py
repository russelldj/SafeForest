#!/usr/bin/env python

from pathlib import Path

import glooey

# import imgviz
import matplotlib.pyplot as plt
import numpy as np
import octomap
import pandas as pd
import pyglet
import pyvista as pv
import tqdm
import imgviz
import trimesh
import trimesh.transformations as tf
import trimesh.viewer
from timerit import Timerit
from scipy.spatial.transform import Rotation as R


def pointcloud_from_depth(depth, fx, fy, cx, cy):
    assert depth.dtype.kind == "f", "depth must be float and have meter values"

    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = ~np.isnan(depth)
    z = np.where(valid, depth, np.nan)
    x = np.where(valid, z * (c - cx) / fx, np.nan)
    y = np.where(valid, z * (r - cy) / fy, np.nan)
    pc = np.dstack((x, y, z))

    return pc


def labeled_scene_widget(scene, label):
    vbox = glooey.VBox()
    vbox.add(glooey.Label(text=label, color=(255, 255, 255)), size=0)
    vbox.add(trimesh.viewer.SceneWidget(scene))
    return vbox


def create_img(K: np.array, imsize: tuple):
    """
    K:
      The camera projection matrix. Assumed to have units of pixels. 
    imsize:
      The width, height of the image in pixels.
    """
    width, height = imsize

    i_focal = K[1, 1]
    i_principal_point = K[1, 2]

    j_focal = K[0, 0]
    j_principal_point = K[0, 2]

    i_indices = np.arange(height)
    j_indices = np.arange(width)

    i_indices = (i_indices - i_principal_point) / i_focal
    j_indices = (j_indices - j_principal_point) / j_focal
    i_indices, j_indices = np.meshgrid(i_indices, j_indices, indexing="ij")
    i_indices, j_indices = [x.flatten() for x in (i_indices, j_indices)]
    homogenous_points = np.stack(
        (j_indices, i_indices, np.ones_like(j_indices)), axis=1
    )
    return homogenous_points


def visualize(occupied, empty, K, width, height, rgb, pcd, mask, resolution, aabb):
    window = pyglet.window.Window(width=int(640 * 0.9 * 3), height=int(480 * 0.9))

    @window.event
    def on_key_press(symbol, modifiers):
        if modifiers == 0:
            if symbol == pyglet.window.key.Q:
                window.on_close()

    gui = glooey.Gui(window)
    hbox = glooey.HBox()
    hbox.set_padding(5)

    camera = trimesh.scene.Camera(resolution=(width, height), focal=(K[0, 0], K[1, 1]))
    camera_marker = trimesh.creation.camera_marker(camera, marker_height=0.1)

    # initial camera pose
    camera_transform = np.array(
        [
            [0.73256052, -0.28776419, 0.6168848, 0.66972396],
            [-0.26470017, -0.95534823, -0.13131483, -0.12390466],
            [0.62712751, -0.06709345, -0.77602162, -0.28781298],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )

    aabb_min, aabb_max = aabb
    bbox = trimesh.path.creation.box_outline(
        aabb_max - aabb_min, tf.translation_matrix((aabb_min + aabb_max) / 2),
    )

    # geom = trimesh.PointCloud(vertices=pcd[mask], colors=rgb[mask])
    # scene = trimesh.Scene(camera=camera, geometry=[bbox, geom, camera_marker])
    # scene.camera_transform = camera_transform
    # hbox.add(labeled_scene_widget(scene, label="pointcloud"))

    geom = trimesh.voxel.ops.multibox(
        occupied, pitch=resolution, colors=[1.0, 0, 0, 0.5]
    )
    scene = trimesh.Scene(camera=camera, geometry=[bbox, geom, camera_marker])
    scene.camera_transform = camera_transform
    hbox.add(labeled_scene_widget(scene, label="occupied"))

    geom = trimesh.voxel.ops.multibox(
        empty, pitch=resolution, colors=[0.5, 0.5, 0.5, 0.5]
    )
    scene = trimesh.Scene(camera=camera, geometry=[bbox, geom, camera_marker])
    scene.camera_transform = camera_transform
    hbox.add(labeled_scene_widget(scene, label="empty"))

    gui.add(hbox)
    pyglet.app.run()


def create_octree(resolution=1):
    octree = octomap.OcTree(resolution)

    FILE = Path(Path.home(), "Downloads/fullCloud_labeled.txt")
    data = pd.read_csv(FILE, names=("x", "y", "z", "labels"))
    xyz = data.iloc[:, :3]
    labels = data.iloc[:, 3]
    xyz = xyz.to_numpy()
    labels = labels.to_numpy()

    CHUNK_SIZE = 1000000
    for i in tqdm.tqdm(range(0, len(xyz), CHUNK_SIZE)):
        octree.insertPointCloud(
            pointcloud=xyz[i : i + CHUNK_SIZE],
            origin=np.array([0, 0, 0], dtype=float),
            maxrange=-1,
            lazy_eval=True,  # Hack to avoid computing freespace
        )
    return octree


def create_camera_points(rotation_matrix, imsize=(1384, 1032), vis=False):
    K = np.array([[719.4674, 0, 682.85536], [0, 719.4674, 555.98205], [0, 0, 1]])
    # fx: 719.4674
    # fy: 719.4674
    # cx: 682.85536
    # cy: 555.98205

    points = create_img(K, imsize=imsize)
    points = np.transpose(np.dot(rotation_matrix, points.transpose()))

    if vis:
        cloud = pv.PolyData(points)
        plotter = pv.Plotter()
        sphere = pv.Sphere(radius=0.1)
        plotter.add_mesh(sphere)
        plotter.add_mesh(cloud)
        plotter.show()

    return points


def read_trajectory(index=0):
    labels = ("X", "Y", "Z", "q_x", "q_y", "q_z", "q_w", "timeStamp")
    FILE = Path(
        Path.home(),
        "data/SafeForestData/datasets/portugal_UAV_12_21/derived/safe_forest_2/slam_outputs/Odom_camera_left/2022-01-14-17-44/odom.txt",
    )
    data = pd.read_csv(FILE, names=labels)
    loc = data.iloc[index, :3].to_numpy()
    quat = data.iloc[index, 3:7].to_numpy()
    rot = R.from_quat(quat)

    rot_matrix = rot.as_matrix()
    return loc, rot_matrix


def main():
    data = imgviz.data.arc2017()
    camera_info = data["camera_info"]
    K = np.array(camera_info["K"]).reshape(3, 3)
    rgb = data["rgb"]
    pcd = pointcloud_from_depth(
        data["depth"], fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )

    nonnan = ~np.isnan(pcd).any(axis=2)
    mask = np.less(pcd[:, :, 2], 2)

    resolution = 1
    octree = create_octree(resolution)

    origin, rotation_matrix = read_trajectory(index=1000)

    IMSIZE = (1384, 1032)
    directions = create_camera_points(rotation_matrix, IMSIZE)

    num_points = len(directions)

    END = np.array([0, 0, 0], dtype=np.float64)
    for _ in Timerit(num=3, verbose=2):
        dists = np.zeros(num_points)
        for i in range(num_points):
            end = END.copy()
            ret = octree.castRay(origin, directions[i], end, True)
            if np.any(end != END):
                dist = np.linalg.norm(end - origin)
                dists[i] = dist
        dists = np.reshape(dists, (IMSIZE[1], IMSIZE[0]))
        plt.imshow(dists)
        plt.show()

    occupied, empty = octree.extractPointCloud()

    aabb_min = octree.getMetricMin()
    aabb_max = octree.getMetricMax()

    visualize(
        occupied=occupied,
        empty=empty,
        K=K,
        width=camera_info["width"],
        height=camera_info["height"],
        rgb=rgb,
        pcd=pcd,
        mask=mask,
        resolution=resolution,
        aabb=(aabb_min, aabb_max),
    )


if __name__ == "__main__":
    main()
