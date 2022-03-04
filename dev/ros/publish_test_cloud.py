#!/usr/bin/env python
# PointCloud2 color cube
# https://answers.ros.org/question/289576/understanding-the-bytes-in-a-pcl2-message/
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import rospy
from numpy_ros import to_message, to_numpy
from safeforest.utils.ros_utils import read_cloud, read_image, read_trajectory
from scipy.spatial.transform import Rotation
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField, Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
from tqdm import tqdm
from safeforest.vis.pyvista_utils import add_origin_cube

# import tf
import pyvista as pv


def plot_cloud_path(cloud, poses, all_timestamps):
    pl = pv.Plotter()
    lidar = pv.PolyData(cloud)
    pl.add_mesh(lidar, color="white")
    breakpoint()
    for timestamps, pose in zip(all_timestamps, poses):
        trajectory = pv.PolyData(pose[:, :3])
        pl.add_mesh(trajectory, scalars=timestamps, point_size=15)

    pl.enable_eye_dome_lighting()
    pl.show()


POINTCLOUD_FILE = Path(Path.home(), "Downloads/fullCloud_labeled.txt")

rospy.init_node("create_cloud_xyzrgb")
pointcloud_pub = rospy.Publisher("point_cloud2", PointCloud2, queue_size=2)
image_pub = rospy.Publisher("image", Image, queue_size=2)
# TODO probably assign the frame
# br = tf.TransformBroadcaster()

CAMERA_TRAJECTORY_FILE = Path(
    Path.home(),
    # "data/DataPortugal_Processed/Odom_camera_left/2022-01-14-17-44/odom.txt",
    "/home/frc-ag-1/data/DataPortugal_Processed/Odom_gps/2022-01-14-17-44/odom.txt",
)
LIDAR_TRAJECTORY_FILE = Path(
    Path.home(), "data/DataPortugal_Processed/Odom_lidar/2022-01-14-17-44/odom.txt",
)
IMAGE_FOLDER = Path(Path.home(), "data/SafeForestData_temp/all_portugal_2_images")
LABELS = ("X", "Y", "Z", "q_x", "q_y", "q_z", "q_w", "timeStamp")
FIELDS = [
    PointField("x", 0, PointField.FLOAT32, 1),
    PointField("y", 4, PointField.FLOAT32, 1),
    PointField("z", 8, PointField.FLOAT32, 1),
    # PointField('rgb', 12, PointField.UINT32, 1),
    PointField("rgba", 12, PointField.UINT32, 1),
]

header = Header()
header.frame_id = "map"

points = read_cloud(POINTCLOUD_FILE)

pose_data = pd.read_csv(CAMERA_TRAJECTORY_FILE, names=LABELS).to_numpy()[:, :-1]
# Hack because only this one contains valid timestamps
_, _, _, pose_timestamps = read_trajectory(LIDAR_TRAJECTORY_FILE)
# pose_data_timestamps = np.hstack((pose_data, np.expand_dims(pose_timestamps, axis=1)))
breakpoint()

rgb = struct.unpack("I", struct.pack("BBBB", 0, 255, 0, 255))[0]
points_with_color_list = [p.tolist() + [rgb] for p in points]
pc2 = point_cloud2.create_cloud(header, FIELDS, points_with_color_list)

bridge = CvBridge()

plotter = pv.Plotter()
plotter.store_image = True
plotter.enable_eye_dome_lighting()
plotter.open_gif("vis/points.gif")
# plotter.show(auto_close=False)
add_origin_cube(plotter)

plot_cloud_path(points, [pose_data], [pose_timestamps])

first = True

for pdt in tqdm(pose_data_timestamps[1000:]):
    RT = np.eye(4)
    xyz = pdt[:3]
    quat = pdt[3:7]
    R = Rotation.from_quat(quat).as_matrix()
    RT[:3, :3] = R

    # Transform points by R inv and t inv
    rotated_points = np.dot(R.T, points.T).T
    rotated_translated_points = rotated_points - xyz
    points_with_color_list = [p.tolist() + [rgb] for p in rotated_translated_points]
    pc2 = point_cloud2.create_cloud(header, FIELDS, points_with_color_list)

    if first:
        point_cloud = pv.PolyData(rotated_translated_points)
        plotter.add_mesh(point_cloud)
        first = False
    else:
        pc = pv.PolyData(rotated_translated_points)
        point_cloud.overwrite(pc)

    plotter.write_frame()

    if rospy.is_shutdown():
        break
    timestamp = pdt[-1]
    # pc2.header.stamp.nsecs = timestamp
    pointcloud_pub.publish(pc2)
    image = read_image(IMAGE_FOLDER, timestamp)
    image = np.flip(image, axis=2)
    # image_message = to_message(Image, image)
    image_message = bridge.cv2_to_imgmsg(image, encoding="passthrough")
    image_pub.publish(image_message)
    # br.sendTransform(xyz, quat)
    print(image.shape)
    # TODO publish image
    # TODO publish pose
    # TODO update timestamp
    # rospy.sleep(1.0)
