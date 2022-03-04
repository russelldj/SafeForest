#!/usr/bin/env python
# PointCloud2 color cube
# https://answers.ros.org/question/289576/understanding-the-bytes-in-a-pcl2-message/
import pdb
import struct
from glob import glob
from os.path import expanduser

import cv2
import numpy as np
import pandas as pd
import rospy
import tf
from cv_bridge import CvBridge
from rospy import Time

# from numpy_ros import to_message, to_numpy
# from safeforest.utils.ros_utils import read_cloud, read_image, read_trajectory
from scipy.spatial.transform import Rotation
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header


def read_trajectory(file):
    labels = ("X", "Y", "Z", "q_x", "q_y", "q_z", "q_w", "timeStamp")
    data = pd.read_csv(file, names=labels)
    locs = data.iloc[:, :3].to_numpy()
    quats = data.iloc[:, 3:7].to_numpy()
    timestamps = data.iloc[:, 7].to_numpy()

    rots = [Rotation.from_quat(q).as_dcm() for q in quats]
    rots = np.stack(rots, axis=2)

    return locs, rots, quats, timestamps


def read_cloud(file):
    return np.loadtxt(file, delimiter=",")[:, :3]


def read_image(
    folder, timestamp,
):
    search_string = folder + "/*png"
    image_names = sorted(glob(search_string))
    stamps = [x.split("/")[-1][:-4] for x in image_names]
    stamps = [int(x) for x in stamps]
    stamps = np.array(stamps)
    dists = np.abs(stamps - timestamp)
    best_match = np.argmin(dists)
    imname = str(image_names[best_match])
    img = np.flip(cv2.imread(imname), axis=2)
    return img


POINTCLOUD_FILE = expanduser("~/Downloads/fullCloud_labeled.txt")

rospy.init_node("create_cloud_xyzrgb")
pointcloud_pub = rospy.Publisher("/velodyne_points", PointCloud2, queue_size=2)
image_pub = rospy.Publisher("/mapping/left/image_color", Image, queue_size=2)
# TODO probably assign the frame
br = tf.TransformBroadcaster()

CAMERA_TRAJECTORY_FILE = expanduser(
    "~/Downloads/DataPortugal_Processed/Odom_camera_left/2022-01-14-17-44/odom.txt",
)
LIDAR_TRAJECTORY_FILE = expanduser(
    "~/Downloads/DataPortugal_Processed/Odom_lidar/2022-01-14-17-44/odom.txt",
)
IMAGE_FOLDER = expanduser("~/data/SafeForestData_temp/all_portugal_2_images")
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
pose_data_timestamps = np.hstack((pose_data, np.expand_dims(pose_timestamps, axis=1)))


rgb = struct.unpack("I", struct.pack("BBBB", 0, 255, 0, 255))[0]
points_with_color_list = [p.tolist() + [rgb] for p in points]
pc2 = point_cloud2.create_cloud(header, FIELDS, points_with_color_list)

bridge = CvBridge()

# timestamp_message = Timestamp.now()

for pdt in pose_data_timestamps[0:-1:10]:
    # RT = np.eye(4)
    xyz = pdt[:3]
    quat = pdt[3:7]
    R = Rotation.from_quat(quat).as_dcm()
    # RT[:3, :3] = R

    # Transform points by R and t
    rotated_points = np.dot(R, points.T).T
    rotated_translated_points = rotated_points + xyz
    pdb.set_trace()
    points_with_color_list = [p.tolist() + [rgb] for p in rotated_translated_points]
    pc2 = point_cloud2.create_cloud(header, FIELDS, points_with_color_list)

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
    # br.sendTransform(xyz, quat, Time.from_seconds(timestamp), "map", "world")
    timestamp_message = rospy.Time.now()
    # timestamp_message.secs = timestamp / 1000000000

    br.sendTransform(xyz, quat, timestamp_message, "aft_mapped", "world")
    print(image.shape)
    # TODO publish image
    # TODO publish pose
    # TODO update timestamp
    # rospy.sleep(1.0)
