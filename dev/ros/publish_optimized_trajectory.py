# /usr/env/bin
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import rospy
import std_msgs
from numpy_ros import to_message, to_numpy
from safeforest.utils.ros_utils import (
    create_point_cloud_message,
    read_cloud,
    read_image,
    read_trajectory,
)
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String
from tqdm import tqdm

POINTCLOUD_FILE = Path(Path.home(), "Downloads/fullCloud_labeled.txt")
CAMERA_TRAJECTORY_FILE = Path(
    Path.home(),
    "data/SafeForestData/datasets/portugal_UAV_12_21/derived/safe_forest_2/slam_outputs/Odom_camera_left/2022-01-14-17-44/odom.txt",
)
LIDAR_TRAJECTORY_FILE = Path(
    Path.home(),
    "data/SafeForestData/datasets/portugal_UAV_12_21/derived/safe_forest_2/slam_outputs/Odom_lidar/2022-01-14-17-44/odom.txt",
)
IMAGE_FOLDER = Path(Path.home(), "data/SafeForestData/temp/all_portugal_2_images")


FIELDS = [
    PointField("x", 0, PointField.FLOAT32, 1),
    PointField("y", 4, PointField.FLOAT32, 1),
    PointField("z", 8, PointField.FLOAT32, 1),
    # PointField('rgb', 12, PointField.UINT32, 1),
    PointField("rgba", 12, PointField.UINT32, 1),
]


def talker():

    rgb = struct.unpack("I", struct.pack("BBBB", 0, 255, 0, 255))[0]
    points = read_cloud(POINTCLOUD_FILE)
    points_with_color_list = [p.tolist() + [rgb] for p in points]
    labels = ("X", "Y", "Z", "q_x", "q_y", "q_z", "q_w", "timeStamp")
    pose_data = pd.read_csv(CAMERA_TRAJECTORY_FILE, names=labels).to_numpy()[:, :-1]
    # Hack because only this one contains valid timestamps
    _, _, _, pose_timestamps = read_trajectory(LIDAR_TRAJECTORY_FILE)
    pose_data_timestamps = np.hstack(
        (pose_data, np.expand_dims(pose_timestamps, axis=1))
    )
    # map_pub = rospy.Publisher("/pointcloud", PointCloud2, queue_size=10)
    rospy.init_node("optimized_map")
    pub = rospy.Publisher("chatter", String, queue_size=10)
    point_pub = rospy.Publisher("points", PointCloud2, queue_size=2)
    rate = rospy.Rate(10)  # 10hz
    i = 0
    for pose in pose_data_timestamps:
        if rospy.is_shutdown():
            break
        print(pose)
        timestamp = pose[-1]

        # pointcloud_message = create_point_cloud_message(points, timestamp, "world")

        header = std_msgs.msg.Header()
        # header.stamp.secs = timestamp
        header.frame_id = "map"
        pc2 = point_cloud2.create_cloud(header, FIELDS, points_with_color_list)
        hello_str = "hello world %s" % rospy.get_time()
        # rospy.loginfo(hello_str)
        pub.publish(hello_str)
        point_pub.publish(pc2)
        rate.sleep()

    # pv.PolyData(points).plot(eye_dome_lighting=True)
    # Unfortunately, the map needs to be in the local coordinate frame
    # for pose_data_timestamp in tqdm(pose_data_timestamps):
    #    if rospy.is_shutdown():
    #        break
    #    timestamp = pose_data_timestamp[-1]
    #    image = read_image(IMAGE_FOLDER, timestamp)
    #    map_message = create_point_cloud_message(points, timestamp, "world")
    #    map_pub.publish(map_message)
    # plt.imshow(image)
    # plt.show()
    # breakpoint()
    # map_pub = rospy.Publisher("chatter", String, queue_size=10)
    # rate = rospy.Rate(10)  # 10hz
    # while not rospy.is_shutdown():
    #    hello_str = "hello world %s" % rospy.get_time()
    #    rospy.loginfo(hello_str)
    #    pub.publish(hello_str)
    #    rate.sleep()


if __name__ == "__main__":
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
