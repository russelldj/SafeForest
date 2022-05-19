#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import String

import numpy as np
import struct
import ctypes
import pyvista as pv
import copy
import open3d as o3d


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries(
        [source_temp, target_temp],
        zoom=0.4459,
        front=[0.9288, -0.2951, -0.2242],
        lookat=[1.6784, 2.0612, 1.4451],
        up=[-0.3402, -0.9189, -0.1996],
    )


class mapICP:
    def __init__(self):
        rospy.init_node("cloud_fixer", anonymous=True)
        rospy.Subscriber("/lio_sam/mapping/map_global", PointCloud2, self.callback)
        self.last_useful_cloud = None
        self.total_transform = np.eye(4)

    def callback(self, data):
        # Taken from http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29
        gen = pc2.read_points(data, skip_nans=True)
        xyz = np.array([[0, 0, 0]])
        rgb = np.array([[0, 0, 0]])
        # self.lock.acquire()
        gen = pc2.read_points(data, skip_nans=True)
        int_data = list(gen)

        for x in int_data:
            test = x[3]
            # cast float32 to int so that bitwise operations are possible
            s = struct.pack(">f", test)
            i = struct.unpack(">l", s)[0]
            # you can get back the float value by the inverse operations
            pack = ctypes.c_uint32(i).value
            r = (pack & 0x00FF0000) >> 16
            g = (pack & 0x0000FF00) >> 8
            b = pack & 0x000000FF
            # prints r,g,b values in the 0-255 range
            # x,y,z can be retrieved from the x[0],x[1],x[2]
            xyz = np.append(xyz, [[x[0], x[1], x[2]]], axis=0)
            rgb = np.append(rgb, [[r, g, b]], axis=0)

        self.register(xyz)
        # cloud = pv.PolyData(xyz)
        # cloud.plot()
        # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

    def register(self, new_cloud, threshold=10.0, trans_init=np.eye(4)):
        new_cloud_o3d = o3d.geometry.PointCloud()
        new_cloud_o3d.points = o3d.utility.Vector3dVector(new_cloud)

        if self.last_useful_cloud is None:
            self.last_useful_cloud = new_cloud_o3d
            return

        reg_p2p = o3d.pipelines.registration.registration_icp(
            self.last_useful_cloud,
            new_cloud_o3d,
            threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        draw_registration_result(
            self.last_useful_cloud, new_cloud_o3d, reg_p2p.transformation
        )


if __name__ == "__main__":
    mapICP()
    rospy.spin()
