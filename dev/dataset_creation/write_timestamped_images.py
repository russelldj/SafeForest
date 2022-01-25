#! /usr/env/python
# Copyright (c) 2015, Rethink Robotics, Inc.

# Using this CvBridge Tutorial for converting
# ROS images to OpenCV2 images
# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

# Using this OpenCV2 tutorial for saving Images:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html

# rospy for the subscriber
import rospy

# ROS Image message
from sensor_msgs.msg import Image

# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError

import ros_numpy

# OpenCV2 for saving an image
import cv2

OUTPUT_DIR = "/home/frc-ag-1/data/SafeForestData/temp/all_portugal_2_images/"


def image_callback(msg):
    print("Received an image!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = ros_numpy.numpify(msg)
    except CvBridgeError as e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg
        time = msg.header.stamp
        cv2.imwrite(OUTPUT_DIR + str(time) + ".png", cv2_img)
        rospy.sleep(1)


def main():
    rospy.init_node("image_listener")
    # Define your image topic
    image_topic = "/mapping/left/image_color"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    # Spin until ctrl + c
    rospy.spin()


if __name__ == "__main__":
    main()
