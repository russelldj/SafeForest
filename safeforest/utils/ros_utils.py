import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd
from imageio import imread
from pathlib import Path
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs
import rospy


def create_point_cloud_message(points, timestamp, frame):
    cloud_ros = PointCloud2()
    cloud_ros.fields.append(
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1)
    )
    cloud_ros.fields.append(
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1)
    )
    cloud_ros.fields.append(
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)
    )

    cloud_ros.width = points.shape[0]
    header = std_msgs.msg.Header()
    header.stamp.secs = timestamp
    header.frame_id = frame
    cloud_ros.header = header
    cloud_ros.data = points.ravel().tobytes()
    return cloud_ros
    ## This represents an un-ordered (non-image point cloud)
    # cloud_ros.height = 1
    ## TODO this number will need to be updated since we no longer know how many points we'll have
    ## TODO make sure all of this specification can stay
    ## TODO figure out how RGB is being encoded if it's just a float
    # cloud_ros.fields.append(
    #    PointField(name="rgb", offset=16, datatype=PointField.FLOAT32, count=1)
    # )
    # if point_type is PointType.SEMANTICS_MAX:
    #    cloud_ros.fields.append(
    #        PointField(
    #            name="semantic_color",
    #            offset=20,
    #            datatype=PointField.FLOAT32,
    #            count=1,
    #        )
    #    )
    #    self.cloud_ros.fields.append(
    #        PointField(
    #            name="confidence", offset=24, datatype=PointField.FLOAT32, count=1
    #        )
    #    )


def read_trajectory(file):
    labels = ("X", "Y", "Z", "q_x", "q_y", "q_z", "q_w", "timeStamp")
    data = pd.read_csv(file, names=labels)
    locs = data.iloc[:, :3].to_numpy()
    quats = data.iloc[:, 3:7].to_numpy()
    timestamps = data.iloc[:, 7].to_numpy()
    rots = [R.from_quat(q).as_matrix() for q in quats]
    rots = np.stack(rots, axis=2)

    return locs, rots, quats, timestamps


def read_cloud(file):
    return np.loadtxt(file, delimiter=",")[:, :3]


def read_image(
    folder, timestamp,
):
    image_names = sorted(Path(folder).glob("*png"))
    stamps = [int(x.stem) for x in image_names]
    stamps = np.array(stamps)
    dists = np.abs(stamps - timestamp)
    best_match = np.argmin(dists)
    imname = str(image_names[best_match])
    img = imread(imname)
    return img
