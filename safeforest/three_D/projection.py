import numpy as np


def filter_points_to_image_extent(points, img_size, offset=0.5, return_indices=True):
    """Return points which are within an image extents
    
    The returned points will be such that rounding them will allow valid indexing into the image.
    Therefore, the minimum valid points is -0.5 and the maximum is axis_size - 0.5.

    inputs:
        points: np.array
            (2, n). The projected points in image coordinates
        img_size: tuple(int):
            (width, height) in pixels
        offset: float
            How to account for subpixel indexing. 0.5 return indices that are valid after rounding. 
            0.0 will return indices valid after truncation.
        return_indices: bool
            Whether to return the indices that were used
        
    returns:
        np.array:
            Valid points after filtering. (2, n).
    """
    img_size_x, img_size_y = img_size
    inside = np.logical_and.reduce(
        (
            points[0] > (-offset),
            points[1] > (-offset),
            points[0] < (img_size_x - offset),
            points[1] < (img_size_y - offset),
        )
    )
    points_inside = points[:, inside]

    if return_indices:
        return points_inside, inside

    return points_inside


def sample_points(img, img_points):
    """Sample the values from an image based on coordinates

    inputs:
        img: np.array
            (h, w, 3) or (h, w). The image to sample from
        img_points: np.array
            (2, n). float or int. Points to sample from. Assumed to be (x, y).T

    returns:
        np.array
        Sampled points concatenated vertically
    """
    # Force non-int points to be ints
    if issubclass(img_points.dtype.type, np.floating):
        img_points = np.round(img_points).astype(np.uint16)

    sampled_values = img[img_points[1], img_points[0]]
    return sampled_values


def generate_cloud_data_common_lidar(
    bgr_img, lidar, intrinsics, image_size, extrinsics=None
):
    """Project lidar points into the image and texture ones which are within the frame of the image 

    Create a 

    bgr_img:
        (numpy array bgr8). (h, w, 3)
    lidar:
        np.array float32 (n, 3). 3D lidar points in the local frame
    extrinsics:
        np.array (4, 4). The relation between the lidar frame and the camera frame
    [x, y, Z] = [X, Y, Z] * intrinsic.T
    """
    if extrinsics is not None:
        lidar_homog = np.concatenate((lidar, np.ones((lidar.shape[0], 1))), axis=1)
        # Perform the matrix multiplication to get the lidar points into the local frame
        lidar_proj = np.dot(extrinsics, lidar_homog.T)
        lidar_transformed = lidar_proj[:3]
    else:
        # Keep the lidar frame the same as it was initially
        lidar_transformed = lidar.T

    # Note that lidar_transformed is (3, n) to facilitate easier multiplication
    in_front_of_camera = lidar_transformed[2] > 0
    # Take only points which are in front of the camera
    lidar_transformed_filtered = lidar_transformed[:, in_front_of_camera]
    # Project each point into the image
    projections_homog = np.dot(intrinsics, lidar_transformed_filtered)
    projections_inhomog = projections_homog[:2] / projections_homog[2]

    # TODO consider using the shape from the current image
    image_points, within_bounds = filter_points_to_image_extent(
        projections_inhomog, image_size
    )
    projections_inhomog = np.asarray(projections_homog)
    # plt.scatter(projections_inhomog[0], projections_inhomog[1])
    # plt.show()
    # breakpoint()

    # TODO remember that we actually need to keep track of which points we used
    sampled_colors = sample_points(bgr_img, image_points)

    # TODO figure out if these need to be set
    # self.xyd_vect[:, 0:2] = None  # self.xy_index * depth_img.reshape(-1, 1)
    # self.xyd_vect[:, 2:3] = None  # depth_img.reshape(-1, 1)

    # This is in the local robot frame, not the camera frame
    in_front_XYZ = lidar.T[:, in_front_of_camera]
    # Transpose to get it to be (n, 3)
    XYZ_vect = in_front_XYZ[:, within_bounds].T
    return XYZ_vect, sampled_colors, image_points
