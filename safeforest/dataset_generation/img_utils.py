import cv2
import numpy as np
from pathlib import Path
from scipy import spatial
import tempfile

from ubelt import symlink


def convert_colors_to_indices(img: np.ndarray, palette: np.ndarray):
    """

    img: (h, w, 3|4) input image of color masks
    palette: (n, 3) Ordered colors in the colormap
    """
    img = img[..., :3]
    im_shape = img.shape
    img = img.reshape((-1, 3))
    dist = spatial.distance.cdist(img, palette)
    indices = np.argmin(dist, axis=1)
    label_image = indices.reshape(im_shape[:2]).astype(np.uint8)
    return label_image


def imwrite_ocv(filename, img):
    """Take an RGB or RGBA image and write it using OpenCV"""
    shape = img.shape
    filename = str(filename)
    if len(shape) == 3 and shape[2] == 3:
        # RGB
        img = np.flip(img, axis=2)
    elif len(shape) == 3 and shape[2] == 4:
        # RGBA
        img = np.stack((np.flip(img[..., :3], axis=2), img[..., 3:4]), axis=2)
    elif len(shape) == 2:
        pass
    else:
        raise ValueError()

    cv2.imwrite(filename, img)


def augmented_images(img_file, label_file, all_disparity_pairs, all_img_files,
                     all_label_files, force_copy, augmentations):
    """TODO.

    The supported augmentations are
        "disparity": (None, (str, str)) If not None, the [0] string represents
            an image, and numpy.load the [1] str to get disparity linking
            img_file to this new image. We can construct a new label_file from
            the disparity.
        "shuffle": (None, int) If not None, then we will create N copies of the
            current image, where backgrounds of other images are overlaid over
            its background.

    All these generated images will be stored in a temporary directory that
    lasts until the generator ends.

    Returns: TODO
    """

    with tempfile.TemporaryDirectory() as save_dir:
        save_dir = Path(save_dir)
        save_dir.joinpath("ann").mkdir()
        save_dir.joinpath("img").mkdir()

        for i, (d_img_file,
                d_label_file) in enumerate(disparity(img_file,
                                                     label_file,
                                                     all_disparity_pairs)):
            for j, (s_img_file,
                    s_label_file) in enumerate(shuffle(d_img_file,
                                                       d_label_file,
                                                       augmentations.get("shuffle", None),
                                                       save_dir,
                                                       all_img_files,
                                                       all_label_files)):
                yield (
                    s_img_file,
                    s_label_file,
                    # If we're dealing with the original image then pass the
                    # original force_copy argument through. If we're dealing
                    # with a generated image, then we must copy things.
                    force_copy if i == j == 0 else True,
                )


def disparity(ifile, lfile, pairs):
    # First, always yield the original (baseline)
    yield ifile, lfile

    # TODO: Actually we should check the augmentation somehow

    # Then, if applicable, yield additional files
    if pairs is not None:
        pass


def shuffle(ifile, lfile, number, save_dir, all_ifiles, all_lfiles):
    # First, always yield the original (baseline)
    yield ifile, lfile

    # Then, if applicable, yield additional files
    if number is not None:
        assert ifile.name == lfile.name

        # Do this N times
        for _ in range(number):

            # Load the image we want to modify
            base_image = cv2.imread(str(ifile), cv2.IMREAD_UNCHANGED)
            base_label = cv2.imread(str(lfile), cv2.IMREAD_UNCHANGED)

            # Choose how many backgrounds to overlay between [1, 5] (chosen
            # by hand, seemed reasonable)
            indices = []
            for _ in range(np.random.randint(1, 6)):

                # Choose another image to take the background from
                index = np.random.randint(0, len(all_ifiles))
                other_image = cv2.imread(str(all_ifiles[index]), cv2.IMREAD_UNCHANGED)
                other_label = cv2.imread(str(all_lfiles[index]), cv2.IMREAD_UNCHANGED)

                # Choose the places where both images have backgrounds
                mask = np.logical_and(base_label == 0, other_label == 0)
                base_image[mask] = other_image[mask]

                # Bookkeeping
                indices.append(index)

            # Construct a modified name
            def name_with_idx(file, indices):
                name = file.name.split(".")
                return f"{name[0]}_idxs-{'-'.join(map(str, indices))}.{name[1]}"
            new_ifile = save_dir.joinpath("img", name_with_idx(ifile, indices))
            new_lfile = save_dir.joinpath("ann", name_with_idx(lfile, indices))

            print(new_ifile.name)
            # Save the modified image and the unmodified label file, the yield
            cv2.imwrite(str(new_ifile), base_image)
            symlink(lfile, new_lfile)
            yield new_ifile, new_lfile
