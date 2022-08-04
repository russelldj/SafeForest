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


def augmented_images(img_file, label_file, all_img_files, all_label_files,
                     force_copy, augmentations):
    """
    Takes a hand-labled img/label pair and creates augmented data from them.
    See below for more information on the types of augmentations possible.

    Arguments:
        img_file: Path() variable pointing to an existing image
        label_file: Path() variable pointing to corresponding labeled image
        all_img_files: List of Path() variables to all real image files
        all_label_files: List of Path() variables to all real label files
        force_copy: (bool) Whether the original image/label pair should be
            copied (True) or symlinked (False)
        augmentations: (dict) Contains allowable augmentations. See below:

    The supported augmentations are
        "disparity": (None | (str, str)) If not None, the [0] string represents
            an image, and numpy.load-ing the [1] str should get disparity
            linking img_file to this new image. We can construct a new
            label_file from the disparity.
        "shuffle": (None | int) If not None, then we will create N copies of the
            current image, where backgrounds of other images are overlaid over
            its background.

    NOTE: All these generated images will be stored in a temporary directory
    that lasts until the generator ends. They must be copied from there, not
    symlinked. This is so this function doesn't have to care about final
    locations or names.

    Yields: An iterator of ("image path", "label path", "(bool) force copy")
        tuples. The very first return is always the provided (unaugmented)
        image/label pair, and the force_copy argument in that case is the one
        passed in. For all other returns force_copy will be forced to be True
        because those augmented images will all be generated and (as noted
        above) will not be available forever.
    """

    with tempfile.TemporaryDirectory() as save_dir:
        save_dir = Path(save_dir)
        save_dir.joinpath("ann").mkdir()
        save_dir.joinpath("img").mkdir()

        for i, (d_img_file,
                d_label_file) in enumerate(disparity(img_file,
                                                     label_file,
                                                     augmentations["disparity"],
                                                     save_dir)):
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


def disparity(ifile, lfile, pair, save_dir):
    '''Generator that generates labels projected using stereo.'''

    # First, always yield the original (baseline)
    yield ifile, lfile

    # Then, if applicable, yield additional files
    if pair is not None:

        # Hard-code the Left/Right relationship between cameras. The idea of
        # +1/-1 is whether to add or subtract the disparity to get to the other
        # camera
        functions = {"cam0": simple_set,
                     "cam1": complex_set,
                     "cam2": simple_set,
                     "cam3": complex_set}
        replace = {"cam0": "cam1", "cam1": "cam0", "cam2": "cam3", "cam3": "cam2"}

        impath, disppath = pair

        old_labels = cv2.imread(str(lfile), cv2.IMREAD_UNCHANGED)
        new_labels = np.zeros_like(old_labels)
        disparities = np.load(disppath)

        # Camera of the labeled image (cam0/1/2/3)
        incam = ifile.name.split("_")[1]

        # Go through the classes in reverse order so the more important labels
        # (like Vine=1) will be written on top.
        for classid in sorted(np.unique(old_labels), reverse=True):
            # Don't propagate the background, everything left unmarked by the
            # other classes will be background.
            if classid == 0:
                continue

            print(classid)

            # Call the appropriate pixel-setting function for this classid
            functions[incam](old_labels, new_labels, disparities, classid)

        # Save all background-labeled points as 255 (invalid class). This is
        # because too many real points are being labeled background and it's
        # screwing everything over.
        new_labels[new_labels == 0] = 255

        # Save the label file
        save_path = save_dir.joinpath(lfile.name.replace(incam, replace[incam]))
        cv2.imwrite(str(save_path), new_labels)

        yield Path(impath), save_path


def simple_set(old_labels, new_labels, disparities, classid):
    '''
    For these cameras, the disparity is "lined up" appropriately, such that
    when you visualize it the disparity "lies over" the image correctly. As
    such we can just apply the disparity to the "where" of the classes.
    '''
    # Find the points for a certain class
    i, j = np.where(old_labels == classid)
    class_disparity = disparities[i, j]

    # Then iterate to only get the points with valid disparity (-1
    # disparity indicates a match was not made)
    valid = np.where(class_disparity > 0)[0]
    i = i[valid]
    j = j[valid]
    class_disparity = disparities[i, j]

    # Set the polarity appropriately and round to the nearest int
    # (pixel location)
    class_disparity = -1 * (class_disparity + 0.5).astype(int)

    # Calculate the new pixel values for these labels based on the
    # disparities. Disparity only affects horizontal pixels, a.k.a. j
    j_shifted = j + class_disparity

    # Modify the array
    if len(i) > 0:
        new_labels[i, j_shifted] = classid


def complex_set(old_labels, new_labels, disparities, classid):
    '''
    For these cameras, the disparity is not lined up appropriately. The best I
    could think of was to search for a horiztonal vector for each pixel and see
    which disparity "pointed at" that pixel. Very time-consuming compared to
    the simple set.
    '''

    max_search = (disparities.max() + 1).astype(int)

    # Find the points for a certain class
    i, j = np.where(old_labels == classid)

    # This feels terrible, but I don't know how else to figure out which point
    # in the right image matches up to a certain left pixel
    i_found = []
    j_found = []
    j_shifted = []

    indices = np.array(range(max_search))

    for ival, jval in zip(i, j):
        todo = disparities[ival, jval:jval+max_search]
        thing = np.abs(indices[:len(todo)] - todo)
        if thing.min() < 1:
            i_found.append(ival)
            j_found.append(jval)
            j_shifted.append(jval + np.argmin(thing))
    i_found = np.array(i_found)
    j_found = np.array(j_found)
    j_shifted = np.array(j_shifted)

    # Modify the array
    if len(i_found) > 0:
        new_labels[i_found, j_shifted] = old_labels[i_found, j_found]


def shuffle(ifile, lfile, number, save_dir, all_ifiles, all_lfiles):
    '''Generator that generates images with shuffled backgrounds.'''

    # First, always yield the original (baseline)
    yield ifile, lfile

    # Then, if applicable, yield additional files
    if number is not None:
        assert ifile.name == lfile.name or lfile.name.endswith(ifile.name), \
               f"Does {ifile} match {lfile}?"

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

                if base_image is None or other_image is None:
                    print(f"base_image is None: {base_image is None}, other_image is None: {other_image is None}\n\t{ifile}\n\t{lfile}")

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
