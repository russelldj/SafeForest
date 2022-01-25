import argparse
from pathlib import Path
import ubelt as ub
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from scipy import spatial
from config import RGB_EXT, SEG_EXT, IMG_DIR, ANN_DIR

SAVED_CLASSES = "dev/classes.npy"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--val-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    return args


def write_folder(
    input_files: list, output_dir: Path, output_extension: str, label: bool = False
) -> None:
    ub.ensuredir(Path(output_dir), mode=0o0755)
    for f in input_files:
        f_folder = str(f.parts[-2])
        f_suffix = f.suffix
        output_f = Path(output_dir, f_folder + output_extension + f_suffix)

        if label:
            label_image = convert_image(f)
            cb = plt.imshow(label_image)
            plt.colorbar(cb)
            plt.show()
            imwrite(output_f, label_image)
        else:
            ub.cmd(f"ln -s {f} {output_f}")


def choose_canonical_colors(files):
    try:
        classes = np.load(SAVED_CLASSES)
    except FileNotFoundError:
        for f in files:
            im = imread(f)[..., :3]
            im = im.reshape((-1, 3))
            unique = np.unique(im, axis=0)
            print(unique.shape)
            if unique.shape[0] == 8:
                classes = unique
                np.save(SAVED_CLASSES, classes)
                break


def convert_image(color_filename: Path):
    """ """
    classes = np.load(SAVED_CLASSES)
    im = imread(color_filename)[..., :3]
    im_shape = im.shape
    im = im.reshape((-1, 3))
    dist = spatial.distance.cdist(im, classes)
    indices = np.argmin(dist, axis=1)
    label_image = indices.reshape(im_shape[:2]).astype(np.uint8)
    return label_image


def main(train_dir, val_dir, output_dir, train_output_ext=None, val_output_ext=None):
    train_files = sorted(train_dir.glob("**/*"))
    val_files = sorted(val_dir.glob("**/*"))

    train_rgb = [x for x in train_files if "rgb" in x.name]
    train_labels = [x for x in train_files if "label" in x.name]
    val_rgb = [x for x in val_files if "rgb" in x.name]
    val_labels = [x for x in val_files if "label" in x.name]

    # choose_canonical_colors(train_labels + val_labels)
    for f, img_f in zip(train_labels, train_rgb):
        img = imread(img_f)
        _, axs = plt.subplots(1, 2)

        label_image = convert_image(f)
        cb = axs[0].imshow(label_image, vmin=0, vmax=7, cmap="inferno")
        axs[1].imshow(img)
        plt.colorbar(cb)
        plt.show()

    write_folder(train_rgb, Path(output_dir, IMG_DIR, "train"), RGB_EXT)
    write_folder(val_rgb, Path(output_dir, IMG_DIR, "val"), RGB_EXT)

    write_folder(train_labels, Path(output_dir, ANN_DIR, "train"), SEG_EXT, label=True)
    write_folder(val_labels, Path(output_dir, ANN_DIR, "val"), SEG_EXT, label=True)


if __name__ == "__main__":
    args = parse_args()
    main(args.train_dir, args.val_dir, args.output_dir)
