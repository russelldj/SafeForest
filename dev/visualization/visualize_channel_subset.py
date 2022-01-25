from imageio import imread, imwrite
from pathlib import Path
import matplotlib.pyplot as plt
import ubelt as ub
import argparse


def show_RG(folder, extension="png"):
    files = sorted(folder.glob("*." + extension))
    for f in files:
        img = imread(f)
        img[..., 2] = 0
        plt.imshow(img)
        plt.show()


def write_RG(input_folder, output_folder, extension="png"):
    files = sorted(input_folder.glob("*." + str(extension)))
    ub.ensuredir(output_folder, mode=0o0755)
    for f in files:
        img = imread(f)
        img[..., 2] = 0
        output_file = Path(output_folder, f.name)
        imwrite(output_file, img)


sete_folder = Path(
    Path.home(),
    "data/SafeForestData/datasets/semfire_segmentation/original/2021_sete_fontes_forest/img",
)
sete_extension = "*.png"

quinta_folder = Path(
    Path.home(),
    "data/SafeForestData/datasets/semfire_segmentation/original/2019_2020_quinta_do_bolao_coimbra/img",
)
quinta_extension = "*.jpg"

portugal_2_folder = Path(
    Path.home(),
    "data/SafeForestData/datasets/portugal_UAV_12_21/derived/safe_forest_2/pred_labels_rui_yamaha/img_dir/train",
)
portugal_2_output = Path(
    Path.home(),
    "data/SafeForestData/datasets/portugal_UAV_12_21/derived/safe_forest_2/RG_images/img_dir/train",
)

portugal_2_folder_val = Path(
    Path.home(),
    "data/SafeForestData/datasets/portugal_UAV_12_21/derived/safe_forest_2/pred_labels_rui_yamaha/img_dir/train",
)
portugal_2_output_val = Path(
    Path.home(),
    "data/SafeForestData/datasets/portugal_UAV_12_21/derived/safe_forest_2/RG_images/img_dir/val",
)

portugal_2_extension = "*.png"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", type=Path)
    parser.add_argument("--output-folder", type=Path)
    parser.add_argument("--extension", type=Path)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    write_RG(args.input_folder, args.output_folder, args.extension)

    # show_RG(sete_folder, sete_extension)
    write_RG(
        quinta_folder,
        "/home/frc-ag-1/data/SafeForestData/datasets/semfire_segmentation/derived/RG_experiments/2019_2020_quinta_do_bolao_coimbra/img/",
        quinta_extension,
    )
    # write_RG(portugal_2_folder, portugal_2_output, portugal_2_extension)
    # write_RG(portugal_2_folder_val, portugal_2_output_val, portugal_2_extension)
    # write_RG(quinta_folder, portugal_2_output_val, quinta_extension)

