import argparse
from pathlib import Path
from safeforest.dataset_generation.file_utils import get_files
from safeforest.dataset_generation.file_utils import ensure_dir_normal_bits
import numpy as np

# Takes a base dataset config
# A base inherited network
# A final derived network
# A folder of folders to train on

# Read in a set of file
SEMFIRE_DATASET = (
    "/home/frc-ag-1/dev/mmsegmentation/configs/_base_/datasets/semfire_sete.py"
)
BASE_NETWORK = "/home/frc-ag-1/dev/mmsegmentation/configs/segformer/segformer_mit-b0_512x512_160k_semfire_sete.py"
DERIVED_NETWORK = "/home/frc-ag-1/dev/mmsegmentation/configs/segformer/segformer_mit-b5_512x512_160k_semfire_sete.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-folder",
        type=Path,
        help="A folder of individual datasets to train on",
        required=True,
    )
    parser.add_argument(
        "--base-dataset",
        type=Path,
        help="The dataset to replace the path for",
        default=SEMFIRE_DATASET,
    )
    parser.add_argument(
        "--base-network",
        type=Path,
        help="The base network that is later derived. e.g. a b0 segformer variant",
        default=BASE_NETWORK,
    )
    parser.add_argument(
        "--derived-network",
        type=Path,
        help="The final network config, assumed to inherit from another network",
        default=DERIVED_NETWORK,
    )
    parser.add_argument(
        "--output-dataset-folder", type=Path, help="Where to write the new configs",
    )
    parser.add_argument(
        "--output-model-folder", type=Path, help="Where to write the new configs",
    )
    args = parser.parse_args()
    return args


def main(
    *,
    dataset_folder,
    base_dataset,
    base_network,
    derived_network,
    output_dataset_folder,
    output_model_folder,
    base_string="b0",
    replace_string="b5",
):
    # Write back to the same location
    if output_dataset_folder is None:
        output_dataset_folder = base_dataset.parent

    if output_model_folder is None:
        output_model_folder = base_network.parent

    # Create the output folder
    ensure_dir_normal_bits(output_dataset_folder)
    ensure_dir_normal_bits(output_model_folder)

    datasets = get_files(dataset_folder, "*", require_dir=True)
    for dataset in datasets:
        # Write out the dataset config
        with open(base_dataset) as file:
            lines = file.readlines()
            is_dataset_line = [x[:9] == "data_root" for x in lines]
            if np.sum(is_dataset_line) != 1:
                raise ValueError("Ambigious")
            dataset_line = np.where(is_dataset_line)[0][0]
            lines[dataset_line] = f'data_root = "{dataset}"\n'

        output_dataset_file = Path(output_dataset_folder, dataset.parts[-1] + ".py")
        print(output_dataset_file)
        with open(output_dataset_file, "w") as output_fh:
            output_fh.writelines(lines)

        # Write out the base config
        with open(base_network) as file:
            lines = file.readlines()

        lines[2] = f'    "{output_dataset_file}",\n'

        output_base_network_file = Path(
            output_model_folder, f"{base_network.stem}_{dataset.parts[-1]}.py"
        )
        with open(output_base_network_file, "w") as outfile_h:
            outfile_h.writelines(lines)

        with open(derived_network) as file:
            lines = file.readlines()
        lines[0] = f'_base_ = ["{output_base_network_file}"]\n'

        output_derived_network_file = str(output_base_network_file).replace(
            base_string, replace_string
        )
        with open(output_derived_network_file, "w") as outfile_h:
            outfile_h.writelines(lines)


if __name__ == "__main__":
    args = parse_args()
    main(
        dataset_folder=args.data_folder,
        base_dataset=args.base_dataset,
        base_network=args.base_network,
        derived_network=args.derived_network,
        output_dataset_folder=args.output_dataset_folder,
        output_model_folder=args.output_model_folder,
    )
