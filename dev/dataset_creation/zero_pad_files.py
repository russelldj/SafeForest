import argparse
from os import rename
from safeforest.dataset_generation.file_utils import pad_filename, get_files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir")
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--end-index", type=int)
    args = parser.parse_args()
    return args


def main(input_dir, start_index=None, end_index=None):
    files = get_files(input_dir, "*")
    padded_files = [
        pad_filename(x, start_index=start_index, end_index=end_index) for x in files
    ]
    for f, p_f in zip(files, padded_files):
        rename(f, p_f)


if __name__ == "__main__":
    args = parse_args()
    main(args.input_dir, args.start_index, args.end_index)
