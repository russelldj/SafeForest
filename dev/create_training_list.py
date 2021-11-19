import argparse
from pathlib import Path
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-dir", type=Path)
    parser.add_argument("--seg-dir", type=Path)
    parser.add_argument("--output-train-file", type=Path)
    parser.add_argument("--output-val-file", type=Path)
    parser.add_argument("--train-frac", type=float, default=0.8)
    args = parser.parse_args()
    return args

def main(rgb_dir, seg_dir, train_file, val_file, train_frac):
    rgb_files = np.asarray(sorted(rgb_dir.glob("*png"))) 
    seg_files = np.asarray(sorted(seg_dir.glob("*png"))) 
    
    random_vals = np.random.uniform(size=(rgb_files.shape[0],))
    train_ids = random_vals < train_frac
    val_ids = np.logical_not(train_ids)

    with open(train_file, "w") as outfile_h:
        for r, s in zip(rgb_files[train_ids], seg_files[train_ids]):
            outfile_h.write(f"{r}, {s}\n")

    with open(val_file, "w") as outfile_h:
        for r, s in zip(rgb_files[val_ids], seg_files[val_ids]):
            outfile_h.write(f"{r}, {s}\n")

    print(len(rgb_files), len(seg_files))

if __name__ == "__main__":
    args = parse_args()
    main(args.rgb_dir, args.seg_dir, args.output_train_file, args.output_val_file, args.train_frac)
