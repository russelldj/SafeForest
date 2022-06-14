'''
Tool to take log file output from segformer and plot what's there
'''

import argparse
import json
from matplotlib import pyplot
import numpy
from pathlib import Path


# Log file looks something like this
# ...
# {"mode": "train", "epoch": 350, "iter": 78700, "lr": 3e-05, "memory": 8407, "data_time": 0.00324, "decode.loss_ce": 0.08722, "decode.acc_seg": 94.0777, "loss": 0.08722, "time": 0.35831}
# {"mode": "train", "epoch": 350, "iter": 78750, "lr": 3e-05, "memory": 8407, "data_time": 0.00313, "decode.loss_ce": 0.08328, "decode.acc_seg": 93.96077, "loss": 0.08328, "time": 0.35801}
# ...

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "log_path",
        type=Path,
        help="Path to log file (json) containing training output",
    )
    args = parser.parse_args()
    assert args.log_path.is_file()
    return args


def extract(log_path, k1, k2):
    '''Get paired arrays, only taking data when both pieces are present.'''
    array1 = []
    array2 = []
    with log_path.open("r") as logfile:
        for line in logfile.readlines():
            linedata = json.loads(line)
            if k1 in linedata and k2 in linedata:
                array1.append(linedata[k1])
                array2.append(linedata[k2])
    return numpy.array(array1), numpy.array(array2)


def main(log_path, classes, save_dir):

    files = []
    for xkey, ykey in (("epoch", "aAcc"),
                       ("epoch", "mIoU"),
                       ("epoch", "mAcc"),
                       ("iter", "decode.loss_ce"),
                       ("iter", "decode.acc_seg"),
                       ("iter", "lr"),
                       ("iter", "loss"),
                       ("iter", "memory")):
        figure, axis = pyplot.subplots()
        x, y = extract(log_path, xkey, ykey)
        axis.plot(x, y, "o-")
        axis.set_xlabel(xkey)
        axis.set_ylabel(ykey)
        if save_dir is not None:
            file = save_dir.joinpath(f"{xkey}_{ykey}.png")
            files.append(file)
            pyplot.savefig(file)

    # Note that these accuracy stats are calculated on the validation data, not
    # the train data, so the "iter" number does not track the main effort. The
    # epochs do however.
    xkey = "epoch"
    for prefix in ("IoU", "Acc"):
        figure, axis = pyplot.subplots()
        axis.set_xlabel(xkey)
        axis.set_ylabel(prefix)
        for suffix in ("background", "vine", "post", "leaves", "trunk", "sign"):
            ykey = f"{prefix}.{suffix}"
            x, y = extract(log_path, xkey, ykey)
            axis.plot(x, y, "o-", label=ykey)
        axis.legend()
        if save_dir is not None:
            file = save_dir.joinpath(f"classes_{xkey}.png")
            files.append(file)
            pyplot.savefig(file)

    if save_dir is None:
        pyplot.show()

    return files


if __name__ == "__main__":
    args = parse_args()
    main(log_path=args.log_path,
         classes=("background", "vine", "post", "leaves", "trunk", "sign"),
         save_dir=None)
