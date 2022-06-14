from imageio import imread
from pathlib import Path
import shutil
import tempfile
import time

from sacred import Experiment
from sacred.observers import MongoObserver
import ubelt

from safeforest.model_evaluation import plot_mmseg_log_stats

from evaluate_model import calc_metrics, sample_for_confusion


EXPERIMENT = Experiment("train_docker_model")
# TODO: Enable after running with real data for a few times
# EXPERIMENT.observers.append(MongoObserver(url="localhost:27017", db_name="mmseg"))
EXPERIMENT.observers.append(MongoObserver(url="localhost:27017", db_name="mmseg-test"))


# Name for the config files in docker
DCFG = "dataset_config.py"
MCFG = "model_config.py"


# TODO: Figure out where test.py lives


@EXPERIMENT.config
def config():
    # TODO: Explain
    dockerfile_path = Path("/home/eric/Desktop/SEMSEGTEST/Dockerfile")
    # TODO: Explain
    workdir = f"WORKDIR_{int(time.time() * 1e6)}"
    # TODO: Explain
    DATA = "/mmsegmentation/data/"
    docker_additions = ["\n",
                        f'ENV DATASET_CFG="{DATA}{DCFG}"\n',
                        f'ENV MODEL_CFG_NAME="{MCFG}"\n',
                        f'ENV MODEL_CFG="{DATA}{MCFG}"\n',
                        'ENV MODEL_NAME="segformer"\n',
                        f'ENV WORKDIR="{workdir}"\n',
                        "CMD cp ${DATASET_CFG} configs/_base_/datasets/ &&" + \
                        " cp ${MODEL_CFG} configs/${MODEL_NAME}/ &&" + \
                        " python tools/train.py /mmsegmentation/configs/${MODEL_NAME}/${MODEL_CFG_NAME} --work-dir " + DATA + "${WORKDIR}/ &&" + \
                        " python " + DATA + "test.py /mmsegmentation/configs/${MODEL_NAME}/${MODEL_CFG_NAME} " + DATA + "FAKE_MMREADY_TESTDATA/ " + DATA + "${WORKDIR}/"
                        ]
    # TODO: Explain
    dataset_cfg_path = Path("/home/eric/Desktop/SEMSEGTEST/fake_vine_dataset_config.py")
    # TODO: Explain
    model_cfg_path = Path("/home/eric/Desktop/SEMSEGTEST/fake_vine_model_segformer_config.py")
    # TODO: Explain
    classes = ("background", "vine", "post", "leaves", "trunk", "sign")
    # TODO: Explain
    shared_volume = Path("/tmp/")
    # TODO: Explain
    train_dir = Path("/tmp/FAKE_MMREADY_DATA/")
    # TODO: Explain
    test_dir = Path("/tmp/FAKE_MMREADY_TESTDATA/")


def build_docker(original, additions, _run):
    with tempfile.TemporaryDirectory() as docker_dir:
        newfile = Path(docker_dir).joinpath("Dockerfile")
        with newfile.open("w") as file:
            for line in original.open("r").readlines():
                file.write(line)
            for line in additions:
                file.write(line)
        ubelt.cmd(f"docker build -t mmsegmentation {docker_dir}", verbose=1)
        _run.add_artifact(newfile)


def record_data(train_dir, test_dir, _run):
    for name, directory in (("train_data", train_dir),
                            ("test_data", test_dir)):
        with tempfile.TemporaryDirectory() as save_dir:
            newfile = Path(save_dir).joinpath(name)
            with newfile.open("w") as file:
                for png in sorted(directory.glob("*/*/*png")):
                    file.write(f"{str(png)}\n")
            _run.add_artifact(newfile)


def prepare_config(dcfg, mcfg, shared, _run):
    shutil.copy(dcfg, shared.joinpath(DCFG))
    shutil.copy(mcfg, shared.joinpath(MCFG))
    _run.add_artifact(dcfg)
    _run.add_artifact(mcfg)


def run_docker(shared_volume):
    ubelt.cmd(f"docker run --name mmseg --rm --gpus all --shm-size=8g -v {shared_volume}:/mmsegmentation/data mmsegmentation",
              verbose=1)


def capture_train_output(workdir, _run):
    for log in workdir.glob("*log*"):
        _run.add_artifact(log)
    latest = workdir.joinpath("latest.pth")
    assert latest.is_symlink()
    model = latest.resolve()
    _run.add_artifact(model)
    return model


def plot_log_data(workdir, classes, _run):
    log_path = [_ for _ in workdir.glob("*log.json")][0]
    with tempfile.TemporaryDirectory() as save_dir_str:
        save_dir = Path(save_dir_str)
        plot_mmseg_log_stats.main(log_path=log_path,
                                  classes=classes,
                                  save_dir=save_dir)
        for png in save_dir.glob("*png"):
            _run.add_artifact(png)


def evaluate_model_on_test(test_dir, workdir, classes, _run):

    def img_generator(base, directory, do_read=False):
        files = sorted(base.joinpath(directory).glob("*png"))
        if do_read:
            return map(imread, files)
        else:
            return files

    with tempfile.TemporaryDirectory() as save_dir_str:
        confusion = sample_for_confusion(
            img_files=img_generator(test_dir, "img_dir"),
            pred_files=img_generator(workdir, "predicted", do_read=True),
            label_files=img_generator(test_dir, "ann_dir"),
            sample_freq=20,
            num_classes=len(classes),
            remap=None,
            log_preds=True,
            palette=None,
            sacred=True,
            verbose=False,
            _run=_run,
            save_file=save_dir_str+"/qualitative_{:06d}.png",
        )
        calc_metrics(
            confusion=confusion,
            classes=classes,
            save_dir=Path(save_dir_str),
            sacred=True,
            _run=_run,
            verbose=False,
        )


@EXPERIMENT.automain
def main(
    dockerfile_path,
    workdir,
    docker_additions,
    dataset_cfg_path,
    model_cfg_path,
    classes,
    shared_volume,
    train_dir,
    test_dir,
    _run,
):

    # Modify/build Docker file, save as artifact
    build_docker(dockerfile_path, docker_additions, _run)

    # Save data files as artifacts
    record_data(train_dir, test_dir, _run)

    # Copy config files and save as artifacts
    prepare_config(dataset_cfg_path, model_cfg_path, shared_volume, _run)

    # Run docker to train the network (will take a long time)
    run_docker(shared_volume)

    # Capture the latest model and the output logs as artifacts
    workdir = shared_volume.joinpath(workdir)
    trained_model_path = capture_train_output(workdir, _run)

    # Plot the log outputs and store as artifacts
    plot_log_data(workdir, classes, _run)

    # Evaluate model to get stats, store as artifacts and metrics
    evaluate_model_on_test(test_dir, workdir, classes, _run)
