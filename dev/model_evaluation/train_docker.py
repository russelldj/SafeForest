'''
Tool to try and learn repeatably using mmseg docker, capturing inputs and
outputs in sacred.

NOTE: You can examine the config like so:
    python train_docker.py print_config
NOTE: You can modify the config like so (works with previous command as well):
    python train_docker.py with model_name=bisenetv2
'''

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
# TODO: Enable after running with final data for a few times
# EXPERIMENT.observers.append(MongoObserver(url="localhost:27017", db_name="mmseg"))
EXPERIMENT.observers.append(MongoObserver(url="localhost:27017", db_name="mmseg-test"))


# Name for the config files in docker and where they are stored
DCFG = "dataset_config.py"
MCFG = "model_config.py"
DATA = "/mmsegmentation/data/"

# Define the config files and their relationship to the model names
FILES = {
    "bisenetv2": "vine_model_bisenet_config.py",
    "fcn": "vine_model_fcn_config.py",
    "segformer": "vine_model_segformer_config.py",
    "unet": "vine_model_unet_config.py",
}


# TODO: Make these configs more general
@EXPERIMENT.config
def config():

    # Set a default model, can be overridden on the command line
    model_name = "unet"
    assert model_name in FILES, f"Given model {model_name} not recognized {FILES.keys()}"

    # Path to the root/original mmseg docker container
    # TODO: Reference the official version - maybe make it a fixed path?
    dockerfile_path = Path("/home/eric/Desktop/SEMSEGTEST/Dockerfile")
    # Give the path to the "working directory" where we'll save our logs/model
    # and the inferenced test images. This was already an mmseg concept.
    workdir = f"WORKDIR_{int(time.time() * 1e6)}"
    # Make a list of lines to add to the docker file to put our files in the
    # right place (based on an assumed volume) and then call training.
    docker_cmd = f"CMD cp {DATA}{DCFG} configs/_base_/datasets/ &&" + \
                 f" cp {DATA}{MCFG} configs/{model_name}/ &&"
    docker_train_additions = [
        "\n",
        docker_cmd + f" python tools/train.py /mmsegmentation/configs/{model_name}/{MCFG} --work-dir {DATA}{workdir}/"
    ]
    docker_test_additions = [
        "\n",
        docker_cmd + f" python {DATA}infer_on_test.py /mmsegmentation/configs/{model_name}/{MCFG} {DATA}REAL_MMREADY_TESTDATA/ {DATA}{workdir}/"
    ]
    # Give the dataset and model configs that we want to use in training.
    # TODO: Save versions in git and reference it here better
    dataset_cfg_path = Path("/home/eric/Desktop/SEMSEGTEST/vine_dataset_config.py")
    model_cfg_path = Path(f"/home/eric/Desktop/SEMSEGTEST/{FILES[model_name]}")
    # Additional files that needs to be copied into the shared volume
    additional_files = [
        Path("/home/eric/Desktop/SEMSEGTEST/SafeForest/safeforest/model_evaluation/infer_on_test.py"),
    ]
    # Pretty simple, give the classes (in the right order) that were labeled
    classes = ("background", "vine", "post", "leaves", "trunk", "sign")
    # State the volume that will get -v linked with the docker container. Files
    # moved here can go into the container.
    shared_volume = Path("/home/eric/Desktop/SEMSEGTEST/")
    # Directory where images/labels are stored in the cityscapes format.
    train_dir = shared_volume.joinpath("REAL_MMREADY_DATA/")
    # Directory where test files are stored DIRECTLY in img_dir/*png and
    # ann_dir/*png. Similar to cityscapes but not quite the same (no val split).
    test_dir = shared_volume.joinpath("REAL_MMREADY_TESTDATA/")


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
    '''
    Record the names of train/test files, saving the actual files would be too
    expensive but this gives a little traceability.
    '''
    for name, directory in (("train_data", train_dir),
                            ("test_data", test_dir)):
        with tempfile.TemporaryDirectory() as save_dir:
            newfile = Path(save_dir).joinpath(name)
            with newfile.open("w") as file:
                for png in sorted(directory.glob("*/*/*png")):
                    file.write(f"{str(png)}\n")
            _run.add_artifact(newfile)


def prepare_config(dcfg, mcfg, additional, shared, _run):
    shutil.copy(dcfg, shared.joinpath(DCFG))
    shutil.copy(mcfg, shared.joinpath(MCFG))
    _run.add_artifact(dcfg)
    _run.add_artifact(mcfg)
    for file in additional:
        shutil.copy(file, shared.joinpath(file.name))
        _run.add_artifact(file)


def run_docker(shared_volume, _run):
    # Capture how long this takes as a metric
    start = time.time()
    ubelt.cmd(f"docker run --name mmseg --rm --gpus all --shm-size=8g -v {shared_volume}:/mmsegmentation/data mmsegmentation",
              verbose=1)
    end = time.time()
    _run.log_scalar("TrainingTimeHrs", (end-start) / 3600)


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


def evaluate_on_test(test_dir, workdir, classes, _run):

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
            sample_freq=1,
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
    docker_train_additions,
    docker_test_additions,
    dataset_cfg_path,
    model_cfg_path,
    additional_files,
    classes,
    shared_volume,
    train_dir,
    test_dir,
    _run,
):

    # Modify/build Docker file for training, save as artifact
    build_docker(dockerfile_path, docker_train_additions, _run)

    # Save data files as artifacts
    record_data(train_dir, test_dir, _run)

    # Copy config files and save as artifacts
    prepare_config(dataset_cfg_path, model_cfg_path, additional_files, shared_volume, _run)

    # Run docker to train the network (will take a long time)
    run_docker(shared_volume, _run)

    # Capture the latest model and the output logs as artifacts
    workdir = shared_volume.joinpath(workdir)
    trained_model_path = capture_train_output(workdir, _run)

    # Plot the log outputs and store as artifacts
    plot_log_data(workdir, classes, _run)

    # Modify/build Docker file for testing, save as artifact
    build_docker(dockerfile_path, docker_test_additions, _run)

    # Run docker to test the network
    run_docker(shared_volume, _run)

    # Evaluate model to get stats, store as artifacts and metrics
    evaluate_on_test(test_dir, workdir, classes, _run)
