'''
Tool to try and learn repeatably using learning methods made by hand, which
will need to follow a certain pattern.
'''

from imageio import imread
from pathlib import Path
import shutil
import tempfile
import time

from sacred import Experiment
from sacred.observers import MongoObserver
import ubelt

from safeforest.models import color_svm
from safeforest.model_evaluation import plot_mmseg_log_stats

from evaluate_model import calc_metrics, sample_for_confusion


EXPERIMENT = Experiment("train_homemade_model")
# TODO: Enable after running with real data for a few times
# EXPERIMENT.observers.append(MongoObserver(url="localhost:27017", db_name="mmseg"))
EXPERIMENT.observers.append(MongoObserver(url="localhost:27017", db_name="mmseg-test"))


# Name for the config files in docker
DCFG = "dataset_config.py"
MCFG = "model_config.py"


# TODO: Figure out where test.py lives


# TODO: Make these configs more general
@EXPERIMENT.config
def config():
    # Give the path to the "working directory" where we will save our model and
    # the inferenced test images
    workdir = Path(f"/tmp/WORKDIR_{int(time.time() * 1e6)}")
    # TODO: Explain
    classes = ("background", "vine", "post", "leaves", "trunk", "sign")
    # TODO: Explain
    chosen_model = color_svm.main
    test_model = color_svm.inference
    # TODO: Explain
    model_kwargs = {"mode": "single", "number": 20}
    # TODO: Explain
    train_dir = Path("/tmp/FAKE_MMREADY_DATA/")
    # TODO: Explain
    test_dir = Path("/tmp/FAKE_MMREADY_TESTDATA/")


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


def train_model(model, train_dir, workdir, kwargs):
    '''Run models with a consistent pattern/API.'''
    return model(datapath=train_dir,
                 savedir=workdir,
                 **kwargs)


def capture_train_output(workdir, _run):
    '''Saves all .pth files, so let's keep that consistent.'''
    for path in workdir.glob("*pth"):
        _run.add_artifact(path)


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
            sample_freq=5,
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
    workdir,
    classes,
    chosen_model,
    test_model,
    model_kwargs,
    train_dir,
    test_dir,
    _run,
):

    # Save data files as artifacts
    record_data(train_dir, test_dir, _run)

    # Train the model
    assert not workdir.is_dir()
    workdir.mkdir()
    classifier, saved = train_model(chosen_model, train_dir, workdir, model_kwargs)

    # Capture the latest model as artifacts
    capture_train_output(workdir, _run)

    # Run the model on the test set and save images
    save_dir = workdir.joinpath("predicted")
    save_dir.mkdir()
    test_model(classifier=classifier,
               imdir=test_dir.joinpath("img_dir"),
               save_dir=save_dir)

    # Evaluate model to get stats, store as artifacts and metrics
    evaluate_on_test(test_dir, workdir, classes, _run)
