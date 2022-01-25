# Overview
`dev` is a place for random stand alone scripts

# Files
* `cf_matrix.py` Helper scripts for confusion matrix visualization.
* `compute_confusion_matrix.py` Compute the confusion matrix from a prediction and groundtruth segmentation.
* `config.py` Stores constants that are meant to be used globally across the project.
* `get_class_probabilities.py` Run inference on a video and extract the probablities.
* `remap_dataset.py` Change or merge the classes in a dataset.
* `show_class_confidence.py` Show the confidence of predictions for different classes.
* `show_label_map.py` Show the colors and classnames for a labelmap as a reference figure.
* `show_seg_video.py` Overlay the class labels onto the input video.
* `video_utils.py` Utility functions for reading/writing/merging/breaking videos.
* `visualize_semantic_labels.py` Probably outdated, TODO figure out what it is.
* `yamaha_to_cityscapes.py` Convert the CMU-Yamaha dataset to the Cityscapes format.
