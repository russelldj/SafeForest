import numpy as np
from sklearn import metrics


def accumulate_confusion_matrix(preds, gt, current_confusion, n_classes=7):
    """Update a confusion matrix with predicted classes and gt class
    TODO futher document
    """
    additional_confusion = metrics.confusion_matrix(
        gt, preds, labels=np.arange(n_classes)
    )
    assert np.all(current_confusion.shape == additional_confusion.shape)
    updated_confusion = current_confusion + additional_confusion
    return updated_confusion
