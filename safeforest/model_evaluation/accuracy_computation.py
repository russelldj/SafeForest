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


def confusion_union(matrix, index):
    # Row + column - double count
    return np.sum(matrix[index, :]) + np.sum(matrix[:, index]) - matrix[index, index]


def compute_mIoU(confusion_matrix: np.array):
    """

    confusion_matrix: (n, n) true label in i, pred in j
    """
    num_classes = confusion_matrix.shape[0]
    IoUs = []
    for i in range(num_classes):
        intersection = confusion_matrix[i, i]
        union = confusion_union(confusion_matrix, i)
        IoU = intersection / union
        IoUs.append(IoU)

    mIoU = np.mean(IoUs)
    return mIoU, IoUs
