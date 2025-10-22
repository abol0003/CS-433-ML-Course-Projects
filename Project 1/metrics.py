import numpy as np


def to_01_labels(y_pm1):
    """
    Convert labels from {-1, 1} format to {0, 1} format.

    Args:
        y_pm1: Array of labels in {-1, 1} format.

    Returns:
        Array of labels in {0, 1} format.
    """
    return (y_pm1 > 0).astype(np.uint8)


def to_pm1_labels(y01):
    """
    Convert labels from {0, 1} format to {-1, 1} format.

    Args:
        y01: Array of labels in {0, 1} format.

    Returns:
        Array of labels in {-1, 1} format.
    """
    return np.where(y01 == 1, 1, -1).astype(np.int32)


def accuracy_score(y_true01, y_pred01):
    """
    Compute classification accuracy.

    Args:
        y_true01: True labels in {0, 1} format.
        y_pred01: Predicted labels in {0, 1} format.

    Returns:
        Accuracy as a float value.
    """
    y_true01 = np.asarray(y_true01)
    y_pred01 = np.asarray(y_pred01)
    return float(np.mean(y_true01 == y_pred01))


def precision_recall_f1(y_true01, y_pred01):
    """
    Compute precision, recall, and F1 score for binary classification.

    Args:
        y_true01: True labels in {0, 1} format.
        y_pred01: Predicted labels in {0, 1} format.

    Returns:
        Tuple of (precision, recall, f1), each as a float.
    """
    eps = 1e-12
    y_true01 = np.asarray(y_true01)
    y_pred01 = np.asarray(y_pred01)
    tp = np.sum((y_true01 == 1) & (y_pred01 == 1))
    fp = np.sum((y_true01 == 0) & (y_pred01 == 1))
    fn = np.sum((y_true01 == 1) & (y_pred01 == 0))
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return float(precision), float(recall), float(f1)


def confusion_matrix(y_true01, y_pred01):
    """
    Compute the 2x2 confusion matrix for binary classification.

    Args:
        y_true01: True labels in {0, 1} format.
        y_pred01: Predicted labels in {0, 1} format.

    Returns:
        A 2x2 numpy array [[TN, FP], [FN, TP]].
    """
    y_true01 = np.asarray(y_true01)
    y_pred01 = np.asarray(y_pred01)
    tn = np.sum((y_true01 == 0) & (y_pred01 == 0))
    fp = np.sum((y_true01 == 0) & (y_pred01 == 1))
    fn = np.sum((y_true01 == 1) & (y_pred01 == 0))
    tp = np.sum((y_true01 == 1) & (y_pred01 == 1))
    return np.array([[tn, fp], [fn, tp]], dtype=int)
