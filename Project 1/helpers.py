"""Some helper functions for project 1."""

import csv
import numpy as np
import os


def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


#=======================================
# Metrics and label conversion functions 
#=======================================


def to01_labels(y_pm1):
    return (y_pm1 > 0).astype(np.uint8)


def to_pm1_labels(y01):
    return np.where(y01 == 1, 1, -1).astype(np.int32)


def accuracy_score(y_true01, y_pred01):
    y_true01 = np.asarray(y_true01)
    y_pred01 = np.asarray(y_pred01)
    return float(np.mean(y_true01 == y_pred01))


def precision_recall_f1(y_true01, y_pred01):
    eps=1e-12
    y_true01 = np.asarray(y_true01)
    y_pred01 = np.asarray(y_pred01)
    tp = np.sum((y_true01 == 1) & (y_pred01 == 1))
    fp = np.sum((y_true01 == 0) & (y_pred01 == 1))
    fn = np.sum((y_true01 == 1) & (y_pred01 == 0))
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    return float(precision), float(recall), float(f1)


def confusion_matrix(y_true01, y_pred01):
    y_true01 = np.asarray(y_true01)
    y_pred01 = np.asarray(y_pred01)
    tn = np.sum((y_true01 == 0) & (y_pred01 == 0))
    fp = np.sum((y_true01 == 0) & (y_pred01 == 1))
    fn = np.sum((y_true01 == 1) & (y_pred01 == 0))
    tp = np.sum((y_true01 == 1) & (y_pred01 == 1))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


# def split_train_val_stratified(y01, val_fraction=0.2, seed=42):
#     rng = np.random.RandomState(seed)
#     pos = np.where(y01 == 1)[0]
#     neg = np.where(y01 == 0)[0]
#     rng.shuffle(pos); rng.shuffle(neg) #mix neg and pos separatly to have balanced val
#     n_pos_val = int(len(pos) * val_fraction)
#     n_neg_val = int(len(neg) * val_fraction)
#     val_idx = np.concatenate([pos[:n_pos_val], neg[:n_neg_val]]) #make sure we have the same ratio
#     rng.shuffle(val_idx)#mix pos and neg again
#     mask = np.ones(y01.shape[0], dtype=bool); mask[val_idx] = False
#     train_idx = np.where(mask)[0]
#     return train_idx, val_idx