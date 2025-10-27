import os

from matplotlib.dates import SA

# ==========================
# PATHS & FILES
# ==========================
DATA_DIR = "./data/dataset/"
SAVE_DIR = "data_saving"
PICT_DIR = "picture"

RAW_DATA = os.path.join(SAVE_DIR, "raw_data.npz")

# ==========================
# before every: python run.py
# ==========================

# 1) ==========================
# change the name of the files, to avoid overwriting unless you don't care
PREPROC_DATA_PATH = os.path.join(SAVE_DIR, "preproc_data_2.npz")
TUNING_PATH = os.path.join(SAVE_DIR, "tuning_2.csv")
SAVE_WEIGHTS = os.path.join(SAVE_DIR, "final_weights.npy")

OUTPUT_PRED = "submission_best.csv"

# 2) ==========================
# check the pipeline

PREPROCESSING     = True    # reuse preprocessed npz if False
HYPERPARAM_TUNING = True   # tune or load best params 
SUBMISSION        = True   # Train final model, Save weights, 

RNG_SEED = 42


# =========================================================
# PREPROCESSING PARAMETERS
# =========================================================
DROP_FIRST_N_CAT_COLS = 26
LOW_CARD_MAX_UNIQUE = 20
MAX_ADDED_ONEHOT = 2000
ONEHOT_DROP_FIRST = True

PCA_VAR = 0.97
PCA_Local = {"variance_ratio": PCA_VAR, "min_cols": 8, "replace": True}
PCA_K = None
ORDINAL_ENCODE = True
ORDINAL_SCALE_TO_UNIT = True

NAN_INDICATOR_MIN_ABS_CORR = 0.1
NAN_INDICATOR_TOPK = None
NAN_INDICATOR_MIN_PREV = 0.1
NAN_INDICATOR_MAX_PREV = 0.9

STD_CONT = False

POLY_ENABLE = False
POLY_ADD_SQUARES_CONT = True
POLY_ADD_INTER_CONT = False
POLY_TOPK_PAIRS = 256
POLY_MIN_ABS_CORR = 0.00

PRUNE_CORR_THRESHOLD = 0.90
ADD_BIAS = False

EPS = 1e-12  # Avoid division by 0
N_FOLDS = 5  # Number of folds for cross-validation

# =========================================================
# HYPERPARAMETER TUNING
# =========================================================
TUNING_MAX_ITERS = 500
NAGFREE_TUNING = True

GAMMA_LOW = 1e-3  # Lower bound for gamma sampling
GAMMA_HIGH = 1.0  # Upper bound for gamma sampling
LAMBDA_LOW = 1e-8
LAMBDA_HIGH = 1e-5
NAGFREE_L_MAX = 1e8
N_TRIALS = 10

# Grid search parameters (for old grid search method)
LAMBDA = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
GAMMA = [1e-3, 1e-2, 1e-1, 1]
MAX_ITERS = 600  # Used in old implementations

USE_ADAM_DEFAULT = True
SCHEDULE_DEFAULT = "cosine"  # cosine, exponential, none
EARLY_STOP_DEFAULT = True
PATIENCE_DEFAULT = 15
TOL_DEFAULT = 1e-8

USE_WEIGHTED_BCE = True

SCHEDULE_CHOICES = ["exponential, cosine"]  # cosine, exponential, none
ADAM_CHOICES = [True]  # can just be better as it just make converge faster


# =========================================================
# FINAL TRAINING & SUBMISSION
# =========================================================
FINAL_MAX_ITERS = 400
CV_KFOLD_PREDICTION = False