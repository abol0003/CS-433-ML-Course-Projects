# Configuration parameters

import os 

DATA_DIR = r"./data/dataset/"
OUTPUT_PRED = "submission_best.csv"

PICT_DIR = "picture"
CONF_MAT_FIG = os.path.join(PICT_DIR, "confusion_matrix.png")
ROC_FIG      = os.path.join(PICT_DIR, "roc_curve.png")
PR_FIG       = os.path.join(PICT_DIR, "pr_curve.png")

SAVE_DIR = "data_saving"
RAW_DATA = os.path.join(SAVE_DIR, "raw_data.npz")
SAVE_PREPROCESSED = os.path.join(SAVE_DIR, "preprocessed_data.npz")
SAVE_BEST         = os.path.join(SAVE_DIR, "best_params.npz")
SAVE_WEIGHTS      = os.path.join(SAVE_DIR, "final_weights.npy")

# Pipeline

DO_PREPROCESS = True # reuse preprocessed npz if False
DO_TUNE       = False # Tune hyperparameters using K-Fold CV or load best params
DO_SUBMISSION = True # When True: train final model, save weights, build submission & plots

# Miscellaneous

RNG_SEED = 42

# Hyperparameters tuning via CV

TUNING_MAX_ITERS = 600 # Iterations executed during K-fold trail training
LAMBDA_LOW, LAMBDA_HIGH = 1e-7, 1e-1 # Lambda values range
GAMMA_LOW, GAMMA_HIGH = 1e-4, 5e-1 # Gamma value range
N_TRIALS = 120 # Amount of trials

# Final training parameters

FINAL_MAX_ITERS  = 3000

# Light One-hot Encoding

LOW_CARD_MAX_UNIQUE = 10    # Decides if a column should be encoded
ONEHOT_PER_FEAT_MAX = 8     
MAX_ADDED_ONEHOT    = 120   # Limits the total number of new columns added by encoding

# Nature of the features

CAT = [1, 2,  ]      # categorical feature indices
DISC = [5, 8, 15, 22]     # discrete feature indices
CONT = [0, 1, 2, 4, 6]    # continuous feature indices

# ADAM Parameters

ADAM_BATCH_SIZE = 512 # None | Int
ADAM_LOSS_TYPE = "weighted_bce" # weighted_bce | focal
ADAM_LR_DECAY = "sqrt" # None | "sqrt" | "exp" | "cos"
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999