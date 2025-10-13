# all configuration variables and constants
import os 

DATA_DIR = r"./data/dataset/"
OUTPUT_PRED = "submission_best.csv"

PICT_DIR = "picture"
CONF_MAT_FIG = os.path.join(PICT_DIR, "confusion_matrix.png")
ROC_FIG      = os.path.join(PICT_DIR, "roc_curve.png")
PR_FIG       = os.path.join(PICT_DIR, "pr_curve.png")

# Paths 
SAVE_DIR = "data_saving"
RAW_DATA = os.path.join(SAVE_DIR, "raw_data.npz")
PREPROC1_DATA_PATH = os.path.join(SAVE_DIR, "preproc_data_1.npz")
PREPROC2_DATA_PATH = os.path.join(SAVE_DIR, "preproc_data_2.npz")
SAVE_BEST         = os.path.join(SAVE_DIR, "best_params.npz")
SAVE_WEIGHTS      = os.path.join(SAVE_DIR, "final_weights.npy")

# Pipeline 
PREPROCESSING = True    # reuse preprocessed npz if False
DO_TUNE       = True    # tune or load best params #HYPERPARAM_TUNING
DO_SUBMISSION = True     # when True: train final model, save weights, build submission & plots #SUBMISSIOn

RNG_SEED = 42

# Tuning parameters 
HOLDOUT_VAL_FRAC = 0.20
TUNING_MAX_ITERS = 400
FINAL_MAX_ITERS  = 600
# GAMMA_GRID  = [1.515e-3, 1.52e-3, 1.525e-3, 1.49e-3, 1.505e-3]
# LAMBDA_GRID = [1.5e-3, 1.505e-3, 1.502e-3, 1.485e-3, 1.49e-3]
# THRESHOLDS  = [0.5302, 0.531, 0.527, 0.53, 0.529, 0.532]
GAMMA_GRID = [1e-3]#, 5e-4]
LAMBDA_GRID = [1e-3]#, 1.5e-3, 1e-1]
THRESHOLDS = [0.50, 0.55]#, 0.60, 0.65, 0.70, 0.75]
#[BEST] lambda=1.500e-03, gamma=1.520e-03, thr=0.53, ACC=0.8586, P=0.3203, R=0.5355, F1=0.4008 
    #0.383	0.902

# Light One-hot Encoding 
LOW_CARD_MAX_UNIQUE = 10    # Decides if a column should be encoded
ONEHOT_PER_FEAT_MAX = 8     
MAX_ADDED_ONEHOT    = 120   # Limits the total number of new columns added by encoding

# Nature of the features
CAT = [1, 2,  ]      # categorical feature indices
DISC = [5, 8, 15, 22]     # discrete feature indices
CONT = [0, 1, 2, 4, 6]    # continuous feature indices