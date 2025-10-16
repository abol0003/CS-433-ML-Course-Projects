# all configuration variables and constants
import os 

#====================================================

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

#====================================================

# Pipeline 
DO_PREPROCESS = False    # reuse preprocessed npz if False
DO_TUNE       = True    # tune or load best params
DO_SUBMISSION = True     # when True: train final model, save weights, build submission & plots
PREPROCESSING     = True    # reuse preprocessed npz if False
HYPERPARAM_TUNING = True    # tune or load best params 
SUBMISSION        = True    # Train final model, Save weights, Submission file

RNG_SEED = 42

#====================================================
# Tuning parameters 
#HOLDOUT_VAL_FRAC = 0.20
#TUNING_MAX_ITERS = 500  # Reduced from higher value for faster testing
FINAL_MAX_ITERS  = 1200
GAMMA_LOW = 1e-7  
GAMMA_HIGH = 1 
LAMBDA_LOW = 1e-10
LAMBDA_HIGH = 1e-1

N_TRIALS = 15  # Number of trials for hyperparameter tuning

#Badr Tuning
# Coarse
LAMBDA = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
GAMMA = [1e-3, 1e-2, 1e-1, 1]
MAX_ITERS = 1000
#initial_w = np.zeros(X_train.shape[1])
# Refine

#====================================================
# Light One-hot Encoding 
LOW_CARD_MAX_UNIQUE = 10    # Decides if a column should be encoded
ONEHOT_PER_FEAT_MAX = 8     
MAX_ADDED_ONEHOT    = 120   # Limits the total number of new columns added by encoding

#====================================================
# Nature of the features (Preproc)
CAT = [1, 2,  ]             # categorical feature indices
DISC = [5, 8, 15, 22]       # discrete feature indices
CONT = [0, 1, 2, 4, 6]      # continuous feature indices


EPS = 1e-12 # Avoid division by 0
N_FOLDS = 5