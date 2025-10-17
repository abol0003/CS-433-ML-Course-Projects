# all configuration variables and constants
import os 

#====================================================

DATA_DIR = r"./data/dataset/"
OUTPUT_PRED = "submission_best.csv"

PICT_DIR = "picture"
# CONF_MAT_FIG = os.path.join(PICT_DIR, "confusion_matrix.png")
# ROC_FIG      = os.path.join(PICT_DIR, "roc_curve.png")
# PR_FIG       = os.path.join(PICT_DIR, "pr_curve.png")

# Paths 
SAVE_DIR = "data_saving"
RAW_DATA = os.path.join(SAVE_DIR, "raw_data.npz")

#PREPROC1_DATA_PATH = os.path.join(SAVE_DIR, "preproc_data_1.npz")

#CHANGE iter the ..._{iter} if changes to not overwrite preproc data you might want to keep 
PREPROC_DATA_PATH = os.path.join(SAVE_DIR, "preproc_data_2.npz")

#CHANGE iter the ..._{iter} if changes in preproc data (or when needed)
TUNING_PATH         = os.path.join(SAVE_DIR, "tuning_1.csv") # saves .csv file 
BEST_PARAM_PATH     = os.path.join(SAVE_DIR, "bestParam_1.csv") # saves .npz file 

#SAVE_BEST           = os.path.join(SAVE_DIR, "best_params.npz") # to suppress 
SAVE_WEIGHTS        = os.path.join(SAVE_DIR, "final_weights.npy") # To not have to retrain the model ??   

#====================================================

# Pipeline 
DO_PREPROCESS = False    # reuse preprocessed npz if False
DO_TUNE       = True    # tune or load best params
DO_SUBMISSION = True     # when True: train final model, save weights, build submission & plots

PREPROCESSING     = False    # reuse preprocessed npz if False
HYPERPARAM_TUNING = True    # tune or load best params 
SUBMISSION        = False   # Train final model, Save weights, Submission file

RNG_SEED = 42

#====================================================
#Badr Tuning
# Coarse
LAMBDA = [1e-4]    #[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
GAMMA = [1e-2]    #[1e-3, 1e-2, 1e-1, 1]
MAX_ITERS = 600
#initial_w = np.zeros(X_train.shape[1])

# Refine
# LAMBDA = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
# GAMMA = [1e-3, 1e-2, 1e-1, 1]

#====================================================
# Light One-hot Encoding 
LOW_CARD_MAX_UNIQUE = 10    # Decides if a column should be encoded
ONEHOT_PER_FEAT_MAX = 8     
MAX_ADDED_ONEHOT    = 120   # Limits the total number of new columns added by encoding

#====================================================
# Nature of the features (Preproc)
# Future masks ??
CAT = [1, 2,  ]             # categorical feature indices
DISC = [5, 8, 15, 22]       # discrete feature indices
CONT = [0, 1, 2, 4, 6]      # continuous feature indices


EPS = 1e-12 # Avoid division by 0
N_FOLDS = 5