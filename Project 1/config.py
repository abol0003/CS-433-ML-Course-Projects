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

#CHANGE iter the ..._{iter} if changes to not overwrite preproc data you might want to keep 
PREPROC_DATA_PATH = os.path.join(SAVE_DIR, "preproc_data_2.npz")

#CHANGE iter the ..._{iter} if changes in preproc data (or when needed)
TUNING_PATH         = os.path.join(SAVE_DIR, "tuning_2.csv") # saves .csv file 
BEST_PARAM_PATH     = os.path.join(SAVE_DIR, "bestParam_2.npz") # saves .npz file 

SAVE_WEIGHTS        = os.path.join(SAVE_DIR, "final_weights.npy") # To not have to retrain the model ??   

#====================================================
# Pipeline

PREPROCESSING     = True    # reuse preprocessed npz if False
HYPERPARAM_TUNING = False   # tune or load best params 
SUBMISSION        = False   # Train final model, Save weights, Submission file
#====================================================
#Tuning
LAMBDA = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
GAMMA = [1e-3, 1e-2, 1e-1, 1]
MAX_ITERS = 600


TUNING_MAX_ITERS = 400
#NAGFREE_TUNING = True

#GAMMA_LOW = 1e-3
#GAMMA_HIGH = 1.0
LAMBDA_LOW = 1e-8
LAMBDA_HIGH = 1e-5
#NAGFREE_L_MAX = 1e8
N_TRIALS = 10

USE_ADAM_DEFAULT = True
SCHEDULE_DEFAULT = "cosine" #cosine, exponential, none

# Early stopping defaults
EARLY_STOP_DEFAULT = True
PATIENCE_DEFAULT = 15
TOL_DEFAULT = 1e-8

USE_WEIGHTED_BCE = True 

SCHEDULE_CHOICES = ["exponential, cosine"] #cosine, exponential, none
ADAM_CHOICES = [True] #can just be better as it just make converge faster

#====================================================
# Light One-hot Encoding 
LOW_CARD_MAX_UNIQUE = 10    # Decides if a column should be encoded
ONEHOT_PER_FEAT_MAX = 8     
MAX_ADDED_ONEHOT    = 120   # Limits the total number of new columns added by encoding

#====================================================
K = 250 # for PCA
EPS = 1e-12 # Avoid division by 0
N_FOLDS = 5

RNG_SEED = 42
