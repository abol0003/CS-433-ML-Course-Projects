import os
import time
import numpy as np
import multiprocessing as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import helpers 
# clearer than : from helpers import * As we use suffixes.
import implementations 
# avoid using Aliases: import implementations as impl
import config 
import preprocessing 
import metrics 
import plots
import cv_utils
import tuning

os.makedirs(config.PICT_DIR, exist_ok=True)
os.makedirs(config.SAVE_DIR, exist_ok=True)
np.random.seed(config.RNG_SEED)


def train_final_model(X_tr, y_tr_01, best_lambda, best_gamma):
    t_final = time.time()

    w0 = np.zeros(X_tr.shape[1], dtype=np.float32)
    w_final, final_loss = implementations.reg_logistic_regression(
        y_tr_01, X_tr, best_lambda, w0, max_iters=config.FINAL_MAX_ITERS, gamma=best_gamma
    )
    print(f"[Final] loss (unpenalized) = {final_loss:.6f}")

    np.save(config.SAVE_WEIGHTS, w_final)
    print(f"[Saved] Final weights -> {config.SAVE_WEIGHTS}")

    print(f"[Final Training] {time.time() - t_final:.1f}s")
    return w_final


def make_submission(X_te, w_final, best_thr, test_ids):
    """Generete predications and submission file."""
    yprob_te = implementations.sigmoid(X_te.dot(w_final))
    ypred_01_te = (yprob_te >= best_thr).astype(int)
    ypred_pm1_te = metrics.to_pm1_labels(ypred_01_te)
    helpers.create_csv_submission(test_ids, ypred_pm1_te, config.OUTPUT_PRED)
    print(f"[Submission] saved -> {config.OUTPUT_PRED}")


def main():

    t = time.time()
    Xtr, Xte, ytr_01 = preprocessing.preprocess2()
    print(f"[Preprocessing] {time.time() - t:.1f}s")

    #==========================================
    t = time.time()   
    #seperation of concerns : 1)tuning.py should contain the tuning logic 2) run.py simply orchestrate everything based on config.py
    best_params = tuning.tune(Xtr, ytr_01) 
    print(f"[Tuning] {time.time() - t:.1f}s")
    #==========================================
    # Extract parameters
    best_lambda = best_params['lambda']
    best_gamma = best_params['gamma']
    best_threshold = best_params['optimal_threshold']
    #...


    #Final training + submission
    t = time.time()   
    if config.DO_SUBMISSION:
        # Enregistre les courbes train/val avec un holdout (run séparé, honnête)
        w_final = train_final_model(
            Xtr, ytr_01, best_lambda, best_gamma
        )
        make_submission(X_te, w_final, best_threshold, test_ids)
        print(f"[TOTAL] {time.time() - t:.1f}s.")
    print(f"[Submission] {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
