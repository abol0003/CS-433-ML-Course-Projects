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


# TO SUPPRESS
def tune_hyperparameter(X_tr, y_tr_01, folds): #TO SUPPRESS FOR FINAL SUBMISSION
    t_tune = time.time()

    if config.DO_TUNE:       
        #Random search over log-uniform grid ( better for computationnal cost )

        lambda_samples = sample_loguniform(config.LAMBDA_LOW, config.LAMBDA_HIGH, config.N_TRIALS)
        gamma_samples  = sample_loguniform(config.GAMMA_LOW,  config.GAMMA_HIGH,  config.N_TRIALS)

        tasks = [(y_tr_01, X_tr, folds, lam, gam, config.TUNING_MAX_ITERS)
                for lam, gam in zip(lambda_samples, gamma_samples)]

        nproc = max(1, (os.cpu_count() or 2) - 4)
        with mp.get_context("spawn").Pool(processes=nproc) as pool:
            results = pool.map(cv_utils.cv_train_and_eval, tasks)

        best = None
        for (lam, gam, thr, acc, prec, rec, f1) in results:
            if (best is None) or (f1 > best[-1]):
                best = (lam, gam, thr, acc, prec, rec, f1)

        best_lambda, best_gamma, best_thr, val_acc, val_prec, val_rec, val_f1 = best
        print(f"[BEST-CV] lambda={best_lambda:.3e}, gamma={best_gamma:.3e}, thr={best_thr:.3f}, "
            f"ACC={val_acc:.4f}, P={val_prec:.4f}, R={val_rec:.4f}, F1={val_f1:.4f}")

        np.savez(
            config.SAVE_BEST,
            lambda_=best_lambda, gamma=best_gamma, thr=best_thr,
            acc=val_acc, prec=val_prec, rec=val_rec, f1=val_f1
        )
        print(f"[Saved] Best params -> {config.SAVE_BEST}")

    else:
        if not os.path.exists(config.SAVE_BEST):
            raise FileNotFoundError(f"{config.SAVE_BEST} not found.")
        npz = np.load(config.SAVE_BEST, allow_pickle=False)
        best_lambda = float(npz["lambda_"])
        best_gamma  = float(npz["gamma"])
        best_thr    = float(npz["thr"])
        val_acc     = float(npz["acc"])
        val_prec    = float(npz["prec"])
        val_rec     = float(npz["rec"])
        val_f1      = float(npz["f1"])
        print(f"[Loaded] Best params from -> {config.SAVE_BEST}")
        print(f"[BEST] lambda={best_lambda:.3e}, gamma={best_gamma:.3e}, thr={best_thr:.3f}, "
          f"ACC={val_acc:.4f}, P={val_prec:.4f}, R={val_rec:.4f}, F1={val_f1:.4f}")
    
    print(f"[Hyperparameter Tuning] {time.time() - t_tune:.1f}s")
    return best_lambda, best_gamma, best_thr


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
    probs_te = implementations.sigmoid(X_te.dot(w_final))
    preds01_te = (probs_te >= best_thr).astype(int)
    preds_pm1_te = metrics.to_pm1_labels(preds01_te)
    helpers.create_csv_submission(test_ids, preds_pm1_te, config.OUTPUT_PRED)
    print(f"[Submission] saved -> {config.OUTPUT_PRED}")


def main():

    t = time.time()
    if config.PREPROCESSING:
        x_train, x_test, y_train_pm1, train_ids, test_ids = helpers.load_csv_data(config.DATA_DIR)
        X_tr, X_te, ytr_01 = preprocessing.preprocess2(x_train, x_test, y_train_pm1, train_ids, test_ids, config.PREPROC2_DATA_PATH)
    else:
        ## MOVE THIS SHIT TO preprocessing.py 
        # call it as func with arg to load the right preproc data
        if not os.path.exists(config.PREPROC2_DATA_PATH):
            raise FileNotFoundError(f"{config.PREPROC2_DATA_PATH} not found.")
        npz = np.load(config.PREPROC2_DATA_PATH) 
        Xtr      = npz["X_train"]
        Xte      = npz["X_test"]
        ytr_01   = npz["y_train"]
        train_ids = npz["train_ids"]
        test_ids  = npz["test_ids"]
        print(f"[Loaded] Preprocessed data from -> {config.PREPROC2_DATA_PATH}")
    print(f"[Preprocessing] {time.time() - t:.1f}s")

    #==========================================
    t = time.time()   
    if config.HYPERPARAM_TUNING:
        best_params = tuning.tune(Xtr, ytr_01)
    else: 
        best_params = tuning.load_tuning_results() 
    print(f"[Tuning] {time.time() - t:.1f}s")
    #==========================================
    # Extract parameters
    best_lambda = best_params['lambda']
    best_gamma = best_params['learning_rate']
    best_threshold = best_params['optimal_threshold']
    #...


    # Final training + submission
    # if config.DO_SUBMISSION:
    #     # Enregistre les courbes train/val avec un holdout (run séparé, honnête)
    #     if getattr(config, "HOLDOUT_VAL_FRAC", 0.0) > 0:
    #         save_training_curves_with_holdout(
    #             X_tr, y_tr_01, best_lambda, best_gamma, use_adam=best_adam, schedule_name=best_sched
    #         )
    #     # Entraînement final sur tout le train (pour la soumission)
    #     w_final = train_final_model(
    #         X_tr, y_tr_01, best_lambda, best_gamma, use_adam=best_adam, schedule_name=best_sched
    #     )
    #     make_submission(X_te, w_final, best_thr, test_ids)
    #     print(f"[TOTAL] {time.time() - t:.1f}s.")


if __name__ == "__main__":
    main()
