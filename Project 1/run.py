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

os.makedirs(config.PICT_DIR, exist_ok=True)
os.makedirs(config.SAVE_DIR, exist_ok=True)
np.random.seed(config.RNG_SEED)


def binary_clf_curves(y_true01, scores):
    y = np.asarray(y_true01)
    s = np.asarray(scores)
    order = np.argsort(-s)
    y = y[order]
    P = np.sum(y == 1)
    N = np.sum(y == 0)
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    tpr = tps / (P + 1e-12)
    fpr = fps / (N + 1e-12)
    precision = tps / (tps + fps + 1e-12)
    recall = tpr
    roc_auc = float(np.trapezoid(tpr, fpr))
    pr_auc = float(np.trapezoid(precision, recall))
    return fpr, tpr, precision, recall, roc_auc, pr_auc


def evaluate_and_plot_final(X_tr, y_tr_01, va_idx, probs_va, thr, out_prefix=""):
    """Compute final metrics on hold-out (with final model), save plots."""
    preds_va = (probs_va >= thr).astype(int)
    acc = metrics.accuracy_score(y_tr_01[va_idx], preds_va)
    prec, rec, f1 = metrics.precision_recall_f1(y_tr_01[va_idx], preds_va)
    print(f"[FINAL] ACC={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")

    # Confusion matrix
    cm = metrics.confusion_matrix(y_tr_01[va_idx], preds_va)
    plots.plot_confusion_matrix(cm, config.CONF_MAT_FIG, class_names=("0","1"))
    print(f"[Figure] Confusion matrix -> {config.CONF_MAT_FIG}")

    # ROC & PR 
    fpr, tpr, precision, recall, roc_auc, pr_auc = binary_clf_curves(y_tr_01[va_idx], probs_va)
    plots.plot_curve(fpr, tpr, config.ROC_FIG, xlabel="FPR", ylabel="TPR", title=f"ROC (AUC={roc_auc:.4f})")
    print(f"[Figure] ROC curve -> {config.ROC_FIG}")
    plots.plot_curve(recall, precision, config.PR_FIG, xlabel="Recall", ylabel="Precision", title=f"PR (AUC={pr_auc:.4f})")
    print(f"[Figure] PR curve -> {config.PR_FIG}")


#========================================
#the two functions above will be removed. 
#========================================


def sample_loguniform(low, high, size, rng=np.random.RandomState(config.RNG_SEED)):
    lo, hi = np.log(low), np.log(high)
    return np.exp(rng.uniform(lo, hi, size))


def preprocess_data():
    t_preprocess = time.time()

    if config.DO_PREPROCESS:
        x_train, x_test, y_train_pm1, train_ids, test_ids = helpers.load_csv_data(config.DATA_DIR)

        X_tr, X_te = preprocessing.preprocess(x_train, x_test)
        
        # is it really necessary ? 
        # can't we not just do that to predictions ({-1, +1} to {0, 1})
        y_tr_01 = metrics.to_01_labels(y_train_pm1) 

        np.savez_compressed(
            config.SAVE_PREPROCESSED,
            X_train=X_tr, X_test=X_te, y_train=y_tr_01,
            train_ids=train_ids, test_ids=test_ids
        )
        print(f"[Saved] Preprocessed data -> {config.SAVE_PREPROCESSED}")

    else:
        if not os.path.exists(config.SAVE_PREPROCESSED):
            raise FileNotFoundError(f"{config.SAVE_PREPROCESSED} not found.")
        npz = np.load(config.SAVE_PREPROCESSED) 
        X_tr     = npz["X_train"]
        X_te     = npz["X_test"]
        y_tr_01  = npz["y_train"]
        train_ids = npz["train_ids"]
        test_ids  = npz["test_ids"]
        print(f"[Loaded] Preprocessed data from -> {config.SAVE_PREPROCESSED}")

    print(f"[Preprocessing] {time.time() - t_preprocess:.1f}s")
    return X_tr, X_te, y_tr_01, train_ids, test_ids


def tune_hyperparameter(X_tr, y_tr_01, folds):
    t_tune = time.time()

    if config.DO_TUNE:       
        #Random search over log-uniform grid ( better for computationnal cost )
        LAMBDA_LOW, LAMBDA_HIGH = 1e-6, 1e-2
        GAMMA_LOW,  GAMMA_HIGH  = 1e-3, 9e-1
        N_TRIALS = 30   
        lambda_samples = sample_loguniform(LAMBDA_LOW, LAMBDA_HIGH, N_TRIALS)
        gamma_samples  = sample_loguniform(GAMMA_LOW,  GAMMA_HIGH,  N_TRIALS)

        tasks = [(y_tr_01, X_tr, folds, lam, gam, config.TUNING_MAX_ITERS)
                for lam, gam in zip(lambda_samples, gamma_samples)]

        nproc = max(1, (os.cpu_count() or 2) - 1)
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
    t0 = time.time()
    print("Loading data from:", config.DATA_DIR)

    X_tr, X_te, y_tr_01, train_ids, test_ids = preprocess_data()

    #tr_idx, va_idx = split_train_val_stratified(y_tr_01, val_fraction=HOLDOUT_VAL_FRAC, seed=RNG_SEED)
       
    N_SPLITS = 5      
    folds = cv_utils.stratified_kfold_indices(y_tr_01, n_splits=N_SPLITS, seed=config.RNG_SEED)
    _, va_idx= folds[0]  #for final eval only

    best_lambda, best_gamma, best_thr = tune_hyperparameter(X_tr, y_tr_01, folds)


    if config.DO_SUBMISSION:
        w_final = train_final_model(X_tr, y_tr_01, best_lambda, best_gamma)
        make_submission(X_te, w_final, best_thr, test_ids)

        # validation metrics & plots using the final model
        #probs_va_final = implementations.sigmoid(X_tr[va_idx].dot(w_final))
        #evaluate_and_plot_final(X_tr, y_tr_01, va_idx, probs_va_final, best_thr)

        print(f"[TOTAL] {time.time() - t0:.1f}s.")
if __name__ == "__main__":
    main()
