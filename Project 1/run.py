import os
import time
import numpy as np
import multiprocessing as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import helpers 
# clearer than : from helpers import * As we use suffixes.
import implementations as impl

import config 

# =========================
# User config
# =========================

os.makedirs(config.PICT_DIR, exist_ok=True)
os.makedirs(config.SAVE_DIR, exist_ok=True)
np.random.seed(config.RNG_SEED)

# =========================
# ROC/PR ( no sklearn )
# =========================
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

def plot_confusion_matrix(cm, path, class_names=("0","1")):
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues") 
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def plot_curve(x, y, path, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.plot(x, y, linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)

# =========================
# Preprocessing
# =========================
def _drop_constant_and_naonly(X):
    """Return indices of non-constant and non-NA-only columns."""
    cols = []
    for j in range(X.shape[1]):
        col = X[:, j]
        valid = ~np.isnan(col)
        if not np.any(valid):
            continue
        vals = col[valid]
        if np.all(vals == vals[0]):
            continue
        cols.append(j)
    return np.array(cols, dtype=int)

# def variance_filter(X, cols, min_var=0.0):
#     if min_var <= 0:
#         return cols
#     keep = []
#     for j in cols:
#         v = np.nanvar(X[:, j])
#         if v >= min_var:
#             keep.append(j)
#     return np.array(keep, dtype=int)

def one_hot_encoding(Xtr, Xte, max_unique=10, per_feat_cap=8, global_cap=120):
    Xtr = np.asarray(Xtr)
    Xte = np.asarray(Xte)
    n_tr, d = Xtr.shape
    assert Xte.shape[1] == d

    new_tr_cols, new_te_cols = [], []
    used_idx = []
    plan = []
    added = 0

    for j in range(d):
        col_tr = Xtr[:, j]
        valid = ~np.isnan(col_tr)
        if not np.any(valid):
            continue
        uniq = np.unique(col_tr[valid])
        if uniq.shape[0] <= max_unique:
            uniq_capped = uniq[:min(len(uniq), per_feat_cap)]
            k_add = max(len(uniq_capped) - 1, 0)
            if added + k_add > global_cap:
                continue
            values_to_encode = uniq_capped[:-1]
            if values_to_encode.size > 0:
                used_idx.append(j)
                plan.append((j, values_to_encode))
                added += k_add
                col_te = Xte[:, j]
                for v in values_to_encode:
                    new_tr_cols.append((col_tr == v).astype(np.float32))
                    new_te_cols.append((col_te == v).astype(np.float32))

    keep_idx = [j for j in range(d) if j not in used_idx]
    Xtr_keep, Xte_keep = Xtr[:, keep_idx], Xte[:, keep_idx]
    if new_tr_cols:
        Xtr_new = np.column_stack([Xtr_keep] + new_tr_cols)
        Xte_new = np.column_stack([Xte_keep] + new_te_cols)
    else:
        Xtr_new, Xte_new = Xtr_keep, Xte_keep

    return Xtr_new, Xte_new, keep_idx, used_idx, plan

def preprocess(x_train, x_test, printable=True):
    """Preprocess train/test sets, return processed matrices."""
    Xtr = np.array(x_train, dtype=np.float32, copy=True)
    Xte = np.array(x_test,  dtype=np.float32, copy=True)

    n_tr, d = Xtr.shape
    if printable:
        print(f"[Preprocess] n_train={n_tr}, n_test={Xte.shape[0]}, n_features={d}")

    # Mean imputation: replace NaN by mean of the column (0 if all NaN)
    col_mean = np.nanmean(Xtr, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    inds_tr = np.where(np.isnan(Xtr))
    inds_te = np.where(np.isnan(Xte))
    Xtr[inds_tr] = np.take(col_mean, inds_tr[1])
    Xte[inds_te] = np.take(col_mean, inds_te[1])

    # Remove constant / NaN-only columns 
    col_keep = _drop_constant_and_naonly(Xtr)
    #col_keep = variance_filter(Xtr, col_keep)
    Xtr = Xtr[:, col_keep]
    Xte = Xte[:, col_keep]
    if printable:
        print(f"[Preprocess] drop const/NA-only -> keep {Xtr.shape[1]} cols")

    # Light one-hot 
    Xtr_new, Xte_new, keep_idx, used_idx, plan = one_hot_encoding(Xtr, Xte,
    max_unique = config.LOW_CARD_MAX_UNIQUE,
    per_feat_cap = config.ONEHOT_PER_FEAT_MAX,
    global_cap = config.MAX_ADDED_ONEHOT)
    if printable:
        print(f"[Preprocess] one-hot: kept {len(keep_idx)} raw cols, "
          f"encoded {len(used_idx)} cols, plan size={sum(len(v) for _, v in plan)}")
    # Standardization
    mean_tr = np.mean(Xtr_new, axis=0).astype(np.float32)
    std_tr  = np.std(Xtr_new, axis=0).astype(np.float32)
    std_tr  = np.where(std_tr == 0, 1.0, std_tr)
    Xtr_s = (Xtr_new - mean_tr) / std_tr
    Xte_s = (Xte_new - mean_tr) / std_tr

    # Bias term for w_0
    Xtr_f = np.hstack([np.ones((Xtr_s.shape[0], 1), dtype=np.float32), Xtr_s])
    Xte_f = np.hstack([np.ones((Xte_s.shape[0], 1), dtype=np.float32), Xte_s])

    if printable:
        print(f"[Preprocess] final dims: train={Xtr_f.shape}, test={Xte_f.shape}")

    return Xtr_f, Xte_f

# =========================
# Model utils
# =========================
def evaluate_and_plot_final(X_tr, y_tr_01, va_idx, probs_va, thr, out_prefix=""):
    """Compute final metrics on hold-out (with final model), save plots."""
    preds_va = (probs_va >= thr).astype(int)
    acc = helpers.accuracy_score(y_tr_01[va_idx], preds_va)
    prec, rec, f1 = helpers.precision_recall_f1(y_tr_01[va_idx], preds_va)
    print(f"[FINAL] ACC={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")

    # Confusion matrix
    cm = helpers.confusion_matrix(y_tr_01[va_idx], preds_va)
    plot_confusion_matrix(cm, config.CONF_MAT_FIG, class_names=("0","1"))
    print(f"[Figure] Confusion matrix -> {config.CONF_MAT_FIG}")

    # ROC & PR 
    fpr, tpr, precision, recall, roc_auc, pr_auc = binary_clf_curves(y_tr_01[va_idx], probs_va)
    plot_curve(fpr, tpr, config.ROC_FIG, xlabel="FPR", ylabel="TPR", title=f"ROC (AUC={roc_auc:.4f})")
    print(f"[Figure] ROC curve -> {config.ROC_FIG}")
    plot_curve(recall, precision, config.PR_FIG, xlabel="Recall", ylabel="Precision", title=f"PR (AUC={pr_auc:.4f})")
    print(f"[Figure] PR curve -> {config.PR_FIG}")



def stratified_kfold_indices(y01, n_splits=5, seed=42): # %val=1/n_splits
    rng = np.random.RandomState(seed)
    y = np.asarray(y01).astype(int)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng.shuffle(pos); rng.shuffle(neg)
    pos_splits = np.array_split(pos, n_splits)
    neg_splits = np.array_split(neg, n_splits)
    folds = []
    for k in range(n_splits):
        va_idx = np.concatenate([pos_splits[k], neg_splits[k]])
        rng.shuffle(va_idx)  #mix pos and neg again
        mask = np.ones(y.shape[0], dtype=bool)
        mask[va_idx] = False
        tr_idx = np.where(mask)[0]
        folds.append((tr_idx, va_idx))
    return folds

# def cv_train_and_eval(args):
#     y_tr_01, X_tr, folds, lam, gam, max_iters = args

#     all_va_idx = []
#     all_probs  = []

#     for (tr_idx, va_idx) in folds:
#         w0 = np.zeros(X_tr.shape[1], dtype=np.float32)
#         w, _ = impl.reg_logistic_regression(
#             y_tr_01[tr_idx], X_tr[tr_idx], lam, w0, max_iters, gam
#         )
#         probs_va = impl.sigmoid(X_tr[va_idx].dot(w))
#         all_va_idx.append(va_idx)
#         all_probs.append(probs_va)

#     va_idx_concat = np.concatenate(all_va_idx)
#     probs_concat  = np.concatenate(all_probs)
#     y_val_concat  = y_tr_01[va_idx_concat]

#     best_thr, best_prec, best_rec, best_f1 = best_threshold_by_f1(y_val_concat, probs_concat)
#     acc = accuracy_score(y_val_concat, (probs_concat >= best_thr).astype(int))

#     return (lam, gam, best_thr, acc, best_prec, best_rec, best_f1)


def cv_train_and_eval(args):
    y_tr_01, X_tr, folds, lam, gam, max_iters = args
    #cross-validation with best threshold found on each fold
    per_fold_probs, per_fold_idx = [], []
    for (tr_idx, va_idx) in folds:
        w0 = np.zeros(X_tr.shape[1], dtype=np.float32)
        w, _ = impl.reg_logistic_regression(y_tr_01[tr_idx], X_tr[tr_idx], lam, w0, max_iters, gam)
        probs_va = impl.sigmoid(X_tr[va_idx].dot(w))
        per_fold_probs.append(probs_va)
        per_fold_idx.append(va_idx)

    va_idx_concat = np.concatenate(per_fold_idx)
    probs_concat  = np.concatenate(per_fold_probs)
    y_val_concat  = y_tr_01[va_idx_concat]
    best_thr, _, _, _ = best_threshold_by_f1(y_val_concat, probs_concat)
    #evaluate with best_thr on each fold and average ( see slide 4a pg 24)
    acc_list, prec_list, rec_list, f1_list = [], [], [], []
    for probs_va, va_idx in zip(per_fold_probs, per_fold_idx):
        preds = (probs_va >= best_thr).astype(int)
        y_va  = y_tr_01[va_idx]
        acc_list.append(helpers.accuracy_score(y_va, preds))
        p, r, f1 = helpers.precision_recall_f1(y_va, preds)
        prec_list.append(p)
        rec_list.append(r)
        f1_list.append(f1)

    return (lam, gam, float(best_thr), float(np.mean(acc_list)), float(np.mean(prec_list)),
            float(np.mean(rec_list)), float(np.mean(f1_list)))


import math

def sample_loguniform(low, high, size, rng=np.random.RandomState(config.RNG_SEED)):
    lo, hi = math.log(low), math.log(high)
    return np.exp(rng.uniform(lo, hi, size))

def best_threshold_by_f1(y_true01, scores):
    y = np.asarray(y_true01)
    s = np.asarray(scores)
    order = np.argsort(-s)
    y = y[order]; s_sorted = s[order]
    P = np.sum(y == 1)
    #vectorized computation of precision/recall/f1
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    precision = tps / (tps + fps + 1e-12)
    recall    = tps / (P + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)
    k = int(np.argmax(f1))
    best_thr = s_sorted[k]               
    return float(best_thr), float(precision[k]), float(recall[k]), float(f1[k])

# =========================
# Main
# =========================
def main():
    t0 = time.time()
    print("Loading data from:", config.DATA_DIR)

    if config.DO_PREPROCESS:
        x_train, x_test, y_train_pm1, train_ids, test_ids = helpers.load_csv_data(config.DATA_DIR)
        uniq, cnt = np.unique(y_train_pm1, return_counts=True)
        print("Label counts (in {-1,+1}):", dict(zip(uniq.astype(int), cnt)))

        X_tr, X_te = preprocess(x_train, x_test,True)
        y_tr_01 = helpers.to01_labels(y_train_pm1)

        np.savez_compressed(
            config.SAVE_PREPROCESSED,
            X_train=X_tr, X_test=X_te, y_train=y_tr_01,
            train_ids=train_ids, test_ids=test_ids
        )
        print(f"[Saved] Preprocessed data -> {config.SAVE_PREPROCESSED}")

    else:
        if not os.path.exists(config.SAVE_PREPROCESSED):
            raise FileNotFoundError(f"{config.SAVE_PREPROCESSED} not found.")
        npz = np.load(config.SAVE_PREPROCESSED, allow_pickle=False)
        X_tr     = npz["X_train"]
        X_te     = npz["X_test"]
        y_tr_01  = npz["y_train"]
        train_ids = npz["train_ids"]
        test_ids  = npz["test_ids"]
        print(f"[Loaded] Preprocessed data from -> {config.SAVE_PREPROCESSED}")
        print(f"[Shapes] X_tr={X_tr.shape}, X_te={X_te.shape}, y={y_tr_01.shape}")


    #tr_idx, va_idx = split_train_val_stratified(y_tr_01, val_fraction=HOLDOUT_VAL_FRAC, seed=RNG_SEED)
    N_TRIALS = 30      
    N_SPLITS = 5      
    folds = stratified_kfold_indices(y_tr_01, n_splits=N_SPLITS, seed=config.RNG_SEED)
    _, va_idx= folds[0]  #for final eval only
    # == Tuning 
    if config.DO_TUNE:       
        #Random search over log-uniform grid ( better for computationnal cost )
        LAMBDA_LOW, LAMBDA_HIGH = 1e-6, 1e-2
        GAMMA_LOW,  GAMMA_HIGH  = 1e-3, 9e-1

        lambda_samples = sample_loguniform(LAMBDA_LOW, LAMBDA_HIGH, N_TRIALS)
        gamma_samples  = sample_loguniform(GAMMA_LOW,  GAMMA_HIGH,  N_TRIALS)

        tasks = [(y_tr_01, X_tr, folds, lam, gam, config.TUNING_MAX_ITERS)
                for lam, gam in zip(lambda_samples, gamma_samples)]

        nproc = max(1, (os.cpu_count() or 2) - 1)
        with mp.get_context("spawn").Pool(processes=nproc) as pool:
            results = pool.map(cv_train_and_eval, tasks)

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
        
    # == Final training 
    if config.DO_SUBMISSION:
        w0 = np.zeros(X_tr.shape[1], dtype=np.float32)
        w_final, final_loss = impl.reg_logistic_regression(
            y_tr_01, X_tr, best_lambda, w0, max_iters=config.FINAL_MAX_ITERS, gamma=best_gamma
        )
        print(f"[Final] loss (unpenalized) = {final_loss:.6f}")

        np.save(config.SAVE_WEIGHTS, w_final)
        print(f"[Saved] Final weights -> {config.SAVE_WEIGHTS}")

        # Test predictions + submission
        probs_te = impl.sigmoid(X_te.dot(w_final))
        preds01_te = (probs_te >= best_thr).astype(int)
        preds_pm1_te = helpers.to_pm1_labels(preds01_te)
        helpers.create_csv_submission(test_ids, preds_pm1_te, config.OUTPUT_PRED)
        print(f"[Submission] saved -> {config.OUTPUT_PRED}")

        # validation metrics & plots using the final model
        probs_va_final = impl.sigmoid(X_tr[va_idx].dot(w_final))
        evaluate_and_plot_final(X_tr, y_tr_01, va_idx, probs_va_final, best_thr)

        print(f"Done in {time.time() - t0:.1f}s.")

if __name__ == "__main__":
    main()
