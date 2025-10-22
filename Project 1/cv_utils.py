import numpy as np 
import metrics
import implementations
import config


def schedule_none(lr0, t, T):
    """
    Constant learning rate schedule.

    Keeps the learning rate fixed during the entire optimization.
    Useful as a baseline for comparison with decaying schemes.
    """
    return lr0


def schedule_cosine(lr0, t, T):
    """
    Cosine annealing schedule.

    Smoothly decreases the learning rate following a half-cosine curve
    from the initial value to zero across all iterations. Provides a
    gradual decay that avoids abrupt convergence slowdowns and can
    improve generalization stability.
    """
    import math
    return lr0 * 0.5 * (1 + math.cos(math.pi * t / max(1, T)))


def schedule_exponential(lr0, t, T, decay=0.99):
    """
    Exponential decay schedule.

    Multiplies the current learning rate by a constant decay factor at
    every iteration. This produces a monotonic and fast decrease, often
    suitable for smooth convex losses or short optimization horizons.
    """
    return lr0 * (decay ** t)


def stratified_kfold_indices(y01, n_splits=5, seed=42):
    """
    Stratified K-Fold splitting.

    Builds reproducible train/validation partitions that preserve class
    proportions in each fold. Each validation set contains roughly the
    same fraction of positives and negatives as the full dataset, which
    ensures unbiased cross-validation when dealing with imbalanced data.
    """
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
        rng.shuffle(va_idx)
        mask = np.ones(y.shape[0], dtype=bool)
        mask[va_idx] = False
        tr_idx = np.where(mask)[0]
        folds.append((tr_idx, va_idx))
    return folds


def best_threshold_by_f1(y_true01, scores):
    """
    Determine the score threshold maximizing F1 score.

    Evaluates precision and recall for all possible cutoffs on sorted
    prediction scores and selects the one giving the highest harmonic
    mean (F1). This ensures the classification threshold balances false
    positives and false negatives optimally for imbalanced tasks.
    """
    eps = 1e-15
    y = np.asarray(y_true01)
    s = np.asarray(scores)
    order = np.argsort(-s)
    y = y[order]; s_sorted = s[order]
    P = np.sum(y == 1)
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    precision = tps / (tps + fps + eps)
    recall = tps / (P + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    k = int(np.argmax(f1))
    best_thr = s_sorted[k]
    return float(best_thr), float(precision[k]), float(recall[k]), float(f1[k])


def cv_train_and_eval(args):
    """
    Cross-validated training and model selection.

    Performs K-Fold training using either the NAG-Free optimizer or the
    Adam/GD variant depending on configuration. For each fold, the model
    is trained on the training split and evaluated on the validation one.
    Out-of-fold probabilities are aggregated to determine the global
    F1-optimal threshold. The function returns averaged metrics across
    folds, providing a robust performance estimate and helping to reduce
    variance in hyperparameter tuning.
    """
    (
        y_tr_01,
        X_tr,
        folds,
        lam,
        gam,
        max_iters,
        use_adam,
        schedule_name,
        early_stopping,
        patience,
        tol,
    ) = args
    if schedule_name == "cosine":
        schedule = schedule_cosine
    elif schedule_name == "exponential":
        schedule = schedule_exponential
    else:
        schedule = None
    per_fold_probs, per_fold_idx = [], []
    for (tr_idx, va_idx) in folds:
        w0 = np.zeros(X_tr.shape[1], dtype=np.float32)
        if config.NAGFREE_TUNING:
            w, _ = implementations.reg_logistic_regression_nagfree(
                y_tr_01[tr_idx],
                X_tr[tr_idx],
                lam,
                w0,
                max_iters=max_iters,
                tol=tol if tol > 0 else getattr(config, "TOL_DEFAULT", 1e-8),
                L_max=getattr(config, "NAGFREE_L_MAX", 1e8),
                verbose=False,
                val_data=(y_tr_01[va_idx], X_tr[va_idx]),
            )
        else:
            w, _ = implementations.reg_logistic_regression(
                y_tr_01[tr_idx],
                X_tr[tr_idx],
                lam,
                w0,
                max_iters=max_iters,
                gamma=gam,
                adam=bool(use_adam),
                schedule=schedule,
                early_stopping=bool(early_stopping),
                patience=patience,
                tol=tol if tol > 0 else getattr(config, "TOL_DEFAULT", 1e-8),
                verbose=False,
                val_data=(y_tr_01[va_idx], X_tr[va_idx]) if early_stopping else None,
            )
        probs_va = implementations.sigmoid(X_tr[va_idx].dot(w))
        per_fold_probs.append(probs_va)
        per_fold_idx.append(va_idx)

    va_idx_concat = np.concatenate(per_fold_idx)
    probs_concat = np.concatenate(per_fold_probs)
    y_val_concat = y_tr_01[va_idx_concat]
    best_thr, _, _, _ = best_threshold_by_f1(y_val_concat, probs_concat)

    acc_list, prec_list, rec_list, f1_list = [], [], [], []
    for probs_va, va_idx in zip(per_fold_probs, per_fold_idx):
        preds = (probs_va >= best_thr).astype(int)
        y_va = y_tr_01[va_idx]
        acc_list.append(metrics.accuracy_score(y_va, preds))
        p, r, f1 = metrics.precision_recall_f1(y_va, preds)
        prec_list.append(p)
        rec_list.append(r)
        f1_list.append(f1)

    return (lam, gam, float(best_thr), float(np.mean(acc_list)),
            float(np.mean(prec_list)), float(np.mean(rec_list)),
            float(np.mean(f1_list)))
