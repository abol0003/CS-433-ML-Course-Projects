# Cross-validation and training utilities 
import numpy as np 
import metrics
import implementations


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


def cv_train_and_eval(args):
    y_tr_01, X_tr, folds, lam, gam, max_iters = args
    #cross-validation with best threshold found on each fold
    per_fold_probs, per_fold_idx = [], []
    for (tr_idx, va_idx) in folds:
        w0 = np.zeros(X_tr.shape[1], dtype=np.float32)
        w, _ = implementations.reg_logistic_regression(y_tr_01[tr_idx], X_tr[tr_idx], lam, w0, max_iters, gam)
        probs_va = implementations.sigmoid(X_tr[va_idx].dot(w))
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
        acc_list.append(metrics.accuracy_score(y_va, preds))
        p, r, f1 = metrics.precision_recall_f1(y_va, preds)
        prec_list.append(p)
        rec_list.append(r)
        f1_list.append(f1)

    return (lam, gam, float(best_thr), float(np.mean(acc_list)), float(np.mean(prec_list)),
            float(np.mean(rec_list)), float(np.mean(f1_list)))



