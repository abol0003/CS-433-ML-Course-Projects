# Cross-validation and training utilities 
import numpy as np 
import metrics
import implementations


#stratified_kfold_indices
def create_stratified_folds(ytr_01, k=5, random_seed=42): 
    """
    Create stratified k-fold cross-validation indices.
    
    Args:
        y_binary: Binary labels (0 or 1)
        k: Number of folds
        random_seed: Random seed for reproducibility
        
    Returns:
        List of (train_indices, val_indices) tuples, one per fold
    """
    rng = np.random.RandomState(random_seed)
    y = np.asarray(ytr_01, dtype=int)

    # Seperate indices by class 
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    # Shuffle indices
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    # Splite each class into k equal parts 
    pos_folds = np.array_split(pos_idx, k)
    neg_folds = np.array_split(neg_idx, k)

    folds = []
    n_samples = len(ytr_01)

    for fold_idx in range(k):
        # Validation set:
        val_idx = np.concatenate([pos_folds[fold_idx], neg_folds[fold_idx]])
        rng.shuffle(val_idx)  # Mix positive and negative samples

        # Training set:
        is_train = np.ones(n_samples, dtype=bool)
        is_train[val_idx] = False
        train_idx = np.where(is_train)[0]

        folds.append((train_idx, val_idx))

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



