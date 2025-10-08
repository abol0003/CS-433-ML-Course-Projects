# Preprocessing functions
import numpy as np
import config


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