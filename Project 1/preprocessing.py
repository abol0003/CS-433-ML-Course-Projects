import numpy as np
import config


def remove_low_validity_features(Xtr, Xte, threshold=0.20):
    """Remove columns whose fraction of non-missing values in the training data is below the threshold.

    Args:
        Xtr (np.ndarray): Training matrix.
        Xte (np.ndarray): Test matrix.
        threshold (float): Minimum required fraction of valid values to keep a feature.

    Returns:
        tuple[np.ndarray, np.ndarray]: Filtered training and test matrices.
    """
    valid_ratio = np.mean(~np.isnan(Xtr), axis=0)
    keep_mask = valid_ratio > float(threshold)
    keep_idx = np.where(keep_mask)[0]
    print(f"[Preprocess] low-validity: kept {keep_idx.size}/{Xtr.shape[1]} (thr={threshold*100:.1f}% valid)")
    return Xtr[:, keep_idx], Xte[:, keep_idx]


def replace_brfss_special_codes(X):
    """Normalize BRFSS special codes and convert units for weight-/height-like columns.

    - Four digits: 7777/9999 -> NaN; 8888 -> 0
    - Three digits: 777/999 -> NaN; 888/555 -> 0
    - Two digits: 77/99 -> NaN; 88 -> 0
    - If a column is detected as weight-like: values in (9000, 9999) become x - 9000*2.20462
    - If a column is detected as height-like: values in [200, 711] interpreted as ft'in and converted to cm
    """
    X = np.array(X, dtype=np.float32, copy=True)
    for j in range(X.shape[1]):
        x = X[:, j] 
        l = 0
        if np.sum((x > 9000) & (x < 9999)) >= 400:
            l = 1 if np.sum(x < 200) > 10 else 2
        if np.isin(x, [7777.0, 8888.0, 9999.0]).any():
            x = np.where(x == 7777.0, np.nan, x)
            x = np.where(x == 9999.0, np.nan, x)
            x = np.where(x == 8888.0, 0.0, x)
        elif np.isin(x, [777.0, 888.0, 999.0, 555.0]).any():
            x = np.where(x == 777.0, np.nan, x)
            x = np.where(x == 999.0, np.nan, x)
            x = np.where(x == 888.0, 0.0, x)
            x = np.where(x == 555.0, 0.0, x)
        elif np.isin(x, [77.0, 88.0, 99.0]).any():
            x = np.where(x == 77.0, np.nan, x)
            x = np.where(x == 99.0, np.nan, x)
            x = np.where(x == 88.0, 0.0, x)
        if l == 1:
            x = np.where((x > 9000.0) & (x < 9999.0), x - 9000.0 * 2.20462, x)
        elif l == 2:
            m = (x >= 200.0) & (x <= 711.0)
            if np.any(m):
                ft = (x // 100.0)
                inch = (x % 100.0)
                x = np.where(m, ft * 30.48 + inch * 2.54, x)
        X[:, j] = x
    return X


def is_integer_array(v, tol=1e-6):
    """Return True if all non-NaN values are within a tolerance of an integer.

    Args:
        v (np.ndarray): Input vector.
        tol (float): Numerical tolerance.

    Returns:
        bool: True if all non-NaN values are near integers.
    """
    v = v[~np.isnan(v)]
    if v.size == 0:
        return False
    return np.all(np.abs(v - np.round(v)) < tol)


def infer_feature_types(X, max_unique_cat=None):
    """Infer coarse feature types: binary, nominal, ordinal, or continuous.

    Uses simple rules: boolean-like to binary, low-cardinality integer-like to ordinal,
    other low-cardinality to nominal, and the rest to continuous.

    Args:
        X (np.ndarray): Data matrix.
        max_unique_cat (int, optional): Upper bound for detecting low-cardinality features.

    Returns:
        dict: Keys are binary, nominal, ordinal, continuous with lists of column indices.
    """
    if max_unique_cat is None:
        max_unique_cat = config.LOW_CARD_MAX_UNIQUE
    _, d = X.shape
    types = {"binary": [], "nominal": [], "ordinal": [], "continuous": []}
    for j in range(d):
        col = X[:, j]
        v = col[~np.isnan(col)]
        if v.size == 0:
            types["nominal"].append(j)
            continue
        uniq = np.unique(v)
        nunique = uniq.size
        if nunique == 2 and set(np.round(uniq).tolist()).issubset({0, 1}):
            types["binary"].append(j)
            continue
        if nunique <= max_unique_cat:
            if is_integer_array(v):
                umin, umax = int(np.min(uniq)), int(np.max(uniq))
                if (umax - umin) <= 6:
                    types["ordinal"].append(j)
                else:
                    types["nominal"].append(j)
            else:
                types["nominal"].append(j)
            continue
        types["continuous"].append(j)
    return types


def smart_impute(
    Xtr, Xte,
    skew_rule=0.5,
    allnan_fill_cont=0.0,
    allnan_fill_nom=-1.0,
    allnan_fill_bin=0.0,
):
    """Impute missing values with simple, robust rules.

    Categorical (binary, nominal, ordinal) features are imputed with the mode, using fallbacks when a column is fully missing.
    Continuous features use the median when mean and median differ notably relative to the standard deviation, otherwise the mean.
    Ordinal score-like features in the unit interval with a handful of distinct levels are treated as continuous.

    Args:
        Xtr (np.ndarray): Training matrix.
        Xte (np.ndarray): Test matrix.
        skew_rule (float): Controls when to prefer median over mean for continuous columns.
        allnan_fill_cont (float): Fallback for continuous columns with only missing values.
        allnan_fill_nom (float): Fallback for nominal or ordinal columns with only missing values.
        allnan_fill_bin (float): Fallback for binary columns with only missing values.

    Returns:
        tuple[np.ndarray, np.ndarray]: Imputed training and test matrices.
    """
    Xtr = np.array(Xtr, dtype=np.float32, copy=True)
    Xte = np.array(Xte, dtype=np.float32, copy=True)
    n, d = Xtr.shape
    assert Xte.shape[1] == d

    types = infer_feature_types(Xtr, max_unique_cat=getattr(config, "LOW_CARD_MAX_UNIQUE", 20))
    cont_set = set(types["continuous"])

    for j in range(d):
        v = Xtr[:, j]
        w = v[~np.isnan(v)]
        if w.size == 0:
            continue
        w_min, w_max = float(np.min(w)), float(np.max(w))
        if (w_min >= -1e-6) and (w_max <= 1.0 + 1e-6):
            nunique = np.unique(np.round(w, 6)).size
            if 3 <= nunique <= 7:
                cont_set.add(j)

    is_cont = np.zeros(d, dtype=bool)
    if cont_set:
        is_cont[list(cont_set)] = True

    fam_bin = [j for j in types["binary"] if not is_cont[j]]
    fam_nom = [j for j in types["nominal"] if not is_cont[j]]
    fam_ord = [j for j in types["ordinal"] if not is_cont[j]]
    fam_cont = sorted(list(cont_set))

    fill_vals = np.empty(d, dtype=np.float32)

    for _, idxs, fallback in (
        ("binary", fam_bin, allnan_fill_bin),
        ("nominal", fam_nom, allnan_fill_nom),
        ("ordinal", fam_ord, allnan_fill_nom),
    ):
        for j in idxs:
            w = Xtr[:, j][~np.isnan(Xtr[:, j])]
            if w.size == 0:
                fill_vals[j] = float(fallback)
            else:
                vals, counts = np.unique(w, return_counts=True)
                winners = np.where(counts == counts.max())[0]
                fill_vals[j] = float(vals[winners].min())

    for j in fam_cont:
        w = Xtr[:, j][~np.isnan(Xtr[:, j])]
        if w.size == 0:
            fill_vals[j] = float(allnan_fill_cont)
        else:
            mean_val = float(np.mean(w))
            median_val = float(np.median(w))
            std_val = float(np.std(w)) + 1e-12
            prefer_median = abs(mean_val - median_val) > float(skew_rule) * std_val
            fill_vals[j] = median_val if prefer_median else mean_val

    nan_tr = np.isnan(Xtr)
    nan_te = np.isnan(Xte)
    if nan_tr.any():
        Xtr[nan_tr] = np.take(fill_vals, np.where(nan_tr)[1])
    if nan_te.any():
        Xte[nan_te] = np.take(fill_vals, np.where(nan_te)[1])

    return Xtr, Xte


def one_hot_encoding_selected(
    Xtr, Xte, cat_idx,
    drop_first=True,
    total_cap=None,
    add_other=False,
):
    """One-hot encode selected categorical columns with optional baseline drop and capacity control.

    Args:
        Xtr (np.ndarray): Training matrix.
        Xte (np.ndarray): Test matrix.
        cat_idx (list[int]): Indices of categorical columns to encode.
        drop_first (bool): If True, drop the most frequent level as baseline.
        total_cap (int or None): Global cap on the number of added dummy columns.
        add_other (bool): If True, add a column capturing unseen levels.

    Returns:
        tuple: (Xtr_new, Xte_new, plan, keep_idx, dummy_map)
    """
    Xtr = np.asarray(Xtr)
    Xte = np.asarray(Xte)
    n_tr, d = Xtr.shape
    assert Xte.shape[1] == d

    if total_cap is None:
        total_cap = getattr(config, "MAX_ADDED_ONEHOT", 10000)

    new_tr_cols, new_te_cols = [], []
    used_idx, plan = [], []
    dummy_map = {}
    added_total = 0

    for j in cat_idx:
        col_tr = Xtr[:, j]
        col_te = Xte[:, j]

        valid_tr = ~np.isnan(col_tr)
        if np.any(valid_tr):
            uniq, counts = np.unique(col_tr[valid_tr], return_counts=True)
            order = np.argsort(-counts)
            uniq = uniq[order]
        else:
            uniq = np.array([], dtype=col_tr.dtype)

        kept_all = uniq

        baseline = None
        kept_for_ohe = kept_all
        if drop_first and kept_all.size > 0:
            baseline = float(kept_all[0])
            kept_for_ohe = kept_all[1:]

        values_to_encode = kept_for_ohe.tolist()
        if add_other:
            values_to_encode.append("__OTHER__")
        values_to_encode.append(np.nan)

        k_add = len(values_to_encode)
        if (total_cap is not None) and (added_total + k_add > int(total_cap)):
            continue

        used_idx.append(j)
        plan.append((j, {
            "kept_values": [float(x) for x in kept_all.tolist()],
            "baseline": (None if baseline is None else float(baseline)),
            "has_other": bool(add_other),
        }))

        tr_isnan = np.isnan(col_tr)
        te_isnan = np.isnan(col_te)
        known_all = set(float(x) for x in kept_all.tolist())

        for v in values_to_encode:
            if v == "__OTHER__":
                tr_col = (~tr_isnan) & (~np.isin(col_tr, list(known_all)))
                te_col = (~te_isnan) & (~np.isin(col_te, list(known_all)))
                new_tr_cols.append(tr_col )
                new_te_cols.append(te_col )
            elif (isinstance(v, float) and np.isnan(v)) or (v is np.nan):
                new_tr_cols.append(tr_isnan )
                new_te_cols.append(te_isnan )
            else:
                new_tr_cols.append((col_tr == v) )
                new_te_cols.append((col_te == v) )

        added_total += k_add
        dummy_map[j] = []

    keep_idx = [jj for jj in range(d) if jj not in used_idx]
    Xtr_keep, Xte_keep = Xtr[:, keep_idx], Xte[:, keep_idx]

    if new_tr_cols:
        Xtr_new = np.column_stack([Xtr_keep] + new_tr_cols)
        Xte_new = np.column_stack([Xte_keep] + new_te_cols)

        base = Xtr_keep.shape[1]
        cursor = base
        k_per_feat = []
        for (j, meta) in plan:
            k_feat = len(meta["kept_values"])
            if drop_first and meta["baseline"] is not None and k_feat > 0:
                k_feat -= 1
            if meta["has_other"]:
                k_feat += 1
            k_feat += 1
            k_per_feat.append(k_feat)

        for (j, meta), k in zip(plan, k_per_feat):
            dummy_map[j] = list(range(cursor, cursor + k))
            cursor += k
    else:
        Xtr_new, Xte_new = Xtr_keep, Xte_keep

    print(f"[Preprocess] one-hot:"
          f" kept_raw={len(keep_idx)}"
          f", encoded_feat={len(used_idx)}"
          f", added_cols={sum(len(v) for v in dummy_map.values())}"
          f", drop_first={drop_first}"
          f", total_cap={total_cap}"
          f", other={add_other}")

    return Xtr_new , Xte_new , plan, keep_idx, dummy_map


def remove_highly_correlated_continuous(Xtr, Xte, cont_idx, y_train, threshold=0.90):
    """Prune continuous features with high pairwise correlation, favoring target-aligned and higher-variance columns.

    Args:
        Xtr (np.ndarray): Training matrix.
        Xte (np.ndarray): Test matrix.
        cont_idx (array-like): Indices of continuous features to consider.
        y_train (np.ndarray): Targets used to guide tie-breaking.
        threshold (float): Absolute correlation threshold for pruning.

    Returns:
        tuple: (Xtr_new, Xte_new, dropped_indices, kept_indices)
    """
    Xtr = np.ascontiguousarray(np.asarray(Xtr, dtype=np.float32))
    Xte = np.ascontiguousarray(np.asarray(Xte, dtype=np.float32))
    y = np.asarray(y_train, dtype=np.float32).ravel()
    cont_idx = np.asarray(cont_idx, dtype=int)
    if cont_idx.size <= 1:
        print(f"[Preprocess] corr prune (continuous): nothing to do (n_cont={cont_idx.size}).")
        kept = list(range(Xtr.shape[1]))
        return Xtr, Xte, [], kept
    Xc = Xtr[:, cont_idx]
    if np.isnan(Xc).any():
        raise ValueError("NaNs in continuous block. Impute or standardize before pruning.")
    corr = np.corrcoef(Xc, rowvar=False)
    y_center = y - y.mean()
    Xc_center = Xc - Xc.mean(axis=0, keepdims=True)
    denom = (np.sqrt((Xc_center**2).sum(axis=0)) * np.sqrt((y_center**2).sum()))
    tgt_corr = np.zeros(Xc.shape[1], dtype=np.float32)
    nz = denom > 0
    tgt_corr[nz] = np.abs((Xc_center[:, nz].T @ y_center) / denom[nz])
    variances = Xc.var(axis=0)
    keep_local = np.ones(Xc.shape[1], dtype=bool)
    D = Xc.shape[1]
    for i in range(D):
        if not keep_local[i]:
            continue
        high = np.abs(corr[i, (i + 1):]) >= threshold
        if not high.any():
            continue
        js = np.where(high)[0] + (i + 1)
        for j in js:
            if not keep_local[j]:
                continue
            ti, tj = tgt_corr[i], tgt_corr[j]
            if ti > tj:
                keep_local[j] = False
            elif tj > ti:
                keep_local[i] = False
            else:
                if variances[i] >= variances[j]:
                    keep_local[j] = False
                else:
                    keep_local[i] = False
    cont_keep_idx = cont_idx[keep_local]
    cont_drop_idx = cont_idx[~keep_local]
    keep_global = np.ones(Xtr.shape[1], dtype=bool)
    keep_global[cont_drop_idx] = False
    Xtr_new = Xtr[:, keep_global]
    Xte_new = Xte[:, keep_global]
    print(f"[Preprocess] corr prune (continuous): thr={threshold} → dropped {cont_drop_idx.size} / kept {cont_keep_idx.size} continuous (final D={Xtr_new.shape[1]})")
    return Xtr_new, Xte_new, cont_drop_idx.tolist(), np.where(keep_global)[0].tolist()


def pca_local_on_ohe(Xtr, Xte, dummy_map, cfg=None):
    """Apply PCA independently within each one-hot block to reduce dimensionality.

    PCA is fit on training data per block and applied to the matched columns in test.
    You can replace original dummy columns or append the components.

    Args:
        Xtr (np.ndarray): Training matrix.
        Xte (np.ndarray): Test matrix.
        dummy_map (dict[int, list[int]]): Mapping from source categorical feature to its one-hot column indices.
        cfg (dict or float or int or None): Configuration for variance target, fixed components, minimum block size, and replacement behavior.

    Returns:
        tuple: (Xtr_out, Xte_out, spec) with projection metadata and component indices.
    """
    Xtr = np.asarray(Xtr, np.float32)
    Xte = np.asarray(Xte, np.float32)

    if cfg is None:
        cfg = getattr(config, "PCA_Local", None)
    if cfg is None or cfg is False:
        return Xtr, Xte, {"groups": {}, "total_k": 0}

    if isinstance(cfg, (float, int)):
        cfg = {"variance_ratio": float(cfg)}
    vr = float(np.clip(cfg.get("variance_ratio", 0.90), 0.0, 1.0)) if "n_components" not in cfg else None
    k_fixed = cfg.get("n_components", None)
    min_cols = int(cfg.get("min_cols", 6))
    replace = bool(cfg.get("replace", True))

    groups = {int(j): list(map(int, idxs)) for j, idxs in dummy_map.items() if len(idxs) >= min_cols}
    if not groups:
        return Xtr, Xte, {"groups": {}, "total_k": 0}

    keep_mask = np.ones(Xtr.shape[1], dtype=bool)
    proj_tr_list, proj_te_list = [], []
    spec_groups = {}
    total_k = 0

    for j, cols in groups.items():
        cols = np.array(cols, dtype=int)
        Xtr_blk = Xtr[:, cols]
        Xte_blk = Xte[:, cols]

        mean_tr = np.mean(Xtr_blk, axis=0, dtype=np.float64)
        Xtr_c = (Xtr_blk - mean_tr) 
        Xte_c = (Xte_blk - mean_tr) 

        _, S, Vt = np.linalg.svd(Xtr_c, full_matrices=False)
        n_samples = Xtr_c.shape[0]
        explained_var = (S ** 2) / max(n_samples - 1, 1)
        explained_ratio = explained_var / np.sum(explained_var)

        k_max = Vt.shape[0]
        if k_fixed is not None:
            k = int(min(max(int(k_fixed), 1), k_max))
        else:
            cumsum = np.cumsum(explained_ratio)
            k = int(np.searchsorted(cumsum, vr, side="left") + 1)

        comps = Vt[:k, :].T 
        Ztr = Xtr_c @ comps
        Zte = Xte_c @ comps

        spec_groups[j] = {
            "cols": cols.tolist(),
            "k": int(k),
            "mean": mean_tr ,
            "components": comps,
            "explained_ratio": explained_ratio[:k] ,
        }
        total_k += k

        if replace:
            keep_mask[cols] = False
        proj_tr_list.append(Ztr)
        proj_te_list.append(Zte)

    if replace:
        Xtr_out = np.column_stack([Xtr[:, keep_mask]] + proj_tr_list) if proj_tr_list else Xtr
        Xte_out = np.column_stack([Xte[:, keep_mask]] + proj_te_list) if proj_te_list else Xte

        base = int(keep_mask.sum())
        pca_idx_cursor = base
        for j in groups.keys():
            k = int(spec_groups[j]["k"])
            spec_groups[j]["pca_component_idx"] = list(range(pca_idx_cursor, pca_idx_cursor + k))
            pca_idx_cursor += k
    else:
        Xtr_out = np.column_stack([Xtr] + proj_tr_list) if proj_tr_list else Xtr
        Xte_out = np.column_stack([Xte] + proj_te_list) if proj_te_list else Xte

        base = Xtr.shape[1]
        pca_idx_cursor = base
        for j in groups.keys():
            k = int(spec_groups[j]["k"])
            spec_groups[j]["pca_component_idx"] = list(range(pca_idx_cursor, pca_idx_cursor + k))
            pca_idx_cursor += k

    print(f"[Preprocess] PCA-Local on OHE: groups={len(groups)} total_k={total_k} replace={replace}")
    return Xtr_out , Xte_out , {"groups": spec_groups, "total_k": int(total_k)}


def standardize(Xtr_new, Xte_new, cont_idx=None, return_updated_idx=False):
    """Standardize features using training statistics and drop zero-variance columns.

    Modes:
      - config.STD_CONT == False: standardize all columns globally and drop any zero-variance columns.
      - config.STD_CONT == True & cont_idx is None: fallback to global standardization (same as above).
      - config.STD_CONT == True & cont_idx set: standardize only those columns and drop zero-variance ones within that subset.
    """

    Xtr_new = np.asarray(Xtr_new, dtype=np.float32)
    Xte_new = np.asarray(Xte_new, dtype=np.float32)
    std_cont = bool(getattr(config, "STD_CONT", False))
    use_subset = std_cont and (cont_idx is not None) and (len(cont_idx) > 0)

    # ---------- GLOBAL STANDARDIZATION ----------
    if not use_subset:
        mean_tr = np.mean(Xtr_new, axis=0) 
        std_tr = np.std(Xtr_new, axis=0) 

        zero_var_mask = std_tr == 0
        std_safe = std_tr.copy()
        std_safe[zero_var_mask] = 1.0

        Xtr_s = (Xtr_new - mean_tr) / std_safe
        Xte_s = (Xte_new - mean_tr) / std_safe

        if np.any(zero_var_mask):
            keep_mask = ~zero_var_mask
            dropped = int(np.sum(zero_var_mask))
            Xtr_s = Xtr_s[:, keep_mask]
            Xte_s = Xte_s[:, keep_mask]
            print(f"[Standardize] mode=global | dropped {dropped} zero-variance columns | Xtr,Xte: {Xtr_s.shape} {Xte_s.shape}")
        else:
            print(f"[Standardize] mode=global | no zero-variance | Xtr,Xte: {Xtr_s.shape} {Xte_s.shape}")

        return (
            (Xtr_s , Xte_s )
            if not return_updated_idx
            else (Xtr_s, Xte_s, np.array(cont_idx if cont_idx is not None else [], dtype=int))
        )

    # ---------- SUBSET STANDARDIZATION ----------
    cont_idx = np.asarray(cont_idx, dtype=int)
    Xtr_s = Xtr_new.copy()
    Xte_s = Xte_new.copy()
    mu = np.mean(Xtr_new[:, cont_idx], axis=0) 
    sd = np.std(Xtr_new[:, cont_idx], axis=0) 

    zero_var_mask_local = sd == 0
    sd_safe = sd.copy()
    sd_safe[sd_safe == 0] = 1.0

    Xtr_s[:, cont_idx] = (Xtr_new[:, cont_idx] - mu) / sd_safe
    Xte_s[:, cont_idx] = (Xte_new[:, cont_idx] - mu) / sd_safe

    if np.any(zero_var_mask_local):
        to_drop_global = cont_idx[zero_var_mask_local]
        keep_mask_global = np.ones(Xtr_s.shape[1], dtype=bool)
        keep_mask_global[to_drop_global] = False
        Xtr_s = Xtr_s[:, keep_mask_global]
        Xte_s = Xte_s[:, keep_mask_global]

        new_positions = np.nonzero(keep_mask_global)[0]
        cont_idx_kept_old = cont_idx[~zero_var_mask_local]
        pos_map = {old: new for new, old in enumerate(new_positions)}
        cont_idx_kept_new = np.array([pos_map[i] for i in cont_idx_kept_old], dtype=int)

        print(f"[Standardize] mode=subset | dropped {to_drop_global.size} zero-variance continuous cols | Xtr,Xte: {Xtr_s.shape} {Xte_s.shape}")
        if return_updated_idx:
            return Xtr_s, Xte_s, cont_idx_kept_new
        return Xtr_s, Xte_s

    print(f"[Standardize] mode=subset | n_cols={cont_idx.size} | Xtr,Xte: {Xtr_s.shape} {Xte_s.shape}")
    if return_updated_idx:
        return Xtr_s, Xte_s, cont_idx
    return Xtr_s, Xte_s




def pca(Xtr, Xte, cols=None, n_components=None, variance_ratio=None, replace=True):
    """Fit PCA on a selected block of columns in the training set and project both splits.

    Args:
        Xtr (np.ndarray): Training matrix.
        Xte (np.ndarray): Test matrix.
        cols (array-like or None): Indices to project; defaults to all columns.
        n_components (int or None): Fixed number of components to keep.
        variance_ratio (float or None): Minimum explained variance ratio to retain.
        replace (bool): If True, replace the original block; otherwise, append components.

    Returns:
        tuple: (Xtr_out, Xte_out, spec) with PCA metadata and resulting indices.
    """
    Xtr = np.asarray(Xtr, np.float32)
    Xte = np.asarray(Xte, np.float32)
    if cols is None:
        cols = np.arange(Xtr.shape[1])
    else:
        cols = np.array(cols, dtype=int)
    Xtr_blk = Xtr[:, cols]
    Xte_blk = Xte[:, cols]
    mean_tr = np.mean(Xtr_blk, axis=0, dtype=np.float64)
    Xtr_c = (Xtr_blk - mean_tr) 
    Xte_c = (Xte_blk - mean_tr) 
    _, S, Vt = np.linalg.svd(Xtr_c, full_matrices=False)
    n_samples = Xtr_c.shape[0]
    explained_var = (S ** 2) / max(n_samples - 1, 1)
    explained_ratio = explained_var / np.sum(explained_var)
    k_max = Vt.shape[0]
    k = k_max
    cfg_k = getattr(config, "PCA_K", None)
    if variance_ratio is not None:
        vr = float(np.clip(variance_ratio, 0.0, 1.0))
        cumsum = np.cumsum(explained_ratio)
        k = int(np.searchsorted(cumsum, vr, side="left") + 1)
    elif n_components is not None:
        k = int(min(max(n_components, 1), k_max))
    elif cfg_k is not None:
        if isinstance(cfg_k, float) and 0 < cfg_k <= 1.0:
            cumsum = np.cumsum(explained_ratio)
            k = int(np.searchsorted(cumsum, cfg_k, side="left") + 1)
        else:
            k = int(min(max(cfg_k, 1), k_max))
    components = Vt[:k, :].T 
    Xtr_proj = Xtr_c @ components
    Xte_proj = Xte_c @ components
    explained = float(np.sum(explained_ratio[:k])) * 100
    print(f"[Preprocess] PCA: block d={Xtr_blk.shape[1]} → k={k} comps ({explained:.2f}% variance), replace={replace}")
    if replace:
        keep_idx = [j for j in range(Xtr.shape[1]) if j not in set(cols)]
        Xtr_out = np.column_stack([Xtr[:, keep_idx], Xtr_proj])
        Xte_out = np.column_stack([Xte[:, keep_idx], Xte_proj])
        base = len(keep_idx)
        pca_component_idx = list(range(base, base + Xtr_proj.shape[1]))
    else:
        Xtr_out = np.column_stack([Xtr, Xtr_proj])
        Xte_out = np.column_stack([Xte, Xte_proj])
        base = Xtr.shape[1]
        pca_component_idx = list(range(base, base + Xtr_proj.shape[1]))
    spec = {
        "cols": np.array(cols, int).tolist(),
        "mean": mean_tr ,
        "components": components,
        "explained_ratio": explained_ratio[:k] ,
        "k": k,
        "replace": bool(replace),
        "pca_component_idx": pca_component_idx,
    }
    return Xtr_out , Xte_out , spec


def remap_indices_after_mask(old_size, keep_mask, idx_list):
    """Remap original column indices to new positions after applying a boolean column mask.

    Args:
        old_size (int): Original number of columns.
        keep_mask (array-like): Boolean mask over original columns.
        idx_list (array-like): Column indices to remap.

    Returns:
        list[int]: Remapped indices that remain after masking.
    """
    keep_mask = np.asarray(keep_mask, bool)
    old_to_new = -np.ones(old_size, dtype=int)
    old_to_new[np.where(keep_mask)[0]] = np.arange(keep_mask.sum())
    idx_old = np.asarray(idx_list, int)
    idx_new = old_to_new[idx_old]
    return idx_new[idx_new >= 0].tolist()


def compute_sample_weights(y_train):
    """Compute per-sample weights that balance classes by inverse frequency.

    Args:
        y_train (array-like): Class labels.

    Returns:
        np.ndarray: Sample weights aligned with the training labels.
    """
    y_train = np.asarray(y_train)
    n_samples = len(y_train)
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    n_classes = len(unique_classes)
    class_weights = n_samples / (n_classes * class_counts)
    class_to_weight = dict(zip(unique_classes, class_weights))
    sample_weights = np.array([class_to_weight[label] for label in y_train], dtype=np.float32)
    print(f"[Class Weighting] Dataset imbalance detected:")
    for cls, count in zip(unique_classes, class_counts):
        pct = 100 * count / n_samples
        weight = class_to_weight[cls]
        print(f"  Class {cls:+d}: {count:6d} samples ({pct:5.2f}%) -> weight: {weight:.4f}")
    return sample_weights


def save(Xtr, Xte, ytr, train_ids, test_ids, filename):
    """Save arrays to a compressed NPZ archive.

    Args:
        Xtr (np.ndarray): Training matrix.
        Xte (np.ndarray): Test matrix.
        ytr (np.ndarray): Training labels.
        train_ids (np.ndarray): Training identifiers.
        test_ids (np.ndarray): Test identifiers.
        filename (str): Output file path.

    Returns:
        None
    """
    np.savez_compressed(filename, X_train=Xtr, X_test=Xte, y_train=ytr, train_ids=train_ids, test_ids=test_ids)


def poly_expand_train(Xtr, cont_mask):
    """Create polynomial features on training data.

    Adds squares of continuous features and the top interaction terms ranked by correlation,
    subject to a budget from the configuration.

    Args:
        Xtr (np.ndarray): Training matrix.
        cont_mask (array-like): Boolean mask for continuous columns.

    Returns:
        tuple: (X_aug, spec) where X_aug includes added features and spec records the recipe.
    """
    cont_idx = np.where(cont_mask)[0]
    spec = {"cont_idx": cont_idx.tolist(), "pairs": [], "n_added": 0, "add_squares": bool(config.POLY_ADD_SQUARES)}
    if cont_idx.size == 0:
        return Xtr, spec
    Xc = Xtr[:, cont_idx].astype(np.float32, copy=False)
    added = []
    if config.POLY_ADD_SQUARES:
        added.append(Xc ** 2)
    n_sq = (Xc.shape[1] if config.POLY_ADD_SQUARES else 0)
    if config.POLY_ADD_INTERACTIONS and Xc.shape[1] > 1:
        budget = int(config.POLY_MAX_ADDED_FEATURES)
        k = max(0, budget - n_sq)
        if k > 0:
            d = Xc.shape[1]
            C = np.nan_to_num(np.corrcoef(Xc, rowvar=False), nan=0.0)
            triples = [(i, j, abs(C[i, j])) for i in range(d) for j in range(i + 1, d)]
            triples.sort(key=lambda t: t[2], reverse=True)
            pairs = [(i, j) for i, j, _ in triples[:k]]
            spec["pairs"] = pairs
            if pairs:
                Z = np.empty((Xc.shape[0], len(pairs)), dtype=Xtr.dtype)
                for idx, (i, j) in enumerate(pairs):
                    Z[:, idx] = Xc[:, i] * Xc[:, j]
                added.append(Z)
    if added:
        Zall = np.concatenate(added, axis=1)
        cap = int(getattr(config, "POLY_MAX_ADDED_FEATURES", Zall.shape[1]))
        if Zall.shape[1] > cap:
            Zall = Zall[:, :cap]
        spec["n_added"] = Zall.shape[1]
        X_aug = np.concatenate([Xtr, Zall], axis=1)
        return X_aug, spec
    return Xtr, spec


def poly_expand_apply(X, spec):
    """Apply a polynomial expansion recipe to a new matrix.

    Args:
        X (np.ndarray): Input matrix.
        spec (dict): Recipe returned by poly_expand_train.

    Returns:
        np.ndarray: Transformed matrix with added features.
    """
    n_added = int(spec.get("n_added", 0))
    if n_added == 0:
        return X
    cont_idx = np.array(spec.get("cont_idx", []), dtype=int)
    if cont_idx.size == 0:
        return X
    Xc = X[:, cont_idx]
    added = []
    if spec.get("add_squares", True):
        added.append(Xc ** 2)
    pairs = spec.get("pairs", [])
    if pairs:
        Z = np.empty((Xc.shape[0], len(pairs)), dtype=X.dtype)
        for idx, (i, j) in enumerate(pairs):
            Z[:, idx] = Xc[:, i] * Xc[:, j]
        added.append(Z)
    if added:
        Zall = np.concatenate(added, axis=1)[:, :n_added]
        return np.concatenate([X, Zall], axis=1)
    return X


def add_predictive_nan_indicators(
    Xtr, Xte, y_train,
    threshold=0.01, top_k=128,
    min_prevalence=0.005, max_prevalence=0.995
):
    """Append missingness-indicator features selected by target correlation, after dropping columns with test-side NaN rate ≥ 0.30."""
    Xtr = np.asarray(Xtr, np.float32)
    Xte = np.asarray(Xte, np.float32)
    y = np.asarray(y_train, np.float32).ravel()

    # ---- 1) ----
    test_nan_rate = np.isnan(Xte).mean(axis=0) if Xte.size else np.array([], dtype=np.float32)
    keep_cols = test_nan_rate < 0.30 if test_nan_rate.size else np.array([], dtype=bool)
    if test_nan_rate.size and not np.all(keep_cols):
        n_drop = int((~keep_cols).sum())
        n_keep = int(keep_cols.sum())
        print(f"[Preprocess] NaN-based feature filter (test): dropped={n_drop}, kept={n_keep}, thr=0.30")
        if n_keep == 0:
            return Xtr[:, :0] , Xte[:, :0] 
        Xtr = Xtr[:, keep_cols]
        Xte = Xte[:, keep_cols]
    elif test_nan_rate.size:
        print(f"[Preprocess] NaN-based feature filter (test): dropped=0, kept={Xtr.shape[1]}, thr=0.30")

    # ---- 2) ----
    Mtr = np.isnan(Xtr) 
    Mte = np.isnan(Xte) 
    if Mtr.size == 0:
        print("[Preprocess] NaN indicators: no features after test-NaN filter")
        return Xtr, Xte
    prev = Mtr.mean(axis=0)
    keep_prev = (prev >= float(min_prevalence)) & (prev <= float(max_prevalence))
    if not np.any(keep_prev):
        print("[Preprocess] NaN indicators: none selected (prevalence filter)")
        return Xtr, Xte

    # ---- 3) ----
    yz = (y - y.mean()) / (y.std() + 1e-12)
    mu = Mtr[:, keep_prev].mean(axis=0)
    sd = Mtr[:, keep_prev].std(axis=0)
    sd[sd == 0] = 1.0
    Z = (Mtr[:, keep_prev] - mu) / sd
    corrs = (Z.T @ yz) / (Xtr.shape[0] - 1)
    scores = np.abs(np.nan_to_num(corrs, 0.0))
    if threshold is not None:
        mask_thr = scores > float(threshold)
        if not np.any(mask_thr):
            print("[Preprocess] NaN indicators: none above threshold")
            return Xtr, Xte
        cand_local = np.where(mask_thr)[0]
    else:
        cand_local = np.arange(scores.size)
    if top_k is not None and cand_local.size > int(top_k):
        order = np.argsort(-scores[cand_local])[:int(top_k)]
        cand_local = cand_local[order]

    # ---- 4) ----
    cand_global = np.where(keep_prev)[0][cand_local]
    if cand_global.size == 0:
        print("[Preprocess] NaN indicators: none selected")
        return Xtr, Xte
    Xtr_aug = np.hstack([Xtr, Mtr[:, cand_global]])
    Xte_aug = np.hstack([Xte, Mte[:, cand_global]])
    print(f"[Preprocess] NaN indicators: add {cand_global.size}/{Xtr.shape[1]} (thr={threshold}, top_k={top_k}, prev∈[{min_prevalence},{max_prevalence}])")
    return Xtr_aug , Xte_aug 

def _listify_idx(idx, d):
    """Return a NumPy array of indices. If None, return all column indices.

    Args:
        idx (array-like or None): Indices or None.
        d (int): Number of columns.

    Returns:
        np.ndarray: Indices as integers.
    """
    if idx is None:
        return np.arange(d, dtype=int)
    return np.array(idx, dtype=int)


def poly_expand_train_v2(
    Xtr,
    y_train,
    cont_idx=None,
    pca_idx=None,
    add_squares_cont=True,
    add_squares_pca=False,
    add_inter_within_cont=True,
    add_inter_within_pca=False,
    add_inter_cross_cont_pca=True,
    top_k_pairs=256,
    min_abs_corr=0.0,
):
    """Select and add polynomial features using a target-correlation score.

    Supports squares within specified index groups and interaction terms within or across groups.
    Keeps the most predictive interactions up to a configurable budget.

    Args:
        Xtr (np.ndarray): Training matrix.
        y_train (np.ndarray): Target vector used for scoring.
        cont_idx (array-like or None): Indices considered as continuous.
        pca_idx (array-like or None): Indices considered as PCA-derived.
        add_squares_cont (bool): Add squares for continuous indices.
        add_squares_pca (bool): Add squares for PCA indices.
        add_inter_within_cont (bool): Add interactions within continuous indices.
        add_inter_within_pca (bool): Add interactions within PCA indices.
        add_inter_cross_cont_pca (bool): Add interactions across the two index groups.
        top_k_pairs (int): Maximum number of interaction terms to keep.
        min_abs_corr (float): Minimum absolute correlation score for any added feature.

    Returns:
        tuple: (Xtr_aug, spec) with the added features and the specification used.
    """
    Xtr = np.asarray(Xtr, np.float32)
    y = np.asarray(y_train, np.float32).ravel()
    d = Xtr.shape[1]
    cont_idx = _listify_idx(cont_idx, d)
    pca_idx = _listify_idx(pca_idx, d)
    squares = []
    if add_squares_cont and cont_idx.size:
        squares += [("square", int(i)) for i in cont_idx]
    if add_squares_pca and pca_idx.size:
        squares += [("square", int(i)) for i in pca_idx]
    pairs = []
    if add_inter_within_cont and cont_idx.size > 1:
        ci = cont_idx
        for a in range(ci.size):
            for b in range(a + 1, ci.size):
                pairs.append(("prod", int(ci[a]), int(ci[b])))
    if add_inter_within_pca and pca_idx.size > 1:
        pi = pca_idx
        for a in range(pi.size):
            for b in range(a + 1, pi.size):
                pairs.append(("prod", int(pi[a]), int(pi[b])))
    if add_inter_cross_cont_pca and cont_idx.size and pca_idx.size:
        for i in cont_idx:
            for j in pca_idx:
                pairs.append(("prod", int(i), int(j)))
    if not squares and not pairs:
        return Xtr, {"squares": [], "pairs": [], "n_added": 0}
    added_blocks = []
    meta_blocks = []
    if squares:
        cols = np.array([t[1] for t in squares], dtype=int)
        Z = (Xtr[:, cols] ** 2) 
        added_blocks.append(Z)
        meta_blocks += [("square", int(k)) for k in cols]

    def score_block(Z):
        yz = (y - y.mean()) / (y.std() + 1e-12)
        mu = Z.mean(axis=0)
        sd = Z.std(axis=0)
        sd[sd == 0] = 1.0
        Zs = (Z - mu) / sd
        corrs = (Zs.T @ yz) / (Xtr.shape[0] - 1)
        return np.abs(np.nan_to_num(corrs, 0.0))

    pair_chunks = []
    chunk_meta = []
    CHUNK = max(1, 16384 // max(1, Xtr.shape[0]))
    if pairs:
        cur = 0
        while cur < len(pairs):
            end = min(len(pairs), cur + CHUNK)
            P = pairs[cur:end]
            Z = np.empty((Xtr.shape[0], len(P)), dtype=np.float32)
            for k, tpl in enumerate(P):
                _, i, j = tpl
                Z[:, k] = Xtr[:, i] * Xtr[:, j]
            sc = score_block(Z)
            pair_chunks.append((Z, sc))
            chunk_meta.append(P)
            cur = end
    scores_sq = np.array([], dtype=np.float32)
    if squares:
        scores_sq = score_block(added_blocks[0])
    scores_pairs = np.array([], dtype=np.float32)
    if pairs:
        scores_pairs = np.concatenate([sc for (_, sc) in pair_chunks], axis=0)
    keep_sq = np.ones_like(scores_sq, dtype=bool)
    if scores_sq.size and min_abs_corr is not None:
        keep_sq = scores_sq >= float(min_abs_corr)
    keep_pairs = np.ones_like(scores_pairs, dtype=bool)
    if scores_pairs.size and min_abs_corr is not None:
        keep_pairs = scores_pairs >= float(min_abs_corr)
    order_sq = np.argsort(-(scores_sq[keep_sq])) if scores_sq.size else np.array([], dtype=int)
    order_pairs = np.argsort(-(scores_pairs[keep_pairs])) if scores_pairs.size else np.array([], dtype=int)
    if scores_pairs.size:
        idx_pairs_local = np.where(keep_pairs)[0][order_pairs]
        if top_k_pairs is not None:
            idx_pairs_local = idx_pairs_local[:int(top_k_pairs)]
    else:
        idx_pairs_local = np.array([], dtype=int)
    added_list = []
    meta_added = []
    if scores_sq.size:
        idx_sq_keep = np.where(keep_sq)[0][order_sq]
        if idx_sq_keep.size:
            Z_sq = added_blocks[0][:, idx_sq_keep]
            added_list.append(Z_sq)
            meta_added += [meta_blocks[k] for k in idx_sq_keep.tolist()]
    if idx_pairs_local.size:
        Z_pairs_keep = []
        pairs_keep_meta = []
        base = 0
        for (Zchunk, _), Pmeta in zip(pair_chunks, chunk_meta):
            nloc = Zchunk.shape[1]
            loc_range = np.arange(base, base + nloc)
            m = np.intersect1d(idx_pairs_local, loc_range, assume_unique=False)
            if m.size:
                take = m - base
                Z_pairs_keep.append(Zchunk[:, take])
                pairs_keep_meta += [Pmeta[t] for t in take.tolist()]
            base += nloc
        if Z_pairs_keep:
            added_list.append(np.column_stack(Z_pairs_keep))
            meta_added += pairs_keep_meta
    if not added_list:
        return Xtr, {"squares": [], "pairs": [], "n_added": 0}
    Zall = np.column_stack(added_list) 
    Xtr_aug = np.column_stack([Xtr, Zall])
    spec = {
        "squares": [t[1] for t in meta_added if t[0] == "square"],
        "pairs": [(t[1], t[2]) for t in meta_added if t[0] == "prod"],
        "n_added": int(Zall.shape[1]),
    }
    print(f"[Preprocess] poly v2: added {spec['n_added']} (squares={len(spec['squares'])}, pairs={len(spec['pairs'])})")
    return Xtr_aug, spec


def poly_expand_apply_v2(X, spec):
    """Apply a v2 polynomial feature recipe to any matrix.

    Args:
        X (np.ndarray): Input matrix.
        spec (dict): Recipe returned by poly_expand_train_v2.

    Returns:
        np.ndarray: Transformed matrix with added features.
    """
    X = np.asarray(X, np.float32)
    added = []
    for i in spec.get("squares", []):
        added.append((X[:, int(i)] ** 2) .reshape(-1, 1))
    for (i, j) in spec.get("pairs", []):
        added.append((X[:, int(i)] * X[:, int(j)]) .reshape(-1, 1))
    if added:
        Z = np.column_stack(added)
        return np.column_stack([X, Z]) 
    return X


def encode_ordinal_as_score(Xtr, Xte, ord_idx, scale_to_unit=True):
    """Encode ordinal features as monotone scores learned on the training set.

    Levels observed in training are mapped to increasing ranks. Unseen levels in test are mapped
    to the maximum rank. Missing values are preserved. Scores can optionally be scaled to the unit interval.

    Args:
        Xtr (np.ndarray): Training matrix.
        Xte (np.ndarray): Test matrix.
        ord_idx (array-like): Indices of ordinal features.
        scale_to_unit (bool): If True, divide ranks by the maximum rank per feature.

    Returns:
        tuple: (Xtr_new, Xte_new, spec) with per-column mapping metadata.
    """
    Xtr = np.asarray(Xtr, np.float32).copy()
    Xte = np.asarray(Xte, np.float32).copy()
    ordinal_maps = {}

    for j in ord_idx:
        vtr = Xtr[:, j]
        cats_sorted = np.unique(vtr[~np.isnan(vtr)])
        if cats_sorted.size == 0:
            ordinal_maps[j] = {"levels": [], "K": 1, "scaled": bool(scale_to_unit)}
            continue

        K = max(len(cats_sorted) - 1, 1)

        ranks_tr = np.full(vtr.shape, np.nan, dtype=np.float32)
        valid_tr = ~np.isnan(vtr)
        if valid_tr.any():
            idx_tr = np.searchsorted(cats_sorted, vtr[valid_tr], side="left")
            in_bounds_tr = idx_tr < cats_sorted.size
            match_tr = np.zeros_like(idx_tr, dtype=bool)
            if in_bounds_tr.any():
                match_tr[in_bounds_tr] = (cats_sorted[idx_tr[in_bounds_tr]] == vtr[valid_tr][in_bounds_tr])
            ranks_tr[valid_tr] = np.where(match_tr, idx_tr, K) 

        vte = Xte[:, j]
        ranks_te = np.full(vte.shape, np.nan, dtype=np.float32)
        valid_te = ~np.isnan(vte)
        if valid_te.any():
            idx_te = np.searchsorted(cats_sorted, vte[valid_te], side="left")
            in_bounds_te = idx_te < cats_sorted.size
            match_te = np.zeros_like(idx_te, dtype=bool)
            if in_bounds_te.any():
                match_te[in_bounds_te] = (cats_sorted[idx_te[in_bounds_te]] == vte[valid_te][in_bounds_te])
            ranks_te[valid_te] = np.where(match_te, idx_te, K) 

        if scale_to_unit and K > 0:
            ranks_tr = ranks_tr / K
            ranks_te = ranks_te / K

        Xtr[:, j] = ranks_tr
        Xte[:, j] = ranks_te

        ordinal_maps[j] = {
            "levels": cats_sorted.tolist(),
            "K": int(K),
            "scaled": bool(scale_to_unit),
        }

    return Xtr, Xte, {"ordinal_maps": ordinal_maps}
