from preprocessing import *
import config
import numpy as np


def preprocess2(Xtr_raw, Xte_raw, ytr_pm1, train_ids, test_ids, filename):
    """
    Processing pipeline.

    Performs feature cleaning, encoding, PCA reduction, imputation, feature generation,
    standardization, and optional correlation pruning. Saves the processed outputs if a path is provided.

    Args:
        Xtr_raw: Raw training feature matrix.
        Xte_raw: Raw test feature matrix.
        ytr_pm1: Training labels in {-1, 1} format.
        train_ids: Array of training sample identifiers.
        test_ids: Array of test sample identifiers.
        filename: Path to save the processed dataset (.npz).

    Returns:
        Tuple (Xtr, Xte, ytr_01) where:
            Xtr: Processed training features.
            Xte: Processed test features.
            ytr_01: Labels converted to {0, 1} format.
    """
    Xtr = np.asarray(Xtr_raw, np.float32)
    Xte = np.asarray(Xte_raw, np.float32)
    ytr_01 = np.where(ytr_pm1 == -1, 0, ytr_pm1).astype(int)
    print(f"[Pipeline] Start | Xtr={Xtr.shape} Xte={Xte.shape} y={ytr_01.shape}")

    drop_n = getattr(config, "DROP_FIRST_N_CAT_COLS", 26)

    if drop_n > 0:
        keep_col1 = Xtr[:, [1]] # Sate can influence health
        Xtr = Xtr[:, drop_n:]
        Xte = Xte[:, drop_n:]
        Xtr = np.hstack([keep_col1, Xtr])
        Xte = np.hstack([keep_col1[:Xte.shape[0]], Xte])
        print(f"[Step0] Dropped useless cols (col 1 kept) | Xtr={Xtr.shape} Xte={Xte.shape}")

    Xtr = replace_brfss_special_codes(Xtr)
    Xte = replace_brfss_special_codes(Xte)
    print(f"[Step1] BRFSS cleaned | NaN train={np.isnan(Xtr).sum()} NaN test={np.isnan(Xte).sum()}")

    types0 = infer_feature_types(Xtr, max_unique_cat=getattr(config, "LOW_CARD_MAX_UNIQUE", 20))
    ord_idx = np.array(types0["ordinal"], dtype=int)
    if ord_idx.size > 0 and config.ORDINAL_ENCODE:
        Xtr, Xte, _ = encode_ordinal_as_score(
            Xtr, Xte, ord_idx=ord_idx, scale_to_unit=config.ORDINAL_SCALE_TO_UNIT
        )
        print(f"[Ordinals] encoded {len(ord_idx)} columns as monotone scores")

    types2 = infer_feature_types(Xtr, max_unique_cat=getattr(config, "LOW_CARD_MAX_UNIQUE", 20))
    cont_idx_std = np.array(types2["continuous"], dtype=int)

    types0 = infer_feature_types(Xtr, max_unique_cat=getattr(config, "LOW_CARD_MAX_UNIQUE", 15))
    cat_nom_idx = types0["nominal"]
    drop_first = bool(getattr(config, "ONEHOT_DROP_FIRST", True))
    total_cap = getattr(config, "MAX_ADDED_ONEHOT", 10000)
    print(f"[Step2] OHE | #nominal={len(cat_nom_idx)} drop_first={drop_first} total_cap={total_cap}")

    Xtr_ohe, Xte_ohe, plan_ohe, keep_idx, dummy_map = one_hot_encoding_selected(
        Xtr, Xte, cat_nom_idx, drop_first=drop_first, total_cap=total_cap, add_other=False
    )
    ohe_start = len(keep_idx)
    ohe_end = Xtr_ohe.shape[1]
    ohe_cols = np.arange(ohe_start, ohe_end, dtype=int)
    Xtr, Xte = Xtr_ohe, Xte_ohe
    print(f"[Step2] OHE done | Xtr={Xtr.shape} Xte={Xte.shape}")

    _keep_idx = np.asarray(keep_idx, dtype=int)
    pos_map = {int(col): i for i, col in enumerate(_keep_idx)}
    cont_idx_std = np.array([pos_map[int(c)] for c in cont_idx_std if int(c) in pos_map], dtype=int)

    if getattr(config, "PCA_Local", None) is not None:
        Xtr_ohe, Xte_ohe, pca_local_spec = pca_local_on_ohe(
            Xtr, Xte, dummy_map, cfg=getattr(config, "PCA_Local", None)
        )
        Xtr, Xte = Xtr_ohe, Xte_ohe
        pca_idx_list = []
        for meta in pca_local_spec.get("groups", {}).values():
            pca_idx_list.extend(meta.get("pca_component_idx", []))
        pca_spec = {"k": int(pca_local_spec.get("total_k", 0)), "pca_component_idx": pca_idx_list}
        pca_idx = np.array(pca_spec.get("pca_component_idx", []), dtype=int)
        print(f"[Step3] PCA-Local on OHE | Xtr={Xtr.shape} Xte={Xte.shape} | total_k={pca_spec['k']}")
    else:
        pca_var = getattr(config, "PCA_VAR", 0.95)
        pca_k = getattr(config, "PCA_K", None)
        if ohe_cols.size > 0:
            Xtr, Xte, pca_spec = pca(
                Xtr, Xte,
                cols=ohe_cols,
                n_components=(None if isinstance(pca_k, float) else pca_k),
                variance_ratio=(pca_var if isinstance(pca_var, float) else None),
                replace=True
            )
            pca_idx = np.array(pca_spec.get("pca_component_idx", []), dtype=int)
        else:
            pca_spec = {"k": 0, "pca_component_idx": []}
            pca_idx = np.array([], dtype=int)
        print(f"[Step3] PCA on OHE | Xtr={Xtr.shape} Xte={Xte.shape} | k={pca_spec.get('k', 0)}")

    n_before = Xtr.shape[1]
    Xtr, Xte = add_predictive_nan_indicators(
        Xtr, Xte, ytr_01,
        threshold=getattr(config, "NAN_INDICATOR_MIN_ABS_CORR", 0.01),
        top_k=getattr(config, "NAN_INDICATOR_TOPK", 128),
        min_prevalence=getattr(config, "NAN_INDICATOR_MIN_PREV", 0.005),
        max_prevalence=getattr(config, "NAN_INDICATOR_MAX_PREV", 0.995)
    )
    print(f"[Step4] NaN indicators | +{Xtr.shape[1] - n_before} cols | Xtr={Xtr.shape} Xte={Xte.shape}")

    nan_tr_before = np.isnan(Xtr).sum()
    Xtr, Xte = smart_impute(
        Xtr, Xte,
        skew_rule=getattr(config, "IMPUTE_SKEW_RULE", 0.5),
        allnan_fill_cont=getattr(config, "IMPUTE_CONT_ALLNAN_FILL", 0.0),
        allnan_fill_nom=getattr(config, "IMPUTE_NOM_ALLNAN_FILL", -1.0),
        allnan_fill_bin=getattr(config, "IMPUTE_BIN_ALLNAN_FILL", 0.0)
    )
    nan_tr_after = np.isnan(Xtr).sum()
    print(f"[Step5] Impute | NaN train: {nan_tr_before} -> {nan_tr_after}")
    assert nan_tr_after == np.isnan(Xtr).sum(), "NaN count changed after imputation!"
    assert np.isnan(Xtr).sum() == 0, "There are still NaNs in Xtr after imputation!"
    assert np.isnan(Xte).sum() == 0, "There are still NaNs in Xte after imputation!"
    if bool(getattr(config, "POLY_ENABLE_V2", False)):
        types1 = infer_feature_types(Xtr, max_unique_cat=getattr(config, "LOW_CARD_MAX_UNIQUE", 15))
        cont_idx_base = np.array(types1["continuous"], dtype=int)
        if pca_spec.get("k", 0) > 0:
            pca_idx = np.array(pca_spec.get("pca_component_idx", []), dtype=int)
        else:
            pca_idx = np.array([], dtype=int)
        Xtr, poly_spec = poly_expand_train_v2(
            Xtr, ytr_01,
            cont_idx=cont_idx_base,
            pca_idx=pca_idx,
            add_squares_cont=bool(getattr(config, "POLY_ADD_SQUARES_CONT", True)),
            add_squares_pca=bool(getattr(config, "POLY_ADD_SQUARES_PCA", False)),
            add_inter_within_cont=bool(getattr(config, "POLY_ADD_INTER_CONT", True)),
            add_inter_within_pca=bool(getattr(config, "POLY_ADD_INTER_PCA", False)),
            add_inter_cross_cont_pca=bool(getattr(config, "POLY_ADD_INTER_CROSS", True)),
            top_k_pairs=getattr(config, "POLY_TOPK_PAIRS", 256),
            min_abs_corr=getattr(config, "POLY_MIN_ABS_CORR", 0.0),
        )
        Xte = poly_expand_apply_v2(Xte, poly_spec)
        print(f"[Step6] Poly v2 | added={poly_spec.get('n_added', 0)} | Xtr={Xtr.shape} Xte={Xte.shape}")


    Xtr, Xte, cont_idx_std = standardize(Xtr, Xte, cont_idx=cont_idx_std, return_updated_idx=True)
    print(f"[Step7] Standardize | Xtr={Xtr.shape} Xte={Xte.shape}")
    assert np.isnan(Xtr).sum() == 0, "There are still NaNs in Xtr after standardization!"
    corr_thr = getattr(config, "PRUNE_CORR_THRESHOLD", None)
    if corr_thr is not None and cont_idx_std.size > 0:
        Xtr, Xte, dropped_g, kept_g = remove_highly_correlated_continuous(
            Xtr, Xte, cont_idx=cont_idx_std, y_train=ytr_01, threshold=float(corr_thr)
        )
        kept_g = np.asarray(kept_g, dtype=int)
        print(f"[Step8] Corr prune | thr={corr_thr} | dropped={len(dropped_g)}")
    else:
        print("[Step8] Corr prune | skipped")

    if bool(getattr(config, "ADD_BIAS", False)):
        Xtr = np.hstack([np.ones((Xtr.shape[0], 1), dtype=np.float32), Xtr])
        Xte = np.hstack([np.ones((Xte.shape[0], 1), dtype=np.float32), Xte])
        print(f"[Step9] Bias added | Xtr={Xtr.shape} Xte={Xte.shape}")

    if filename:
        save(Xtr, Xte, ytr_01, train_ids, test_ids, filename)
        print(f"[Step10] Saved -> {filename}")

    print(f"[Pipeline] Done | Xtr={Xtr.shape} Xte={Xte.shape}")
    return Xtr.astype(np.float32), Xte.astype(np.float32), ytr_01
