# Preprocessing functions
import numpy as np
import config


def mean_impute(Xtr, Xte):
    """
    Replace NaN values in Xtr and Xte with the mean of each column (0 if all NaN in a column, based on Xtr).
    """
    col_mean = np.nanmean(Xtr, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    nan_idx_tr = np.where(np.isnan(Xtr))
    nan_idx_te = np.where(np.isnan(Xte))
    Xtr[nan_idx_tr] = np.take(col_mean, nan_idx_tr[1])
    Xte[nan_idx_te] = np.take(col_mean, nan_idx_te[1])
    return Xtr, Xte


def smart_impute(Xtr, Xte):
    """
    Smart imputation based on feature distribution characteristics (check viz in viz_eda.ipynb). 
    Uses median for skewed distributions, mean for normal, and mode for discrete/categorical.
    
    Args:
        Xtr: Training data array
        Xte: Test data array
    
    Returns:
        Xtr, Xte: Imputed arrays
    """
    Xtr = np.array(Xtr, dtype=np.float32, copy=True)
    Xte = np.array(Xte, dtype=np.float32, copy=True)
    
    for j in range(Xtr.shape[1]):
        col = Xtr[:, j]
        valid_mask = ~np.isnan(col) # boolean array
        
        if not np.any(valid_mask):
            # All NaNs, fill with 0
            fill_value = 0.0
        else:
            valid_vals = col[valid_mask] #
            n_unique = len(np.unique(valid_vals))
            
            # Check if discrete/categorical (few unique values)
            # Maybe there are discrete features with over 10 values... (Need to check that)
            if n_unique <= 10:
                # Use mode (most frequent value)
                unique, counts = np.unique(valid_vals, return_counts=True)
                fill_value = unique[np.argmax(counts)] # select most frequent value
            else:
                # Check skewness for continuous features
                mean_val = np.mean(valid_vals)
                median_val = np.median(valid_vals)
                
                # If mean and median differ significantly, distribution is skewed
                if abs(mean_val - median_val) > 0.5 * np.std(valid_vals):
                    # Use median for skewed distributions
                    fill_value = median_val
                else:
                    # Use mean for approximately normal distributions
                    fill_value = mean_val
        
        # Apply imputation
        nan_idx_tr = np.isnan(Xtr[:, j])
        nan_idx_te = np.isnan(Xte[:, j])
        Xtr[nan_idx_tr, j] = fill_value
        Xte[nan_idx_te, j] = fill_value
    
    return Xtr, Xte

#==========================================

def filter_constant_and_nan_columns(Xtr, Xte):
    """Return indices of non-constant and non-NA-only columns."""
    cols = []
    for j in range(Xtr.shape[1]):
        col = Xtr[:, j]
        valid = ~np.isnan(col)
        if not np.any(valid):
            continue
        vals = col[valid]
        if np.all(vals == vals[0]):
            continue
        cols.append(j)
    col_keep = np.array(cols, dtype=int)
    return Xtr[:, col_keep], Xte[:, col_keep] 


def one_hot_encoding(Xtr, Xte): # Does it really do something ???? 
                                # Why don't we OneHotEncode all columns that meets the requirements => require to check how complete them (to avoid too many new columns -- why would it undesired?)
                                # OR use randomness to choose which columns we choose to OneHotEncode 
                                # OR Check the nature of the feature directly (hand-picking -- might be the most efficient ?)  
    """
    Performs limited one-hot encoding on low-cardinality columns (=have a small number of different possible values) of the input feature matrices (Xtr, Xte).

    For each column in Xtr and Xte:
        - If the column has a small number of unique values (as defined by config.LOW_CARD_MAX_UNIQUE),
        and the total number of new columns does not exceed config.MAX_ADDED_ONEHOT,
        create new binary columns for each unique value (except the last one).
        - Each new column indicates whether the original value matches a specific category.
        - Columns with too many unique values or exceeding the cap are left unchanged.

    Args:
        Xtr : array-like, shape (n_samples_train, n_features)
            Training feature matrix.
        Xte : array-like, shape (n_samples_test, n_features)
            Test feature matrix.

    Returns:
        Xtr_new : ndarray
            Transformed training matrix with one-hot encoded columns added.
        Xte_new : ndarray
            Transformed test matrix with one-hot encoded columns added.
        keep_idx : list of int
            Indices of columns that were kept in their original form.
        used_idx : list of int
            Indices of columns that were one-hot encoded.
        plan : list of tuples
            Encoding plan, each tuple contains (column index, values encoded).
    """
    Xtr = np.asarray(Xtr)
    Xte = np.asarray(Xte)
    n_tr, d = Xtr.shape
    assert Xte.shape[1] == d # sanity check

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
        if uniq.shape[0] <= config.LOW_CARD_MAX_UNIQUE:
            uniq_capped = uniq[:min(len(uniq), config.ONEHOT_PER_FEAT_MAX)]
            k_add = max(len(uniq_capped) - 1, 0)
            if added + k_add > config.MAX_ADDED_ONEHOT:
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

    
    print(f"[Preprocess] one-hot: kept {len(keep_idx)} raw cols, "
        f"encoded {len(used_idx)} cols, plan size={sum(len(v) for _, v in plan)}")

    return Xtr_new, Xte_new
 
 
def standardize(Xtr_new, Xte_new):
    mean_tr = np.mean(Xtr_new, axis=0).astype(np.float32) 
    std_tr  = np.std(Xtr_new, axis=0).astype(np.float32)
    std_tr  = np.where(std_tr == 0, 1.0, std_tr) # avoid division by 0 -- safe guard as we already removed constant columns
    Xtr_s = (Xtr_new - mean_tr) / std_tr
    Xte_s = (Xte_new - mean_tr) / std_tr # can't use test stats to standardize! 
    return Xtr_s, Xte_s


def preprocess(x_train, x_test):
    """
    Preprocess train/test sets, return processed matrices.
    """
    Xtr = np.array(x_train, dtype=np.float32, copy=True) # make a copy (default args are passed by reference!)
    Xte = np.array(x_test,  dtype=np.float32, copy=True)

    n_tr, d = Xtr.shape
   
    print(f"[Preprocess] n_train={n_tr}, n_test={Xte.shape[0]}, n_features={d}")

    # Mean imputation: replace NaN by mean of the column (0 if all NaN)
    col_mean = np.nanmean(Xtr, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    inds_tr = np.where(np.isnan(Xtr))
    inds_te = np.where(np.isnan(Xte))
    Xtr[inds_tr] = np.take(col_mean, inds_tr[1])
    Xte[inds_te] = np.take(col_mean, inds_te[1])

    # Remove constant and NaN-only columns 
    Xtr, Xte = filter_constant_and_nan_columns(Xtr, Xte)
    
    print(f"[Preprocess] drop const/NA-only -> keep {Xtr.shape[1]} cols")

    # Light one-hot encoding
    Xtr, Xte = one_hot_encoding(Xtr, Xte)
        
    # Standardization
    Xtr, Xte = standardize(Xtr, Xte)

    # Bias term for w_0
    Xtr_f = np.hstack([np.ones((Xtr.shape[0], 1), dtype=np.float32), Xtr])
    Xte_f = np.hstack([np.ones((Xte.shape[0], 1), dtype=np.float32), Xte])

    print(f"[Preprocess] final dims: train={Xtr_f.shape}, test={Xte_f.shape}")

    return Xtr_f, Xte_f


def preprocess2():
    return







