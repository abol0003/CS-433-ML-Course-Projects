# Preprocessing functions
import numpy as np
import os 
import config
import metrics
import helpers



def remove_low_validity_features(Xtr, Xte, threshold=0.05):
    # Shoudld we remove even more ???
    """
    Remove features with less than a specified percentage of valid (non-NaN) data.
    
    Args:
        Xtr: Training data array
        Xte: Test data array
        threshold: Minimum percentage of valid data required to keep a feature (default: 0.05 = 5%)
    
    Returns:
        Xtr, Xte: Filtered arrays with only features that have sufficient valid data
    """
    Xtr = np.array(Xtr, dtype=np.float32, copy=True)
    Xte = np.array(Xte, dtype=np.float32, copy=True)
    
    n_samples = Xtr.shape[0]
    cols_to_keep = []
    
    for j in range(Xtr.shape[1]):
        col = Xtr[:, j]
        valid_count = np.sum(~np.isnan(col))
        valid_ratio = valid_count / n_samples
        
        if valid_ratio > threshold:
            cols_to_keep.append(j)
    
    cols_to_keep = np.array(cols_to_keep, dtype=int)
    
    print(f"[Preprocess] remove low-validity features: kept {len(cols_to_keep)}/{Xtr.shape[1]} cols (threshold: {threshold*100:.1f}% valid data)")
    
    return Xtr[:, cols_to_keep], Xte[:, cols_to_keep]


#==========================================


def mean_impute(Xtr, Xte): ## DANGER: has to be the fist step in preproc pipeline if use of CAT, DISC, CONT
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
 
 
#==========================================


def variance_treshold(Xtr, Xte, threshold=0.01): #DANGER: This step must be before standardize ! caveat: for continuous feature, variance can be much higher.. 
    #for on-hoz encoded features: variance is always <= 0.25
    """
    Remove features (columns) with variance below the given threshold.
    Assumes missing values have already been imputed.

    Args:
        Xtr: Training data array 
        Xte: Test data array 
        threshold: Minimum variance required to keep a feature (default: 0.01)

    Returns:
        Xtr_new, Xte_new: Arrays with low-variance features removed
    """
    Xtr = np.array(Xtr, dtype=np.float32, copy=True)
    Xte = np.array(Xte, dtype=np.float32, copy=True)
    variances = np.var(Xtr, axis=0)
    cols_to_keep = np.where(variances >= threshold)[0]

    print(f"[Preprocess] variance threshold: kept {len(cols_to_keep)}/{Xtr.shape[1]} cols (threshold: {threshold})")

    return Xtr[:, cols_to_keep], Xte[:, cols_to_keep]


def remove_highly_correlated_features(Xtr, Xte, y_train, threshold=0.90):
    """
    Correlation-based feature selection to remove highly correlated features.
    For each pair of features with correlation >= threshold, keeps the one that:
    1) Has higher correlation with target (y_train)
    2) If tied, has less missing data (from x_train_raw)
    3) If still tied, has higher variance
    
    Args:
        Xtr: Training data array (already preprocessed/imputed)
        Xte: Test data array (already preprocessed/imputed)
        y_train: Target labels for training data
        threshold: Correlation threshold for considering features as highly correlated (default: 0.90)
    
    Returns:
        Xtr_new, Xte_new: Arrays with highly correlated features removed
    """
   
    Xtr = np.array(Xtr, dtype=np.float32, copy=True)
    Xte = np.array(Xte, dtype=np.float32, copy=True)
    
    n_features = Xtr.shape[1]
    
    # Correlation matrix 
    corr_matrix = np.corrcoef(Xtr, rowvar=False)
    
    # Correlation with target
    target_corr = np.zeros(n_features)
    for j in range(n_features):
        target_corr[j] = abs(np.corrcoef(Xtr[:, j], y_train)[0, 1])
    
    # Variances
    variances = np.var(Xtr, axis=0)
    
    # Find highly correlated pairs
    features_to_remove = set()
    
    for i in range(n_features):
        if i in features_to_remove:
            continue # skips the rest of current loop
            
        for j in range(i + 1, n_features):
            if j in features_to_remove:
                continue
            
            if abs(corr_matrix[i, j]) >= threshold:
                # Decide which feature to remove
                # Criterion 1: Higher correlation with target
                if target_corr[i] > target_corr[j]:
                    features_to_remove.add(j)
                elif target_corr[j] > target_corr[i]:
                    features_to_remove.add(i)
                else:
                    # Criterion 2: Higher variance
                    if variances[i] > variances[j]:
                        features_to_remove.add(j)
                    else:
                        features_to_remove.add(i)
    
    features_to_keep = [i for i in range(n_features) if i not in features_to_remove]
    
    print(f"[Preprocess] correlation-based selection: removed {len(features_to_remove)} features "
          f"(threshold: {threshold}), kept {len(features_to_keep)}/{n_features} cols")

    return Xtr[:, features_to_keep], Xte[:, features_to_keep]


#==========================================


def standardize(Xtr_new, Xte_new):
    mean_tr = np.mean(Xtr_new, axis=0).astype(np.float32) 
    std_tr  = np.std(Xtr_new, axis=0).astype(np.float32)
    std_tr  = np.where(std_tr == 0, 1.0, std_tr) # avoid division by 0 -- safe guard as we already removed constant columns
    Xtr_s = (Xtr_new - mean_tr) / std_tr
    Xte_s = (Xte_new - mean_tr) / std_tr # can't use test stats to standardize! 
    return Xtr_s, Xte_s


#==========================================


def pca(Xtr, Xte, n_components=config.K):
    """
    Apply Principal Component Analysis (PCA) for dimensionality reduction.
    
    PCA transforms features into orthogonal principal components ordered by 
    variance explained. This can help:
    - Reduce multicollinearity
    - Speed up training
    - Reduce overfitting (regularization effect)
    - Denoise data
    
    Args:
        Xtr: Training data array (standardized, n_samples x n_features)
        Xte: Test data array (standardized, n_samples x n_features)
        n_components: Number of components to keep. If None, uses explained_variance_ratio
    
    Returns:
        Xtr_pca, Xte_pca: Transformed arrays with reduced dimensionality
    """
    Xtr = np.array(Xtr, dtype=np.float32, copy=True)
    Xte = np.array(Xte, dtype=np.float32, copy=True)
    
    n_samples, n_features = Xtr.shape
    
    # Compute covariance matrix
    # Use (X^T @ X) / (n-1) for numerical stability
    cov_matrix = (Xtr.T @ Xtr) / (n_samples - 1)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total_variance = np.sum(eigenvalues)
    variance_explained = np.sum(eigenvalues[:n_components]) / total_variance
    
    print(f"[Preprocess] PCA: keeping {n_components}/{n_features} components "
            f"(explains {variance_explained*100:.2f}% variance)")
    
    # Select top n_components eigenvectors
    principal_components = eigenvectors[:, :n_components]
    
    # Project data onto principal components
    Xtr_pca = Xtr @ principal_components
    Xte_pca = Xte @ principal_components
    
    return Xtr_pca, Xte_pca


def poly_features():
    """
    meh. risk of overfitting ? Give it a try if time allows, especially if combined with PCA 
    """
    return 


def compute_sample_weights(y_train):
    """
    Compute sample weights to handle class imbalance in the dataset.
    
    For imbalanced datasets (+1: 91.17%, -1: 8.83%), this assigns higher
    weights to minority class samples to balance their influence during training.
    
    Weight formula for 'balanced' strategy:
        w_i = n_samples / (n_classes * n_samples_in_class_i)
    
    Args:
        y_train: Training labels array (n_samples,), expected in {-1, +1} or {0, 1} format
        strategy: Weighting strategy. Options:
            - 'balanced': Automatically adjust weights inversely proportional to class frequencies
            - 'uniform': Equal weights for all samples (returns ones)
    
    Returns:
        sample_weights: Array of shape (n_samples,) with weight for each sample
    """
    y_train = np.asarray(y_train)
    n_samples = len(y_train)  

    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    n_classes = len(unique_classes)
    
    # Compute weight for each class: n_samples / (n_classes * count_for_class)
    class_weights = n_samples / (n_classes * class_counts)
    
    # Create mapping from class label to weight
    class_to_weight = dict(zip(unique_classes, class_weights))
    
    # Assign weight to each sample based on its class
    sample_weights = np.array([class_to_weight[label] for label in y_train], 
                                dtype=np.float32)
    
    # Log class distribution and weights
    print(f"[Class Weighting] Dataset imbalance detected:")
    for cls, count in zip(unique_classes, class_counts):
        pct = 100 * count / n_samples
        weight = class_to_weight[cls]
        print(f"  Class {cls:+d}: {count:6d} samples ({pct:5.2f}%) -> weight: {weight:.4f}")
    
    return sample_weights
    


#==========================================
#==========================================


# def preprocess1(Xtr_raw, Xte_raw):
#     """
#     Preprocess train/test sets, return processed matrices.
#     """
#     Xtr_raw = np.array(Xtr_raw, dtype=np.float32, copy=True) # make a copy (default args are passed by reference!)
#     Xte_raw = np.array(Xte_raw,  dtype=np.float32, copy=True)
   

#     Xtr, Xte = mean_impute(Xtr_raw, Xte_raw)

#     Xtr, Xte = filter_constant_and_nan_columns(Xtr, Xte)
    
#     #print(f"[Preprocess] drop const/NA-only -> keep {Xtr.shape[1]} cols")

#     # Light one-hot encoding
#     Xtr, Xte = one_hot_encoding(Xtr, Xte)
        
#     # Standardization
#     Xtr, Xte = standardize(Xtr, Xte)

#     # Bias term for w_0
#     Xtr_f = np.hstack([np.ones((Xtr.shape[0], 1), dtype=np.float32), Xtr])
#     Xte_f = np.hstack([np.ones((Xte.shape[0], 1), dtype=np.float32), Xte])

#     print(f"[Preprocess] final dims: train={Xtr_f.shape}, test={Xte_f.shape}")

#     # func in implementations.py assumes y takes {0,1} !
#     y_tr_01 = metrics.to_01_labels(y_train_pm1) 

#     return Xtr_f, Xte_f


def preprocess2():
    if config.PREPROCESSING:
        Xtr_raw, Xte_raw, ytr_pm1, train_ids, test_ids = helpers.load_csv_data(config.DATA_DIR)

        Xtr_raw = np.array(Xtr_raw, dtype=np.float32, copy=True) # make a copy (default args are passed by reference!)
        Xte_raw = np.array(Xte_raw,  dtype=np.float32, copy=True)
        print(Xtr_raw.shape[1])

        print("[Preprocess] Removing low-validity features...")
        Xtr, Xte = remove_low_validity_features(Xtr_raw, Xte_raw)
        print(Xtr.shape[1])

        print("[Preprocess] Imputing missing values (smart)...")
        Xtr, Xte = smart_impute(Xtr, Xte)
        print(Xtr.shape[1])

        print("[Preprocess] Removing low-variance features...")
        Xtr, Xte = variance_treshold(Xtr, Xte)
        print(Xtr.shape[1])

        print("[Preprocess] Removing highly correlated features...")
        Xtr, Xte = remove_highly_correlated_features(Xtr, Xte, ytr_pm1)
        print(Xtr.shape[1])
        
        print("[Preprocess] One-hot encoding categorical features...")
        Xtr, Xte = one_hot_encoding(Xtr, Xte)
        print(Xtr.shape[1])

        print("[Preprocess] Standardizing features...")    
        Xtr, Xte = standardize(Xtr, Xte)
        print(Xtr.shape[1])

        print("[Preprocess] PCA...") 
        #Xtr, Xte = pca(Xtr, Xte)
        print(Xtr.shape[1])

        print("[Preprocess] Adding bias term...")
        Xtr = np.hstack([np.ones((Xtr.shape[0], 1), dtype=np.float32), Xtr])
        Xte = np.hstack([np.ones((Xte.shape[0], 1), dtype=np.float32), Xte])
        print(Xtr.shape[1])

        ytr_01 = metrics.to_01_labels(ytr_pm1) 

        print("[Preprocess] Computing sample weights for class imbalance...")
        sample_weights = compute_sample_weights(ytr_pm1)

        print(f"[Preprocess] Saving preprocessed data")
        save(Xtr, Xte, ytr_01, train_ids, test_ids, sample_weights)
    else:
        Xtr, Xte, ytr_01, train_ids, test_ids, sample_weights = load_preproc_data()

    return Xtr, Xte, ytr_01, train_ids, test_ids, sample_weights


#==========================================
#==========================================


def save(Xtr, Xte, ytr, train_ids, test_ids, sample_weights, filename=config.PREPROC_DATA_PATH):
    np.savez_compressed(
        filename,
        X_train   = Xtr, 
        X_test    = Xte, 
        y_train   = ytr,
        train_ids = train_ids, 
        test_ids  = test_ids,
        sample_weights = sample_weights
    )


def load_preproc_data(filename=config.PREPROC_DATA_PATH): 
    """Load best hyperparameters from disk."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found.")
    npz = np.load(filename) 
    Xtr       = npz["X_train"]
    Xte       = npz["X_test"]
    ytr_01    = npz["y_train"]
    train_ids = npz["train_ids"]
    test_ids  = npz["test_ids"]
    sample_weights = npz["sample_weights"]
    print(f"[Loaded] Preprocessed data from -> {filename}")
    return Xtr, Xte, ytr_01, train_ids, test_ids, sample_weights


