# Cross-validation and training utilities 
import numpy as np 
import metrics
import implementations
import config 


#stratified_kfold_indices
def create_stratified_folds(ytr_01, k=config.N_FOLDS, random_seed=config.RNG_SEED): 
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

#best_threshold_by_f1()
def find_optimal_threshold(y_true, y_prob):
    """
    Find the optimal classification threshold that maximizes F1 score.
    
    Args:
        y_true: Binary ground truth labels (0 or 1), shape (n_samples,)
        y_prob: Predicted probabilities/scores, shape (n_samples,)
        
    Returns:
        tuple: (optimal_threshold, precision, recall, f1_score)
            - optimal_threshold: Best threshold for classification
            - precision: Precision at optimal threshold
            - recall: Recall at optimal threshold  
            - f1_score: F1 score at optimal threshold
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Sort scores in descending order
    sorted_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[sorted_indices]
    y_prob_sorted = y_prob[sorted_indices]

    # Total number of positive samples
    n_positives = np.sum(y_true == 1)

    # Compute cumulative TP and FP counts at each threshold
    cumulative_tp = np.cumsum(y_true_sorted == 1)
    cumulative_fp = np.cumsum(y_true_sorted == 0)
    
    # Compute precision, recall, and F1 at each threshold
    # Add small epsilon to avoid division by zero
    precision = cumulative_tp / (cumulative_tp + cumulative_fp + config.EPS)
    recall = cumulative_tp / (n_positives + config.EPS)
    f1_scores = 2 * precision * recall / (precision + recall + config.EPS)
    
    # Find index with maximum F1 score
    best_idx = np.argmax(f1_scores)
    
    # Use midpoint between consecutive scores as threshold
    # This ensures consistent classification for edge cases
    if best_idx < len(y_prob_sorted) - 1:
        optimal_threshold = (y_prob_sorted[best_idx] + y_prob_sorted[best_idx + 1]) / 2.0
    else:
        # For the last element, use a value slightly below it
        optimal_threshold = y_prob_sorted[best_idx] - config.EPS
    
    return (
        float(optimal_threshold),
        float(precision[best_idx]),
        float(recall[best_idx]),
        float(f1_scores[best_idx])
    )


# def cross_validate_logistic_regression(y, X, lambda_, gamma, max_iters=config.MAX_ITERS):
#     """
#     Perform k-fold cross-validation for regularized logistic regression.
    
#     Trains a model on each fold, finds the optimal classification threshold
#     that maximizes F1 score across all validation data, and evaluates
#     performance metrics per fold.
    
#     Args:
#         y: Binary labels (0 or 1), shape (n_samples,)
#         X: Feature matrix, shape (n_samples, n_features)
#         folds: List of (train_indices, val_indices) tuples
#         lambda_: L2 regularization strength
#         gamma: Gradient descent step size
#         max_iters: Maximum number of training iterations
        
#     Returns:
#         dict: Cross-validation results containing:
#             - 'lambda': Regularization parameter
#             - 'learning_rate': Learning rate used
#             - 'max_iters': Maximum iterations used
#             - 'optimal_threshold': Best classification threshold
#             - 'mean_accuracy': Average accuracy across folds
#             - 'std_accuracy': Standard deviation of accuracy
#             - 'mean_precision': Average precision across folds
#             - 'std_precision': Standard deviation of precision
#             - 'mean_recall': Average recall across folds
#             - 'std_recall': Standard deviation of recall
#             - 'mean_f1': Average F1 score across folds
#             - 'std_f1': Standard deviation of F1 across folds
#     """
#     n_features = X.shape[1]
#     folds = create_stratified_folds(y)
    
#     # Phase 1: Train models and collect validation predictions
#     fold_predictions = []
#     fold_indices = []
    
#     for train_idx, val_idx in folds:
#         # Initialize weights
#         initial_w = np.zeros(n_features, dtype=np.float64) # put this in config.py ??
        
#         # Train model on this fold
#         weights, _ = implementations.reg_logistic_regression( 
#             y[train_idx], X[train_idx], lambda_, initial_w, max_iters, gamma
#         )
        
#         # Get validation predictions
#         val_probabilities = implementations.sigmoid(X[val_idx] @ weights)
        
#         fold_predictions.append(val_probabilities)
#         fold_indices.append(val_idx)
    
#     # Phase 2: Find optimal threshold across all validation data
#     all_val_indices = np.concatenate(fold_indices)
#     all_val_probabilities = np.concatenate(fold_predictions)
#     all_val_labels = y[all_val_indices]
    
#     optimal_threshold, _, _, _ = find_optimal_threshold(
#         all_val_labels, all_val_probabilities
#     )
    
#     # Phase 3: Evaluate each fold with the optimal threshold
#     fold_metrics = {
#         'accuracy': [],
#         'precision': [],
#         'recall': [],
#         'f1': []
#     }
    
#     for val_probs, val_idx in zip(fold_predictions, fold_indices):
#         predictions = (val_probs >= optimal_threshold).astype(int)
#         true_labels = y[val_idx]
        
#         fold_metrics['accuracy'].append(
#             metrics.accuracy_score(true_labels, predictions)
#         )
        
#         precision, recall, f1 = metrics.precision_recall_f1(
#             true_labels, 
#             predictions
#         )
#         fold_metrics['precision'].append(precision)
#         fold_metrics['recall'].append(recall)
#         fold_metrics['f1'].append(f1)
    
#     # Return structured results with all statistics
#     return {
#         'lambda': float(lambda_),
#         'gamma': float(gamma),
#         'max_iters': int(max_iters),
#         'mean_accuracy': float(np.mean(fold_metrics['accuracy'])),
#         'std_accuracy': float(np.std(fold_metrics['accuracy'])),
#         'mean_precision': float(np.mean(fold_metrics['precision'])),
#         'std_precision': float(np.std(fold_metrics['precision'])),
#         'mean_recall': float(np.mean(fold_metrics['recall'])),
#         'std_recall': float(np.std(fold_metrics['recall'])),
#         'mean_f1': float(np.mean(fold_metrics['f1'])),
#         'std_f1': float(np.std(fold_metrics['f1'])),
#         'optimal_threshold': float(optimal_threshold)
#     }

#==========================================

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


def cv_train_and_eval(y_tr_01, X_tr, lam, gam, max_iters, use_adam, schedule_name, early_stopping, patience, tol): #(args):
    """
    Cross-validated training and evaluation with advanced optimization options.

    Performs stratified K-fold cross-validation for regularized logistic regression
    with support for multiple optimizers (Adam/GD), learning rate schedules 
    (cosine/exponential), and early stopping. For each fold, trains on the 
    training split and collects out-of-fold predictions. Aggregates all validation
    predictions to find the globally optimal F1-maximizing threshold, then 
    evaluates each fold using this shared threshold.

    Args:
        args (tuple): Configuration tuple containing:
            - y_tr_01 (np.ndarray): Binary labels (0/1) for entire training set
            - X_tr (np.ndarray): Feature matrix for entire training set
            - lam (float): L2 regularization strength (lambda)
            - gam (float): Initial learning rate (gamma)
            - max_iters (int): Maximum training iterations per fold
            - use_adam (int/bool): If True/1, use Adam optimizer; else use GD
            - schedule_name (str): Learning rate schedule ('cosine', 'exponential', or None)
            - early_stopping (int/bool): If True/1, enable early stopping
            - patience (int): Number of iterations to wait for improvement before stopping
            - tol (float): Minimum improvement threshold for early stopping

    Returns:
        dict: Cross-validation results containing:
            - 'lambda' (float): Regularization parameter used
            - 'gamma' (float): Initial learning rate used
            - 'max_iters' (int): Maximum iterations used
            - 'mean_accuracy' (float): Average accuracy across all folds
            - 'std_accuracy' (float): Standard deviation of accuracy across folds
            - 'mean_precision' (float): Average precision across all folds
            - 'std_precision' (float): Standard deviation of precision across folds
            - 'mean_recall' (float): Average recall across all folds
            - 'std_recall' (float): Standard deviation of recall across folds
            - 'mean_f1' (float): Average F1 score across all folds
            - 'std_f1' (float): Standard deviation of F1 across folds
            - 'optimal_threshold' (float): Global F1-optimal classification threshold
            - 'adam' (bool): Whether Adam optimizer was used
            - 'schedule' (str): Learning rate schedule used ('cosine', 'exponential', or 'none')
    """
    # (
    #     y_tr_01, X_tr, lam, gam, max_iters, use_adam, schedule_name, early_stopping, patience, tol
    # ) = args

    folds = create_stratified_folds(y_tr_01)

    if schedule_name == "cosine":
        schedule = schedule_cosine
    elif schedule_name == "exponential":
        schedule = schedule_exponential
    else:
        schedule = None

    per_fold_probs, per_fold_idx = [], []

    for (tr_idx, va_idx) in folds:
        w0 = np.zeros(X_tr.shape[1], dtype=np.float32)

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
    best_thr, _, _, _ = find_optimal_threshold(y_val_concat, probs_concat)

    acc_list, prec_list, rec_list, f1_list = [], [], [], []
    for probs_va, va_idx in zip(per_fold_probs, per_fold_idx):
        preds = (probs_va >= best_thr).astype(int)
        y_va = y_tr_01[va_idx]
        acc_list.append(metrics.accuracy_score(y_va, preds))
        p, r, f1 = metrics.precision_recall_f1(y_va, preds)
        prec_list.append(p)
        rec_list.append(r)
        f1_list.append(f1)

    return {
        'lambda': float(lam),
        'gamma': float(gam),
        'max_iters': int(max_iters),
        'mean_accuracy': float(np.mean(acc_list)),
        'std_accuracy': float(np.std(acc_list)),
        'mean_precision': float(np.mean(prec_list)),
        'std_precision': float(np.std(prec_list)),
        'mean_recall': float(np.mean(rec_list)),
        'std_recall': float(np.std(rec_list)),
        'mean_f1': float(np.mean(f1_list)),
        'std_f1': float(np.std(f1_list)),
        'optimal_threshold': float(best_thr),
        'adam': bool(use_adam),
        'schedule': schedule_name if schedule_name else 'none'
    }

