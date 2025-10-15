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


def cross_validate_logistic_regression(y, X, lambda_, gamma, max_iters=config.MAX_ITERS):
    """
    Perform k-fold cross-validation for regularized logistic regression.
    
    Trains a model on each fold, finds the optimal classification threshold
    that maximizes F1 score across all validation data, and evaluates
    performance metrics per fold.
    
    Args:
        y: Binary labels (0 or 1), shape (n_samples,)
        X: Feature matrix, shape (n_samples, n_features)
        folds: List of (train_indices, val_indices) tuples
        lambda_: L2 regularization strength
        gamma: Gradient descent step size
        max_iters: Maximum number of training iterations
        
    Returns:
        dict: Cross-validation results containing:
            - 'lambda_': Regularization parameter
            - 'gamma': Learning rate used
            - 'optimal_threshold': Best classification threshold
            - 'mean_accuracy': Average accuracy across folds
            - 'mean_precision': Average precision across folds
            - 'mean_recall': Average recall across folds
            - 'mean_f1': Average F1 score across folds
            - 'std_f1': Standard deviation of F1 across folds
    """
    n_features = X.shape[1]
    folds = create_stratified_folds(y)
    
    # Phase 1: Train models and collect validation predictions
    fold_predictions = []
    fold_indices = []
    
    for train_idx, val_idx in folds:
        # Initialize weights
        initial_w = np.zeros(n_features, dtype=np.float64)
        
        # Train model on this fold
        weights, _ = implementations.reg_logistic_regression( 
            y[train_idx], X[train_idx], lambda_, initial_w, max_iters, gamma
        )
        
        # Get validation predictions
        val_probabilities = implementations.sigmoid(X[val_idx] @ weights)
        
        fold_predictions.append(val_probabilities)
        fold_indices.append(val_idx)
    
    # Phase 2: Find optimal threshold across all validation data
    all_val_indices = np.concatenate(fold_indices)
    all_val_probabilities = np.concatenate(fold_predictions)
    all_val_labels = y[all_val_indices]
    
    optimal_threshold, _, _, _ = find_optimal_threshold(
        all_val_labels, all_val_probabilities
    )
    
    # Phase 3: Evaluate each fold with the optimal threshold
    fold_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for val_probs, val_idx in zip(fold_predictions, fold_indices):
        predictions = (val_probs >= optimal_threshold).astype(int)
        true_labels = y[val_idx]
        
        fold_metrics['accuracy'].append(
            metrics.accuracy_score(true_labels, predictions)
        )
        
        precision, recall, f1 = metrics.precision_recall_f1(
            true_labels, 
            predictions
        )
        fold_metrics['precision'].append(precision)
        fold_metrics['recall'].append(recall)
        fold_metrics['f1'].append(f1)
    
    # Return structured results
    return {
        'lambda': float(lambda_),
        'learning_rate': float(gamma),
        'optimal_threshold': float(optimal_threshold),
        'mean_accuracy': float(np.mean(fold_metrics['accuracy'])),
        'mean_precision': float(np.mean(fold_metrics['precision'])),
        'mean_recall': float(np.mean(fold_metrics['recall'])),
        'mean_f1': float(np.mean(fold_metrics['f1'])),
        'std_f1': float(np.std(fold_metrics['f1']))
    }


# def cv_train_and_eval(args):
#     y_tr_01, X_tr, folds, lam, gam, max_iters = args
#     #cross-validation with best threshold found on each fold
#     per_fold_probs, per_fold_idx = [], []
#     for (tr_idx, va_idx) in folds:
#         w0 = np.zeros(X_tr.shape[1], dtype=np.float32)
#         w, _ = implementations.reg_logistic_regression(y_tr_01[tr_idx], X_tr[tr_idx], lam, w0, max_iters, gam)
#         probs_va = implementations.sigmoid(X_tr[va_idx].dot(w))
#         per_fold_probs.append(probs_va)
#         per_fold_idx.append(va_idx)

#     va_idx_concat = np.concatenate(per_fold_idx)
#     probs_concat  = np.concatenate(per_fold_probs)
#     y_val_concat  = y_tr_01[va_idx_concat]
#     best_thr, _, _, _ = find_optimal_threshold(y_val_concat, probs_concat)
#     #evaluate with best_thr on each fold and average ( see slide 4a pg 24)
#     acc_list, prec_list, rec_list, f1_list = [], [], [], []
#     for probs_va, va_idx in zip(per_fold_probs, per_fold_idx):
#         preds = (probs_va >= best_thr).astype(int)
#         y_va  = y_tr_01[va_idx]
#         acc_list.append(metrics.accuracy_score(y_va, preds))
#         p, r, f1 = metrics.precision_recall_f1(y_va, preds)
#         prec_list.append(p)
#         rec_list.append(r)
#         f1_list.append(f1)

#     return (lam, gam, float(best_thr), float(np.mean(acc_list)), float(np.mean(prec_list)),
#             float(np.mean(rec_list)), float(np.mean(f1_list)))



