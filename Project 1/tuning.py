# Hyperparameter tuning orchestration
import numpy as np
import multiprocessing as mp
import os
import cv_utils
import config


def grid_search_cv(X, y, lambda_grid=config.LAMBDA, gamma_grid=config.GAMMA, max_iters=config.MAX_ITERS):
    """
    Perform random search over hyperparameters using cross-validation.
    
    Args:
        X: Feature matrix
        y: Binary labels (0/1)
        n_trials: Number of random configurations to try
        max_iters: Max iterations for each training run
        
    Returns:
        dict: Best hyperparameters and their CV performance
    """
    
    # Prepare tasks for parallel execution
    tasks = [
        (y, X, lam, gam, max_iters)
        for lam in lambda_grid
        for gam in gamma_grid
    ]
    
    # Run parallel cross-validation
    n_processes = max(1, (os.cpu_count() or 2) - 4) #Badr: 8 CPUs
    with mp.get_context("spawn").Pool(processes=n_processes) as pool:
        results = pool.starmap(cv_utils.cross_validate_logistic_regression, tasks)
    
    # Find best configuration by F1 score
    best_result = max(results, key=lambda r: r['mean_f1'])
    
    print(f"[BEST CV] lambda={best_result['lambda']:.3e}, "
          f"gamma={best_result['learning_rate']:.3e}, "
          f"threshold={best_result['optimal_threshold']:.3f}, "
          f"F1={best_result['mean_f1']:.4f} (±{best_result['std_f1']:.4f})")
    
    return best_result


def save_tuning_results(results, filepath=config.SAVE_BEST):
    """Save best hyperparameters to disk."""
    np.savez(
        filepath,
        lambda_=results['lambda'],
        gamma=results['learning_rate'],
        threshold=results['optimal_threshold'],
        mean_f1=results['mean_f1'],
        std_f1=results['std_f1'],
        mean_accuracy=results['mean_accuracy'],
        mean_precision=results['mean_precision'],
        mean_recall=results['mean_recall']
    )
    print(f"[Saved] Best hyperparameters -> {filepath}")


def load_tuning_results(filepath=config.SAVE_BEST):
    """Load best hyperparameters from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found.")
    
    npz = np.load(filepath)
    results = {
        'lambda': float(npz['lambda_']),
        'learning_rate': float(npz['gamma']),
        'optimal_threshold': float(npz['threshold']),
        'mean_f1': float(npz['mean_f1']),
        'std_f1': float(npz['std_f1']),
        'mean_accuracy': float(npz['mean_accuracy']),
        'mean_precision': float(npz['mean_precision']),
        'mean_recall': float(npz['mean_recall'])
    }
    
    print(f"[Loaded] Best hyperparameters from -> {filepath}")
    return results


def tune(X, y, force_retune=False):
    """
    Main tuning function: run or load hyperparameter search.
    
    Args:
        X: Feature matrix
        y: Binary labels
        force_retune: If True, always retune even if saved results exist
        
    Returns:
        dict: Best hyperparameters
    """
    if force_retune or not os.path.exists(config.SAVE_BEST):
        results = random_search_cv(X, y)
        save_tuning_results(results)
    else:
        results = load_tuning_results()
    
    return results