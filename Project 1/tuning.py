# Hyperparameter tuning orchestration
import numpy as np
import multiprocessing as mp
import os
import cv_utils
import config


def grid_search_cv(X, y, lambda_grid=config.LAMBDA, gamma_grid=config.GAMMA, max_iters=config.MAX_ITERS):
    """
    Perform grid search over hyperparameters using cross-validation.
    Results are parallelized across hyperparameter combinations.
    
    Args:
        X: Feature matrix
        y: Binary labels (0/1)
        lambda_grid: List of regularization values to try
        gamma_grid: List of learning rates to try
        max_iters: Max iterations for each training run
        
    Returns:
        tuple: (best_result dict, all_results list)
            - best_result: Dictionary with best hyperparameters
            - all_results: List of dictionaries with all grid search results
    """
    
    # Prepare tasks for parallel execution
    tasks = [
        (y, X, lam, gam, max_iters)
        for lam in lambda_grid
        for gam in gamma_grid
    ]
    
    print(f"[Grid Search] Testing {len(tasks)} hyperparameter combinations...")
    print(f"[Parallelization] Using {max(1, (os.cpu_count() or 2) - 4)} processes")
    
    # Run parallel cross-validation
    n_processes = max(1, (os.cpu_count() or 2) - 4) #Badr: 8 CPUs
    with mp.get_context("spawn").Pool(processes=n_processes) as pool:
        results = pool.starmap(cv_utils.cross_validate_logistic_regression, tasks)
    
    # Find best configuration by F1 score
    best_result = max(results, key=lambda r: r['mean_f1'])
    
    print(f"\n[BEST CV] lambda={best_result['lambda']:.3e}, "
          f"gamma={best_result['learning_rate']:.3e}, "
          f"threshold={best_result['optimal_threshold']:.3f}")
    print(f"          F1={best_result['mean_f1']:.4f} (±{best_result['std_f1']:.4f}), "
          f"Acc={best_result['mean_accuracy']:.4f} (±{best_result['std_accuracy']:.4f})")
    
    return best_result, results


def save_tuning_results(results, results_list, filepath_npz=config.SAVE_BEST, filepath_csv=None):
    """
    Save best hyperparameters and all grid search results.
    
    Args:
        results: Dictionary with best hyperparameters
        results_list: List of dictionaries with all grid search results
        filepath_npz: Path to save best parameters (NPZ format)
        filepath_csv: Path to save all results (CSV format). 
                      If None, derives from filepath_npz
    """
    # Save best parameters to NPZ for backward compatibility
    np.savez(
        filepath_npz,
        lambda_=results['lambda'],
        gamma=results['gamma'],
        max_iters=results['max_iters'],
        mean_accuracy=results['mean_accuracy'],
        std_accuracy=results['std_accuracy'],
        mean_precision=results['mean_precision'],
        std_precision=results['std_precision'],
        mean_recall=results['mean_recall'],
        std_recall=results['std_recall'],
        mean_f1=results['mean_f1'],
        std_f1=results['std_f1'],
        optimal_threshold=results['optimal_threshold']
    )
    print(f"[Saved] Best hyperparameters -> {filepath_npz}")
    
    # Save all results to CSV
    if filepath_csv is None:
        filepath_csv = filepath_npz.replace('.npz', '_all_results.csv')
    
    # Write CSV manually without pandas
    # Column order: lambda, gamma, max_iters, acc_mean, acc_std, prec_mean, prec_std, rec_mean, rec_std, f1_mean, f1_std, optimal_threshold
    with open(filepath_csv, 'w') as f:
        # Write header
        f.write('lambda,gamma,max_iters,mean_accuracy,std_accuracy,mean_precision,std_precision,mean_recall,std_recall,mean_f1,std_f1,optimal_threshold\n')
        
        # Write each result row
        for r in results_list:
            f.write(f"{r['lambda']:.6f},{r['gamma']:.6f},{r['max_iters']},"
                   f"{r['mean_accuracy']:.6f},{r['std_accuracy']:.6f},"
                   f"{r['mean_precision']:.6f},{r['std_precision']:.6f},"
                   f"{r['mean_recall']:.6f},{r['std_recall']:.6f},"
                   f"{r['mean_f1']:.6f},{r['std_f1']:.6f},"
                   f"{r['optimal_threshold']:.6f}\n")
    
    print(f"[Saved] All grid search results ({len(results_list)} combinations) -> {filepath_csv}")


def load_tuning_results(filepath=config.SAVE_BEST):
    """Load best hyperparameters from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found.")
    
    npz = np.load(filepath)
    results = {
        'lambda': float(npz['lambda_']),
        'gamma': float(npz['gamma']),
        'max_iters': int(npz['max_iters']),
        'mean_accuracy': float(npz['mean_accuracy']),
        'std_accuracy': float(npz['std_accuracy']),
        'mean_precision': float(npz['mean_precision']),
        'std_precision': float(npz['std_precision']),
        'mean_recall': float(npz['mean_recall']),
        'std_recall': float(npz['std_recall']),
        'mean_f1': float(npz['mean_f1']),
        'std_f1': float(npz['std_f1']),
        'optimal_threshold': float(npz['optimal_threshold'])
    }
    
    print(f"[Loaded] Best hyperparameters from -> {filepath}")
    return results


def load_best_from_csv(filepath_csv):
    """
    Load best hyperparameters from CSV by selecting row with highest f1_mean.
    
    Args:
        filepath_csv: Path to CSV file with all grid search results
        
    Returns:
        dict: Best hyperparameters in same format as load_tuning_results()
    """
    if not os.path.exists(filepath_csv):
        raise FileNotFoundError(f"{filepath_csv} not found.")
    
    # Read CSV manually without pandas.. casse la tête
    with open(filepath_csv, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    header = lines[0].strip().split(',')
    
    # Parse data rows
    best_f1 = -1.0
    best_row = None
    
    for line in lines[1:]:  # Skip header
        values = line.strip().split(',')
        row_dict = dict(zip(header, values))
        
        # Check if this row has better F1
        f1_mean = float(row_dict['f1_mean'])
        if f1_mean > best_f1:
            best_f1 = f1_mean
            best_row = row_dict
    
    if best_row is None:
        raise ValueError(f"No valid data found in {filepath_csv}")
    
    # Convert to same format as load_tuning_results()
    results = {
        'lambda': float(best_row['lambda']),
        'gamma': float(best_row['gamma']),
        'max_iters': int(best_row['max_iters']),
        'mean_accuracy': float(best_row['mean_accuracy']),
        'std_accuracy': float(best_row['std_accuracy']),
        'mean_precision': float(best_row['mean_precision']),
        'std_precision': float(best_row['std_precision']),
        'mean_recall': float(best_row['mean_recall']),
        'std_recall': float(best_row['std_recall']),
        'mean_f1': float(best_row['mean_f1']),
        'std_f1': float(best_row['std_f1']),
        'optimal_threshold': float(best_row['optimal_threshold'])
    }
    
    print(f"[Loaded] Best hyperparameters from CSV -> {filepath_csv}")
    print(f"[BEST] lambda={results['lambda']:.3e}, gamma={results['gamma']:.3e}, "
          f"F1={results['mean_f1']:.4f} (±{results['std_f1']:.4f})")
    
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
        best_result, results_list = grid_search_cv(X, y)
        save_tuning_results(best_result, results_list)
    else:
        best_result = load_tuning_results()
    
    return best_result