import os, sys, numpy as np, multiprocessing as mp
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from pathlib import Path as _Path
for _p in [_Path.cwd(), _Path.cwd().parent, _Path.cwd().parent.parent]:
    if (_p / "implementations.py").exists():
        sys.path.insert(0, str(_p))
        break

import config, implementations, metrics, cv_utils


def sample_grid(
    scale: str = "log",
    n_gamma: int = 10,
    n_lambda: int = 10,
    gamma_low: Optional[float] = None,
    gamma_high: Optional[float] = None,
    lambda_low: Optional[float] = None,
    lambda_high: Optional[float] = None,
    include_lambda_zero: bool = False,
):
    """
    Create gamma/lambda axes for heatmaps or scans.
    - scale: "log" or "linear"
    - include_lambda_zero: when True, prepend 0 to λ axis (useful to compare with no-regularization)
    """
    gamma_low = float(config.GAMMA_LOW if gamma_low is None else gamma_low)
    gamma_high = float(config.GAMMA_HIGH if gamma_high is None else gamma_high)
    lambda_low = float(config.LAMBDA_LOW if lambda_low is None else lambda_low)
    lambda_high = float(config.LAMBDA_HIGH if lambda_high is None else lambda_high)

    if scale == "log":
        gammas = np.logspace(np.log10(gamma_low), np.log10(gamma_high), int(n_gamma))
        if include_lambda_zero:
            # Prepend 0 then positive logspace (cannot include 0 inside logspace)
            if n_lambda < 2:
                lambdas = np.array([0.0])
            else:
                pos = np.logspace(np.log10(max(lambda_low, 1e-12)), np.log10(lambda_high), int(n_lambda) - 1)
                lambdas = np.concatenate([[0.0], pos])
        else:
            lambdas = np.logspace(np.log10(lambda_low), np.log10(lambda_high), int(n_lambda))
    else:
        gammas = np.linspace(gamma_low, gamma_high, int(n_gamma))
        if include_lambda_zero:
            lambdas = np.linspace(0.0, lambda_high, int(n_lambda))
        else:
            lambdas = np.linspace(lambda_low, lambda_high, int(n_lambda))

    return gammas.astype(float), lambdas.astype(float)


@dataclass
class Curves:
    gammas: List[float]
    lambdas: List[float]
    gd_f1: List[float]
    sgd_f1: List[float]
    ridge_f1: List[float]
    logreg_f1_vs_gamma: List[float]
    reglog_f1_vs_gamma: List[float]
    reglog_f1_vs_lambda: List[float]


@dataclass
class Summary:
    best_f1_gd: float
    best_f1_sgd: float
    best_f1_ls: float
    best_f1_ridge: float
    best_f1_log_plain: float
    best_f1_reglog_gamma: float
    best_f1_reglog_lambda: float


@dataclass
class GridSearchResults:
    gammas: List[float]
    lambdas: List[float]
    f1_grid: List[List[float]]
    best_gamma: float
    best_lambda: float
    best_f1: float


def _train_val_f1_for_reglog(args):
    """Worker: compute mean/std F1 on train and val for one (gamma, lambda)."""
    idx_i, idx_j, gam, lam, X, y01, folds, max_it = args
    f_tr, f_va = [], []
    for tr, va in folds:
        # Fit regularized logistic regression
        w, _ = implementations.reg_logistic_regression(
            y01[tr], X[tr], lam, np.zeros(X.shape[1], np.float32), max_it, gam
        )
        # Train
        p_tr = implementations.sigmoid(X[tr] @ w)
        thr_tr, *_ = cv_utils.best_threshold_by_f1(y01[tr], p_tr)
        preds_tr = (p_tr >= thr_tr).astype(int)
        f1_tr = metrics.precision_recall_f1(y01[tr], preds_tr)[2]
        f_tr.append(float(f1_tr))
        # Validation
        p_va = implementations.sigmoid(X[va] @ w)
        thr_va, *_ = cv_utils.best_threshold_by_f1(y01[va], p_va)
        preds_va = (p_va >= thr_va).astype(int)
        f1_va = metrics.precision_recall_f1(y01[va], preds_va)[2]
        f_va.append(float(f1_va))
    tr_mean, tr_std = float(np.mean(f_tr)), float(np.std(f_tr, ddof=1 if len(f_tr) > 1 else 0))
    va_mean, va_std = float(np.mean(f_va)), float(np.std(f_va, ddof=1 if len(f_va) > 1 else 0))
    return idx_i, idx_j, tr_mean, tr_std, va_mean, va_std


def compute_reglog_train_val_grid(
    X_tr: np.ndarray,
    y_train_01: np.ndarray,
    folds,
    gammas: np.ndarray,
    lambdas: np.ndarray,
    max_it: Optional[int] = None,
):
    """Serial computation of train/val F1 mean/std over a γ×λ grid."""
    max_it = int(config.TUNING_MAX_ITERS if max_it is None else max_it)
    G, L = len(gammas), len(lambdas)
    tr_mean = np.zeros((G, L), float)
    tr_std = np.zeros((G, L), float)
    va_mean = np.zeros((G, L), float)
    va_std = np.zeros((G, L), float)

    for i, g in enumerate(gammas):
        for j, lmb in enumerate(lambdas):
            _, _, tm, ts, vm, vs = _train_val_f1_for_reglog((i, j, float(g), float(lmb), X_tr, y_train_01, folds, max_it))
            tr_mean[i, j], tr_std[i, j], va_mean[i, j], va_std[i, j] = tm, ts, vm, vs

    return {
        "gammas": np.array(gammas, float),
        "lambdas": np.array(lambdas, float),
        "train_mean": tr_mean,
        "train_std": tr_std,
        "val_mean": va_mean,
        "val_std": va_std,
    }


def compute_reglog_train_val_grid_parallel(
    X_tr: np.ndarray,
    y_train_01: np.ndarray,
    folds,
    gammas: np.ndarray,
    lambdas: np.ndarray,
    max_it: Optional[int] = None,
    processes: Optional[int] = None,
):
    """Parallel computation of train/val F1 mean/std over a γ×λ grid."""
    max_it = int(config.TUNING_MAX_ITERS if max_it is None else max_it)
    G, L = len(gammas), len(lambdas)
    tr_mean = np.zeros((G, L), float)
    tr_std = np.zeros((G, L), float)
    va_mean = np.zeros((G, L), float)
    va_std = np.zeros((G, L), float)

    params = [
        (i, j, float(g), float(lmb), X_tr, y_train_01, folds, max_it)
        for i, g in enumerate(gammas)
        for j, lmb in enumerate(lambdas)
    ]

    nproc = int(processes if processes is not None else max(1, (os.cpu_count() or 2) - 4))
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=nproc) as pool:
        for i, j, tm, ts, vm, vs in pool.imap_unordered(_train_val_f1_for_reglog, params):
            tr_mean[i, j], tr_std[i, j], va_mean[i, j], va_std[i, j] = tm, ts, vm, vs

    return {
        "gammas": np.array(gammas, float),
        "lambdas": np.array(lambdas, float),
        "train_mean": tr_mean,
        "train_std": tr_std,
        "val_mean": va_mean,
        "val_std": va_std,
    }


# ----- Iteration-wise loss curves for overfitting analysis -----
def _compute_fold_loss_curve(args):
    """Worker: compute loss/F1 curves for one fold."""
    k, tr_idx, va_idx, X, y01, gamma, lambda_, max_iters = args
    
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def _logistic_loss_unreg(Xm, y, w):
        # y in {0,1}
        z = Xm @ w
        # stable: log(1+exp(z)) - y*z
        return float(np.mean(np.logaddexp(0.0, z) - y * z))

    T = int(max_iters)
    d = X.shape[1]
    
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr, yva = y01[tr_idx].astype(float), y01[va_idx].astype(float)

    # L2 on all weights except bias at column 0
    reg_mask = np.ones(d, dtype=float)
    reg_mask[0] = 0.0

    w = np.zeros(d, dtype=float)
    n_tr = Xtr.shape[0]
    
    train_loss = np.zeros(T, dtype=float)
    val_loss = np.zeros(T, dtype=float)
    train_f1 = np.zeros(T, dtype=float)
    val_f1 = np.zeros(T, dtype=float)

    for t in range(T):
        # forward
        z_tr = Xtr @ w
        p_tr = _sigmoid(z_tr)

        # gradient: mean(X^T (p - y)) + 2*lambda*w (no reg on bias)
        grad = (Xtr.T @ (p_tr - ytr)) / n_tr
        grad += 2.0 * float(lambda_) * (reg_mask * w)

        # step
        w -= float(gamma) * grad

        # metrics
        train_loss[t] = _logistic_loss_unreg(Xtr, ytr, w)
        val_loss[t] = _logistic_loss_unreg(Xva, yva, w)

        # F1 with fixed 0.5 threshold on probability
        ytr_pred01 = (p_tr >= 0.5).astype(int)
        yva_pred01 = (_sigmoid(Xva @ w) >= 0.5).astype(int)
        train_f1[t] = metrics.precision_recall_f1(ytr.astype(int), ytr_pred01)[2]
        val_f1[t] = metrics.precision_recall_f1(yva.astype(int), yva_pred01)[2]

    return k, train_loss, val_loss, train_f1, val_f1


def compute_reglog_loss_curves(
    X: np.ndarray,
    y01: np.ndarray,
    folds,
    gamma: float,
    lambda_: float,
    max_iters: int = 400,
):
    """Compute per-iteration logistic loss (unpenalized) and F1 on train and validation across folds.
    Uses multiprocessing (spawn) with max(1, cpu_count - 3) processes.

    Returns:
      {
        'iters': np.arange(T),
        'train_per_fold': [n_folds, T] loss, 'val_per_fold': [n_folds, T] loss,
        'train_mean','train_std','val_mean','val_std',
        'train_f1_per_fold': [n_folds, T], 'val_f1_per_fold': [n_folds, T],
        'train_f1_mean','train_f1_std','val_f1_mean','val_f1_std'
      }
    """
    T = int(max_iters)
    n_folds = len(folds)

    # Prepare args for each fold
    params = [
        (k, tr, va, X, y01, gamma, lambda_, T)
        for k, (tr, va) in enumerate(folds)
    ]

    # Multiprocessing with Windows-safe spawn
    nproc = max(1, (os.cpu_count() or 4) - 3)
    ctx = mp.get_context("spawn")
    
    results = []
    with ctx.Pool(processes=nproc) as pool:
        results = pool.map(_compute_fold_loss_curve, params)

    # Collect results
    train_loss = np.zeros((n_folds, T), dtype=float)
    val_loss = np.zeros((n_folds, T), dtype=float)
    train_f1 = np.zeros((n_folds, T), dtype=float)
    val_f1 = np.zeros((n_folds, T), dtype=float)

    for k, tr_loss, va_loss, tr_f1, va_f1 in results:
        train_loss[k] = tr_loss
        val_loss[k] = va_loss
        train_f1[k] = tr_f1
        val_f1[k] = va_f1

    ddof = 1 if n_folds > 1 else 0
    return {
        'iters': np.arange(T, dtype=int),
        'train_per_fold': train_loss,
        'val_per_fold': val_loss,
        'train_mean': train_loss.mean(axis=0),
        'train_std': train_loss.std(axis=0, ddof=ddof),
        'val_mean': val_loss.mean(axis=0),
        'val_std': val_loss.std(axis=0, ddof=ddof),
        'train_f1_per_fold': train_f1,
        'val_f1_per_fold': val_f1,
        'train_f1_mean': train_f1.mean(axis=0),
        'train_f1_std': train_f1.std(axis=0, ddof=ddof),
        'val_f1_mean': val_f1.mean(axis=0),
        'val_f1_std': val_f1.std(axis=0, ddof=ddof),
    }


def save_heatmap_stats(path: str, stats: dict, gammas: Optional[np.ndarray] = None, lambdas: Optional[np.ndarray] = None, meta: Optional[Dict[str, Any]] = None):
    """Save heatmap stats with optional axis arrays and metadata to match notebook usage."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    g = np.array(stats["gammas"] if gammas is None else gammas, float)
    l = np.array(stats["lambdas"] if lambdas is None else lambdas, float)
    meta_json = None
    if meta is not None:
        try:
            import json
            meta_json = json.dumps(meta)
        except Exception:
            meta_json = None
    np.savez(
        path,
        gammas=g,
        lambdas=l,
        train_mean=np.array(stats["train_mean"], float),
        train_std=np.array(stats["train_std"], float),
        val_mean=np.array(stats["val_mean"], float),
        val_std=np.array(stats["val_std"], float),
        meta_json=meta_json if meta_json is not None else "{}",
    )
    print(f"[Saved] {path}")


def load_heatmap_stats(path: str):
    """Load heatmap stats; returns (stats_dict, gammas, lambdas, meta_dict) for compatibility."""
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=False)
    stats = {
        "gammas": d["gammas"],
        "lambdas": d["lambdas"],
        "train_mean": d["train_mean"],
        "train_std": d["train_std"],
        "val_mean": d["val_mean"],
        "val_std": d["val_std"],
    }
    try:
        import json
        meta = json.loads(str(d["meta_json"])) if "meta_json" in d else {}
    except Exception:
        meta = {}
    return stats, stats["gammas"], stats["lambdas"], meta


# ----- Curves helpers expected by the notebook -----
def compute_hyperparam_curves(
    X_tr: np.ndarray,
    y_tr_pm1: np.ndarray,
    y_tr_01: np.ndarray,
    folds,
    n_trials: Optional[int] = None,
    seed: Optional[int] = None,
    max_iters: Optional[int] = None,
    use_multiprocessing: bool = False,
    processes: Optional[int] = None,
):
    """Compute minimal set of curves and summary as dicts.
    Uses run_all under the hood; ignores some params if unsupported.
    """
    curves_dc, summary_dc = {}, {}
    curves, summary = run_all(
        X_tr, y_tr_pm1, y_tr_01, folds,
        seed=seed,
        use_multiprocessing=use_multiprocessing,
        processes=processes,
    )
    # Convert dataclasses to dicts
    curves_dc = {
        "gammas": np.array(curves.gammas, float),
        "lambdas": np.array(curves.lambdas, float),
        "gd_f1": np.array(curves.gd_f1, float),
        "sgd_f1": np.array(curves.sgd_f1, float),
        "ridge_f1": np.array(curves.ridge_f1, float),
        "logreg_f1_vs_gamma": np.array(curves.logreg_f1_vs_gamma, float),
        "reglog_f1_vs_gamma": np.array(curves.reglog_f1_vs_gamma, float),
        "reglog_f1_vs_lambda": np.array(curves.reglog_f1_vs_lambda, float),
    }
    summary_dc = {
        "best_f1_gd": float(summary.best_f1_gd),
        "best_f1_sgd": float(summary.best_f1_sgd),
        "best_f1_ls": float(summary.best_f1_ls),
        "best_f1_ridge": float(summary.best_f1_ridge),
        "best_f1_log_plain": float(summary.best_f1_log_plain),
        "best_f1_reglog_gamma": float(summary.best_f1_reglog_gamma),
        "best_f1_reglog_lambda": float(summary.best_f1_reglog_lambda),
    }
    return curves_dc, summary_dc


def save_curves(path: str, curves: Dict[str, Any], summary: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        gammas=np.array(curves["gammas"], float),
        lambdas=np.array(curves["lambdas"], float),
        gd_f1=np.array(curves["gd_f1"], float),
        sgd_f1=np.array(curves["sgd_f1"], float),
        ridge_f1=np.array(curves["ridge_f1"], float),
        logreg_f1_vs_gamma=np.array(curves["logreg_f1_vs_gamma"], float),
        reglog_f1_vs_gamma=np.array(curves["reglog_f1_vs_gamma"], float),
        reglog_f1_vs_lambda=np.array(curves["reglog_f1_vs_lambda"], float),
        best_f1_gd=float(summary.get("best_f1_gd", 0.0)),
        best_f1_sgd=float(summary.get("best_f1_sgd", 0.0)),
        best_f1_ls=float(summary.get("best_f1_ls", 0.0)),
        best_f1_ridge=float(summary.get("best_f1_ridge", 0.0)),
        best_f1_log_plain=float(summary.get("best_f1_log_plain", 0.0)),
        best_f1_reglog_gamma=float(summary.get("best_f1_reglog_gamma", 0.0)),
        best_f1_reglog_lambda=float(summary.get("best_f1_reglog_lambda", 0.0)),
    )
    print(f"[Saved] {path}")


def load_curves(path: str):
    curves_dc, summary_dc = load_results(path)
    # Convert dataclasses to dicts
    curves = {
        "gammas": np.array(curves_dc.gammas, float),
        "lambdas": np.array(curves_dc.lambdas, float),
        "gd_f1": np.array(curves_dc.gd_f1, float),
        "sgd_f1": np.array(curves_dc.sgd_f1, float),
        "ridge_f1": np.array(curves_dc.ridge_f1, float),
        "logreg_f1_vs_gamma": np.array(curves_dc.logreg_f1_vs_gamma, float),
        "reglog_f1_vs_gamma": np.array(curves_dc.reglog_f1_vs_gamma, float),
        "reglog_f1_vs_lambda": np.array(curves_dc.reglog_f1_vs_lambda, float),
    }
    summary = {
        "best_f1_gd": float(summary_dc.best_f1_gd),
        "best_f1_sgd": float(summary_dc.best_f1_sgd),
        "best_f1_ls": float(summary_dc.best_f1_ls),
        "best_f1_ridge": float(summary_dc.best_f1_ridge),
        "best_f1_log_plain": float(summary_dc.best_f1_log_plain),
        "best_f1_reglog_gamma": float(summary_dc.best_f1_reglog_gamma),
        "best_f1_reglog_lambda": float(summary_dc.best_f1_reglog_lambda),
    }
    return curves, summary


def sample_loguniform(low: float, high: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.exp(rng.uniform(np.log(low), np.log(high), size=n))


def _mse_and_f1_from_w(w: np.ndarray, Xva: np.ndarray, yva_pm1: np.ndarray) -> Tuple[float, float]:
    yhat = Xva @ w
    mse = float(np.mean((yva_pm1 - yhat) ** 2))
    preds_pm1 = np.where(yhat >= 0, 1, -1)
    preds_01 = metrics.to_01_labels(preds_pm1)
    f1 = metrics.precision_recall_f1(metrics.to_01_labels(yva_pm1), preds_01)[2]
    return mse, float(f1)


def _eval_gd_for_gamma(args):
    g, X_tr, y_pm1, folds, max_it = args
    f1_scores = []
    for tr, va in folds:
        w, _ = implementations.mean_squared_error_gd(
            y_pm1[tr], X_tr[tr], np.zeros(X_tr.shape[1], np.float32), max_it, g
        )
        mse, f1 = _mse_and_f1_from_w(w, X_tr[va], y_pm1[va])
        f1_scores.append(f1)
    return float(np.mean(f1_scores))


def _eval_sgd_for_gamma(args):
    g, X_tr, y_pm1, folds, max_it = args
    f1_scores = []
    for tr, va in folds:
        w, _ = implementations.mean_squared_error_sgd(
            y_pm1[tr], X_tr[tr], np.zeros(X_tr.shape[1], np.float32), max_it * 5, g / 2.0
        )
        mse, f1 = _mse_and_f1_from_w(w, X_tr[va], y_pm1[va])
        f1_scores.append(f1)
    return float(np.mean(f1_scores))


def _eval_ridge_for_lambda(args):
    lmb, X_tr, y_pm1, folds = args
    f1_scores = []
    for tr, va in folds:
        w, _ = implementations.ridge_regression(y_pm1[tr], X_tr[tr], lmb)
        mse, f1 = _mse_and_f1_from_w(w, X_tr[va], y_pm1[va])
        f1_scores.append(f1)
    return float(np.mean(f1_scores))



def _eval_reglog_f1_for_gamma(args):
    gam, lam, X_tr, y01, folds, max_it = args
    f = []
    for tr, va in folds:
        w, _ = implementations.reg_logistic_regression(
            y01[tr], X_tr[tr], lam, np.zeros(X_tr.shape[1], np.float32), max_it, gam
        )
        probs = implementations.sigmoid(X_tr[va] @ w)
        thr, *_ = cv_utils.best_threshold_by_f1(y01[va], probs)
        preds = (probs >= thr).astype(int)
        f.append(metrics.precision_recall_f1(y01[va], preds)[2])
    return float(np.mean(f))


def _eval_reglog_f1_for_lambda(args):
    lmb, gam, X_tr, y01, folds, max_it = args
    f = []
    for tr, va in folds:
        w, _ = implementations.reg_logistic_regression(
            y01[tr], X_tr[tr], lmb, np.zeros(X_tr.shape[1], np.float32), max_it, gam
        )
        probs = implementations.sigmoid(X_tr[va] @ w)
        thr, *_ = cv_utils.best_threshold_by_f1(y01[va], probs)
        preds = (probs >= thr).astype(int)
        f.append(metrics.precision_recall_f1(y01[va], preds)[2])
    return float(np.mean(f))


def _eval_logistic_plain_f1_for_gamma(args):
    gam, X_tr, y01, folds, max_it = args
    f = []
    for tr, va in folds:
        w, _ = implementations.logistic_regression(
            y01[tr], X_tr[tr], np.zeros(X_tr.shape[1], np.float32), max_it, gam
        )
        probs = implementations.sigmoid(X_tr[va] @ w)
        thr, *_ = cv_utils.best_threshold_by_f1(y01[va], probs)
        preds = (probs >= thr).astype(int)
        f.append(metrics.precision_recall_f1(y01[va], preds)[2])
    return float(np.mean(f))


def _eval_reglog_grid_point(args):
    gam, lam, X_tr, y01, folds, max_it = args
    f = []
    for tr, va in folds:
        w, _ = implementations.reg_logistic_regression(
            y01[tr], X_tr[tr], lam, np.zeros(X_tr.shape[1], np.float32), max_it, gam
        )
        probs = implementations.sigmoid(X_tr[va] @ w)
        thr, *_ = cv_utils.best_threshold_by_f1(y01[va], probs)
        preds = (probs >= thr).astype(int)
        f.append(metrics.precision_recall_f1(y01[va], preds)[2])
    return float(np.mean(f))


def run_2d_grid_search(X_tr, y_train_01, folds, n_gamma=5, n_lambda=5, seed=None):
    import time
    t0 = time.time()
    print(f"\n[2D GRID] Starting {n_gamma}×{n_lambda} = {n_gamma*n_lambda} evaluations...")
    
    seed = int(config.RNG_SEED if seed is None else seed)
    MAX_IT = int(config.TUNING_MAX_ITERS)
    
    gammas = np.sort(sample_loguniform(config.GAMMA_LOW, config.GAMMA_HIGH, n_gamma, seed))
    lambdas = np.sort(sample_loguniform(config.LAMBDA_LOW, config.LAMBDA_HIGH, n_lambda, seed + 100))
    print
    grid_params = [(g, l, X_tr, y_train_01, folds, MAX_IT) for g in gammas for l in lambdas]
    
    nproc = max(1, (os.cpu_count() or 2) - 3)
    ctx = mp.get_context("spawn")
    
    print(f"[2D GRID] Using {nproc} processes...")
    with ctx.Pool(processes=nproc) as pool:
        f1_flat = pool.map(_eval_reglog_grid_point, grid_params)
    
    f1_grid = np.array(f1_flat).reshape(n_gamma, n_lambda)
    
    best_idx = np.unravel_index(np.argmax(f1_grid), f1_grid.shape)
    best_gamma = float(gammas[best_idx[0]])
    best_lambda = float(lambdas[best_idx[1]])
    best_f1 = float(f1_grid[best_idx])
    
    print(f"[2D GRID] Completed in {time.time()-t0:.1f}s")
    print(f"[2D GRID] Best: γ={best_gamma:.2e}, λ={best_lambda:.2e}, F1={best_f1:.4f}")
    
    return GridSearchResults(
        gammas=list(map(float, gammas)),
        lambdas=list(map(float, lambdas)),
        f1_grid=f1_grid.tolist(),
        best_gamma=best_gamma,
        best_lambda=best_lambda,
        best_f1=best_f1
    )



import time

def run_all(
    X_tr,
    y_train_pm1,
    y_train_01,
    folds,
    seed=None,
    use_multiprocessing: bool = True,
    processes: Optional[int] = None,
):
    t0 = time.time()
    print("[RUN] Starting hyperparameter sweep")

    GLO, GHI = config.GAMMA_LOW, config.GAMMA_HIGH
    LLO, LHI = config.LAMBDA_LOW, config.LAMBDA_HIGH
    NTR = config.N_TRIALS
    MAX_IT = int(config.TUNING_MAX_ITERS)
    seed = int(config.RNG_SEED if seed is None else seed)

    gammas = np.sort(sample_loguniform(GLO, GHI, NTR, seed))
    lambdas = np.sort(sample_loguniform(LLO, LHI, NTR, seed + 1))
    fix_gam = float(np.sqrt(GLO * GHI))
    fix_lam = float(np.sqrt(LLO * LHI))
    print(f"[RUN] Sampled {NTR} gammas and {NTR} lambdas")

    nproc_ = max(1, (os.cpu_count() or 2) - 4)
    nproc = int(processes) if processes else nproc_
    use_mp = bool(use_multiprocessing and nproc > 1)
    ctx = mp.get_context("spawn") if use_mp else None

    def _map(func, iterable):
        if use_mp:
            with ctx.Pool(processes=nproc) as pool:
                return pool.map(func, iterable)
        # sequential
        return [func(arg) for arg in iterable]

    t_lin = time.time()
    print(f"[LINEAR] Using {nproc if use_mp else 1} {'processes' if use_mp else 'process (sequential)'}")
    gd_mse = _map(_eval_gd_for_gamma, [(g, X_tr, y_train_pm1, folds, MAX_IT) for g in gammas])
    print(f"[DONE] GD in {time.time()-t_lin:.1f}s")

    t_sgd = time.time()
    sgd_mse = _map(_eval_sgd_for_gamma, [(g, X_tr, y_train_pm1, folds, MAX_IT) for g in gammas])
    print(f"[DONE] SGD in {time.time()-t_sgd:.1f}s")

    t_ridge = time.time()
    ridge_mse = _map(_eval_ridge_for_lambda, [(l, X_tr, y_train_pm1, folds) for l in lambdas])
    print(f"[DONE] Ridge in {time.time()-t_ridge:.1f}s")

    t_log = time.time()
    print("[LOGISTIC] Starting")
    reglog_f1_vs_gamma = _map(
        _eval_reglog_f1_for_gamma, [(g, fix_lam, X_tr, y_train_01, folds, MAX_IT) for g in gammas]
    )
    print(f"[DONE] Reg-Logistic (γ-scan) in {time.time()-t_log:.1f}s")

    t_log2 = time.time()
    reglog_f1_vs_lambda = _map(
        _eval_reglog_f1_for_lambda, [(l, fix_gam, X_tr, y_train_01, folds, MAX_IT) for l in lambdas]
    )
    print(f"[DONE] Reg-Logistic (λ-scan) in {time.time()-t_log2:.1f}s")

    t_log3 = time.time()
    logreg_f1_vs_gamma = _map(
        _eval_logistic_plain_f1_for_gamma, [(g, X_tr, y_train_01, folds, MAX_IT) for g in gammas]
    )
    print(f"[DONE] Plain Logistic in {time.time()-t_log3:.1f}s")

    best_f1_gd = float(np.max(gd_mse))
    best_f1_sgd = float(np.max(sgd_mse))
    best_f1_ls = 0.0
    best_f1_ridge = float(np.max(ridge_mse))
    best_f1_log_plain = float(np.max(logreg_f1_vs_gamma))
    best_f1_reglog_gamma = float(np.max(reglog_f1_vs_gamma))
    best_f1_reglog_lambda = float(np.max(reglog_f1_vs_lambda))

    curves = Curves(
        gammas=list(map(float, gammas)),
        lambdas=list(map(float, lambdas)),
        gd_f1=list(map(float, gd_mse)),
        sgd_f1=list(map(float, sgd_mse)),
        ridge_f1=list(map(float, ridge_mse)),
        logreg_f1_vs_gamma=list(map(float, logreg_f1_vs_gamma)),
        reglog_f1_vs_gamma=list(map(float, reglog_f1_vs_gamma)),
        reglog_f1_vs_lambda=list(map(float, reglog_f1_vs_lambda)),
    )

    summary = Summary(
        best_f1_gd=best_f1_gd,
        best_f1_sgd=best_f1_sgd,
        best_f1_ls=best_f1_ls,
        best_f1_ridge=best_f1_ridge,
        best_f1_log_plain=best_f1_log_plain,
        best_f1_reglog_gamma=best_f1_reglog_gamma,
        best_f1_reglog_lambda=best_f1_reglog_lambda,
    )

    print(f"[TOTAL] Completed in {time.time()-t0:.1f}s")
    return curves, summary


def save_results(curves: Curves, summary: Summary, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, gammas=np.array(curves.gammas), lambdas=np.array(curves.lambdas),
        gd_f1=np.array(curves.gd_f1), sgd_f1=np.array(curves.sgd_f1), ridge_f1=np.array(curves.ridge_f1),
        logreg_f1_vs_gamma=np.array(curves.logreg_f1_vs_gamma),
        reglog_f1_vs_gamma=np.array(curves.reglog_f1_vs_gamma),
        reglog_f1_vs_lambda=np.array(curves.reglog_f1_vs_lambda),
        best_f1_gd=summary.best_f1_gd, best_f1_sgd=summary.best_f1_sgd, best_f1_ls=summary.best_f1_ls,
        best_f1_ridge=summary.best_f1_ridge, best_f1_log_plain=summary.best_f1_log_plain,
        best_f1_reglog_gamma=summary.best_f1_reglog_gamma, best_f1_reglog_lambda=summary.best_f1_reglog_lambda)


def load_results(path: str | os.PathLike):
    d = np.load(path)
    gd_key = "gd_f1" if "gd_f1" in d else "gd_mse"
    sgd_key = "sgd_f1" if "sgd_f1" in d else "sgd_mse"
    ridge_key = "ridge_f1" if "ridge_f1" in d else "ridge_mse"
    
    curves = Curves(gammas=d["gammas"].tolist(), lambdas=d["lambdas"].tolist(),
        gd_f1=d[gd_key].tolist(), sgd_f1=d[sgd_key].tolist(), ridge_f1=d[ridge_key].tolist(),
        logreg_f1_vs_gamma=d["logreg_f1_vs_gamma"].tolist(),
        reglog_f1_vs_gamma=d["reglog_f1_vs_gamma"].tolist(),
        reglog_f1_vs_lambda=d["reglog_f1_vs_lambda"].tolist())
    
    summary = Summary(best_f1_gd=float(d["best_f1_gd"]), best_f1_sgd=float(d["best_f1_sgd"]),
        best_f1_ls=float(d["best_f1_ls"]), best_f1_ridge=float(d["best_f1_ridge"]),
        best_f1_log_plain=float(d["best_f1_log_plain"]),
        best_f1_reglog_gamma=float(d["best_f1_reglog_gamma"]),
        best_f1_reglog_lambda=float(d["best_f1_reglog_lambda"]))
    return curves, summary


def save_grid_search(grid_results: GridSearchResults, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, gammas=np.array(grid_results.gammas), lambdas=np.array(grid_results.lambdas),
        f1_grid=np.array(grid_results.f1_grid), best_gamma=grid_results.best_gamma,
        best_lambda=grid_results.best_lambda, best_f1=grid_results.best_f1)
    print(f"[Saved] {path}")


def load_grid_search(path: str | os.PathLike) -> GridSearchResults:
    d = np.load(path)
    return GridSearchResults(gammas=d["gammas"].tolist(), lambdas=d["lambdas"].tolist(),
        f1_grid=d["f1_grid"].tolist(), best_gamma=float(d["best_gamma"]),
        best_lambda=float(d["best_lambda"]), best_f1=float(d["best_f1"]))


if __name__ == "__main__":
    import os, sys, numpy as np
    from pathlib import Path

    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "implementations.py").exists())
    preproc_path = ROOT / config.SAVE_PREPROCESSED
    npz = np.load(preproc_path)
    X_tr, X_te = npz["X_train"], npz["X_test"]
    y_tr_01 = npz["y_train"]
    train_ids = npz["train_ids"]
    test_ids = npz["test_ids"]
    y_tr_pm1 = metrics.to_pm1_labels(y_tr_01)

    folds = cv_utils.stratified_kfold_indices(y_tr_01, n_splits=5, seed=config.RNG_SEED)
    curves, summary = run_all(X_tr, y_tr_pm1, y_tr_01, folds)

    save_dir = ROOT / config.SAVE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "curves_results.npz"
    save_results(curves, summary, str(out))
    print(f"[Saved] {out}")
    print(summary)

