import numpy as np
import config
from helpers import *


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    "hey"
    ws = [initial_w]
    w = initial_w
    loss = compute_loss(y, tx, w)

    for _ in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w -= gamma * grad
        # store w and loss
        ws.append(w)
        # losses.append(loss)
    loss = compute_loss(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    "hey"
    w = initial_w
    ws = [w]
    loss = compute_loss(y, tx, w)

    for _ in range(max_iters):
        idx = np.random.randint(0, len(y))
        y_b = y[idx]
        tx_b = tx[idx]

        grad = compute_gradient(y_b, tx_b, w, sgd=True)
        w -= gamma * grad

        # loss = compute_loss(y, tx, w)
        # losses.append(loss)
        ws.append(w)
    loss = compute_loss(y, tx, w)

    return w, loss


def least_squares(y, tx):
    "hey"
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)  # w=a^-1 x b
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    "hey"
    N, D = tx.shape
    A = tx.T.dot(tx) + 2 * N * lambda_ * np.identity(D)
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)
    loss = compute_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    "hey"
    w = initial_w
    use_weighted = getattr(config, "USE_WEIGHTED_BCE", False)
    if use_weighted:
        # class-balanced weights for y in {0,1}
        n_pos = float(np.sum(y))
        n_tot = float(y.size)
        n_neg = n_tot - n_pos
        a_pos = n_tot / (2.0 * max(1.0, n_pos))
        a_neg = n_tot / (2.0 * max(1.0, n_neg))
        w_samp = (y * a_pos + (1.0 - y) * a_neg).astype(np.float32, copy=False)
        denom_w = float(np.sum(w_samp))
        for _ in range(max_iters):
            p = sigmoid(tx.dot(w))
            resid = (p - y)
            grad = tx.T.dot(resid * w_samp) / denom_w
            w -= gamma * grad
    else:
        for _ in range(max_iters):
            w -= gamma * logistic_gradient(y, tx, w)
    return w, logistic_loss(y, tx, w)


# def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
#     "hey"
#     w = initial_w
#     for _ in range(max_iters):
#         w -= gamma * logistic_gradient(y, tx, w, lambda_=lambda_)
#     return w, logistic_loss(y, tx, w, lambda_=0)

def reg_logistic_regression(
    y,
    tx,
    lambda_,
    initial_w,
    max_iters,
    gamma,
    display=False,
    adam=True,
    schedule=None,
    early_stopping=False,
    patience=10,
    tol=1e-6,
    verbose=False,
    callback=None,
):
    """Regularized logistic regression (L2) with options for Adam, LR schedule, and early stopping.

    Extras (backward-compatible):
    - If config.USE_WEIGHTED_BCE is True, use class-balanced weights in the BCE gradient
      (alpha_pos = N/(2*N_pos), alpha_neg = N/(2*N_neg)). Penalty stays as lambda * sum(w**2).
    - Signature unchanged; behavior controlled via config only.
    """
    rng = np.random.RandomState(config.RNG_SEED)
    w = initial_w.astype(np.float32, copy=True)

    # Adam buffers (used if adam=True)
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    b1, b2, eps = 0.9, 0.999, 1e-8

    best_loss = np.inf
    best_w = w.copy()
    wait = 0

    n = y.size
    if (verbose or display):
        print(
            f"[Train] adam={adam} schedule={'on' if schedule else 'none'} early_stop={early_stopping} "
            f"lambda={lambda_:.3e} gamma={gamma:.3e} iters={max_iters}"
        )

    # Optional class-balanced weighting (no API change: toggled via config)
    use_weighted = getattr(config, "USE_WEIGHTED_BCE", False)
    if use_weighted:
        # y expected in {0,1}
        n_pos = float(np.sum(y))
        n_tot = float(y.size)
        n_neg = n_tot - n_pos
        # avoid division by zero in extreme edge-cases
        a_pos = n_tot / (2.0 * max(1.0, n_pos))
        a_neg = n_tot / (2.0 * max(1.0, n_neg))
        w_samp = (y * a_pos + (1.0 - y) * a_neg).astype(np.float32, copy=False)
        denom_w = float(np.sum(w_samp))
    else:
        w_samp = None
        denom_w = float(y.size)

    for t in range(1, max_iters + 1):
        lr = schedule(gamma, t - 1, max_iters) if schedule else gamma
        y_b = y
        tx_b = tx

        # plain logistic gradient
        p = sigmoid(tx_b.dot(w))
        resid = (p - y_b)
        if use_weighted:
            # Weighted BCE gradient: X^T((p - y) * w_i) / sum(w_i)
            grad = tx_b.T.dot(resid * w_samp) / denom_w
        else:
            grad = tx_b.T.dot(resid) / y_b.size
        # add L2 penalty (on all weights; bias included)
        g_reg = grad.copy()
        g_reg += 2.0 * lambda_ * w

        if adam:
            m = b1 * m + (1 - b1) * g_reg
            v = b2 * v + (1 - b2) * (g_reg * g_reg)
            m_hat = m / (1 - b1**t)
            v_hat = v / (1 - b2**t)
            w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
        else:
            w = w - lr * g_reg

        # Monitor unpenalized (unweighted) loss for consistency with evaluation
        cur_loss = logistic_loss(y, tx, w, lambda_=0)
        if callback is not None:
            try:
                callback(t, w, float(cur_loss), float(lr))
            except Exception:
                pass

        if early_stopping:
            if cur_loss + tol < best_loss:
                best_loss = cur_loss
                best_w = w.copy()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose or display:
                        print(f"[EarlyStop] iter={t} best_loss={best_loss:.6f}")
                    w = best_w
                    break

        if (verbose or display) and (t % max(1, max_iters // 10) == 0):
            pen = logistic_loss(y, tx, w, lambda_=lambda_)
            print(f"[Iter {t}/{max_iters}] unpen={cur_loss:.6f} pen={pen:.6f}")

    final_loss = logistic_loss(y, tx, w, lambda_=0)
    return w, final_loss



## Additional function computed#####


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_loss(y, tx, w, lambda_=0):
    sig = sigmoid(tx.dot(w))
    eps = 1e-12
    loss = -np.mean(y * np.log(sig + eps) + (1 - y) * np.log(1 - sig + eps))
    if lambda_ > 0:
        loss += lambda_ * np.sum(w**2)
        # Do not penalize bias term
        #loss += lambda_ * np.sum(w[1:] ** 2)
    return loss


def class_weights(y):
    """Return class-balanced weights (alpha_pos, alpha_neg) for y in {0,1}.
    alpha_c = N / (2 * N_c)."""
    n_pos = float(np.sum(y))
    n_tot = float(y.size)
    n_neg = n_tot - n_pos
    a_pos = n_tot / (2.0 * max(1.0, n_pos))
    a_neg = n_tot / (2.0 * max(1.0, n_neg))
    return a_pos, a_neg


def weighted_logistic_loss(y, tx, w, weights, lambda_=0):
    """Weighted BCE loss with optional L2 penalty over all weights (bias included).
    - y in {0,1}, weights >= 0 with shape (N,)
    - Normalizes by sum(weights) to be scale-invariant"""
    sig = sigmoid(tx.dot(w))
    eps = 1e-12
    weights = weights.astype(np.float64, copy=False)
    denom = np.sum(weights) if np.ndim(weights) == 1 else float(len(y))
    # weighted average
    loss = -np.sum(weights * (y * np.log(sig + eps) + (1 - y) * np.log(1 - sig + eps))) / max(1e-12, denom)
    if lambda_ > 0:
        loss += lambda_ * np.sum(w**2)
    return float(loss)


def logistic_gradient(y, tx, w, lambda_=0):
    """Plain logistic gradient; optional L2 on all weights when lambda_>0 (kept for backward compat).
    Note: reg_logistic_regression handles L2 without penalizing bias; callers relying on this helper
    should be aware this version penalizes all weights when lambda_>0.
    """
    grad = tx.T.dot(sigmoid(tx.dot(w)) - y) / len(y)
    if lambda_ > 0:
        grad += 2 * lambda_ * w
    return grad


## Additional function needed taken from lab 2


def compute_loss(y, tx, w):
    err = y - tx.dot(w)
    return 0.5 * np.mean(err**2)  # np.mean(np.abs(err)) for MAE


def compute_gradient(y, tx, w, sgd=False):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    err = y - tx.dot(w)
    if sgd:
        grad = -(tx.T.dot(err))
    else:
        grad = -(tx.T.dot(err)) / len(y)
    return grad
