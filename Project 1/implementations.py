import numpy as np
from helpers import *

import config

# -------------- CORE IMPLEMENTATIONS --------------

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
    for _ in range(max_iters):
        w -= gamma * logistic_gradient(y, tx, w)
    return w, logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    "hey"
    w = initial_w
    for _ in range(max_iters):
        w -= gamma * logistic_gradient(y, tx, w, lambda_=lambda_)
    return w, logistic_loss(y, tx, w, lambda_=0)


## Additional function computed#####


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_loss(y, tx, w, lambda_ = 0):
    sig = sigmoid(tx.dot(w))
    eps = 1e-10
    loss = -np.mean(y * np.log(sig + eps) + (1 - y) * np.log(1 - sig + eps))
    if lambda_ > 0:
        loss += lambda_ * np.sum(w**2)
    return loss


def logistic_gradient(y, tx, w, lambda_=0):
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

# -------------- FURTHER IMPROVEMENTS (ADAM AND ADDITIONAL LOSS FUNCTIONS) --------------

def sigmoid_stable(z):
    # Sanitization
    z = np.asarray(z, dtype = np.float64)

    # Initializing output
    out = np.empty_like(z)

    # Computing indices where z is positive
    pos = z >= 0

    # Where positive, compute sigmoid (no overflow risk)
    out[pos]  = 1.0 / (1.0 + np.exp(-z[pos]))

    # Where negative, compute e / (1 + e)
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)

    return out

def class_weights(y):
    """
    Balanced class weights ala N/(2*N_c).
    """

    # Sanitization
    y = np.asarray(y).astype(int)

    # Computing the amount of samples per classes
    n = len(y)
    n_pos = np.sum(y == 1)
    n_neg = n - n_pos

    # Guard against edge cases
    if n_pos == 0 or n_neg == 0:
        return 1.0, 1.0

    # Computing and returning the weights
    alpha_pos = n / (2.0 * n_pos)
    alpha_neg = n / (2.0 * n_neg)
    return float(alpha_pos), float(alpha_neg)

def weighted_logistic_loss(y, tx, w, alpha_pos = 1.0, alpha_neg = 1.0, lambda_ = 0.0, eps = 1e-10):
    """
    Weighted binary cross-entropy (cost-sensitive BCE).
    """

    z = tx.dot(w)
    p = sigmoid_stable(z)

    # Creating per-class weights (vector sized [N,] containing the weight for each sample based on their class)
    a = np.where(y == 1, alpha_pos, alpha_neg).astype(p.dtype)

    # Computing the weighted loss ( epsilon to avoid log(0) )
    loss = -np.mean(a * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    # L2 penalty
    if lambda_ > 0:
        loss += lambda_ * np.sum(w**2)

    return loss

def weighted_logistic_gradient(y, tx, w, alpha_pos = 1.0, alpha_neg = 1.0, lambda_ = 0.0):
    """
    Weighted logistic gradient.
    """
    # Computing logits and probabilities
    z = tx.dot(w)
    p = sigmoid_stable(z)

    # Creating per-class weights
    a = np.where(y == 1, alpha_pos, alpha_neg).astype(p.dtype)

    # Computing the gradient vector
    grad = tx.T.dot(a * (p - y)) / len(y)

    # L2 penalty
    if lambda_ > 0:
        grad += 2.0 * lambda_ * w
    return grad

def focal_logistic_loss(y, tx, w, alpha=0.5, gamma=2.0, lambda_=0.0, eps=1e-10):
    """
    Focal loss (binary). alpha in [0,1]; gamma >= 0.
    """
    p = sigmoid_stable(tx.dot(w))
    # pt = p when y=1; pt = 1-p when y=0
    pt = y * p + (1 - y) * (1 - p)
    # alpha weighting by class
    alpha_t = y * alpha + (1 - y) * (1 - alpha)
    loss = -np.mean(alpha_t * ((1 - pt) ** gamma) *
                    (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    if lambda_ > 0:
        loss += lambda_ * np.sum(w**2)
    return loss

def focal_logistic_gradient(y, tx, w, alpha=0.5, gamma=2.0, lambda_=0.0, eps=1e-10):
    """
    Gradient of focal loss (binary). Derived w.r.t. logits via chain rule.
    """
    z = tx.dot(w)
    p = sigmoid_stable(z)
    pt = y * p + (1 - y) * (1 - p)
    alpha_t = y * alpha + (1 - y) * (1 - alpha)

    # dL/dp for binary focal
    # L = -alpha_t * (1-pt)^gamma * [ y log p + (1-y) log(1-p) ]
    # Using known derivative form:
    g_factor = alpha_t * ((1 - pt) ** gamma) * (gamma * ( (y - p) / (1 - pt + eps) ))
    bce_grad = (p - y)
    dL_dp = g_factor + alpha_t * ((1 - pt) ** gamma) * (bce_grad / (p * (1 - p) + eps))

    # dL/dz = dL/dp * dp/dz = dL/dp * p*(1-p)
    dL_dz = dL_dp * p * (1 - p)
    grad = tx.T.dot(dL_dz) / len(y)

    if lambda_ > 0:
        grad += 2.0 * lambda_ * w
    return grad


# ===== Adam optimizer for logistic variants =====

def adam_logistic(
    y, tx, initial_w, max_iters, gamma = 1e-3, lambda_ = 0.0,
    # Weighted BCE params (auto-balance if None)
    alpha_pos = None, alpha_neg = None,
    # Focal params
    focal_alpha = 0.5, focal_gamma = 2.0,
):
    """
    Adam applied to logistic regression with a pluggable loss:
      - 'weighted_bce' (default) or
      - 'focal'
    Returns (w, unpenalized BCE loss) to match your project reporting.
    """
    rng = np.random.default_rng(config.RNG_SEED)
    w = np.array(initial_w, dtype=np.float64, copy=True)

    eps = 1e-8

    # Initialization of loss and gradient function references
    if config.ADAM_LOSS_TYPE == "weighted_bce":
        if alpha_pos is None or alpha_neg is None:
            alpha_pos, alpha_neg = class_weights(y)
        def loss_function(y_, tx_, w_):
            return weighted_logistic_loss(y_, tx_, w_, alpha_pos, alpha_neg, lambda_)
        def gradient_function(y_, tx_, w_):
            return weighted_logistic_gradient(y_, tx_, w_, alpha_pos, alpha_neg, lambda_)
    elif config.ADAM_LOSS_TYPE == "focal":
        def loss_function(y_, tx_, w_):
            return focal_logistic_loss(y_, tx_, w_, focal_alpha, focal_gamma, lambda_)
        def gradient_function(y_, tx_, w_):
            return focal_logistic_gradient(y_, tx_, w_, focal_alpha, focal_gamma, lambda_)
    else:
        raise ValueError("loss_type must be 'weighted_bce' or 'focal'")

    # Initialization of ADAM's momentum and adaptive step size
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    t = 0

    # Auxiliary function that generates sample batches
    def next_batch():
        # Case in which entire dataset is used
        if config.ADAM_BATCH_SIZE is None or config.ADAM_BATCH_SIZE >= len(y):
            return y, tx

        # Case in which batches are used
        idx = rng.choice(len(y), size = config.ADAM_BATCH_SIZE, replace = False)
        return y[idx], tx[idx]

    # Training loop
    for _ in range(max_iters):
        # Generating batch
        y_b, tx_b = next_batch()

        # Generating gradient vector
        g = gradient_function(y_b, tx_b, w)

        # Increasing time step
        t += 1

        # Updating momentum and step size
        m = config.ADAM_BETA_1 * m + (1 - config.ADAM_BETA_1) * g
        v = config.ADAM_BETA_2 * v + (1 - config.ADAM_BETA_2) * (g * g)

        # Compensation for zero initialization, useful in early step when m and v are zeros
        mhat = m / (1 - config.ADAM_BETA_1 ** t)
        vhat = v / (1 - config.ADAM_BETA_2 ** t)

        # Learning rate decay
        if config.ADAM_LR_DECAY == "sqrt":
            gamma_t = gamma / np.sqrt(t)
        elif config.ADAM_LR_DECAY == "exp":
            gamma_t = gamma * (0.96 ** (t / 100))
        elif config.ADAM_LR_DECAY == "cos":
            gamma_t = 0.5 * gamma * (1 + np.cos(np.pi * t / max_iters))
        else:
            gamma_t = gamma

        # Updating weights
        w -= gamma * mhat / (np.sqrt(vhat) + eps)

    # Computing loss value
    loss = loss_function(y, tx, w)
    return w, loss
