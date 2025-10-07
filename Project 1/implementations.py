import numpy as np
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


def logistic_loss(y, tx, w, lambda_=0):
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


## Additional function needed taken from lab 2#####


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
