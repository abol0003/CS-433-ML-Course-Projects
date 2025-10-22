import numpy as np
from tqdm.auto import trange


def nagfree(
    x0,
    g,
    maxit,
    tol=1e-6,
    L_max=1e10,
    L_init=None,
    restart_threshold=1.05,
    track_history=False,
    disable_tqdm=False,
):
    """
    NAG-Free optimizer with adaptive smoothness and stability safeguards.

    Uses an adaptive estimate of the Lipschitz constant and Nesterov-like momentum
    without requiring explicit learning rates. Ensures numerical stability through
    clipping and optional diagnostics tracking.

    Args:
        x0: Initial parameter vector.
        g: Callable returning the gradient at a given point.
        maxit: Maximum number of iterations.
        tol: Convergence tolerance based on the gradient norm.
        L_max: Upper bound on the Lipschitz constant to prevent instability.
        L_init: Optional initial Lipschitz estimate; if None, estimated automatically.
        restart_threshold: Reserved for future restart logic (unused here).
        track_history: If True, returns diagnostics arrays in addition to the final weights.
        disable_tqdm: If True, disables progress display.

    Returns:
        If track_history is False:
            Final optimized vector.
        If track_history is True:
            Tuple (final vector, history dict) where history includes:
                'grad_norms', 'L_estimates', 'm_estimates', 'alphas', 'iterations'.
    """
    delta = 1e-6

    if L_init is None:
        y_probe = x0 + delta * np.random.randn(*x0.shape)
        gk = g(x0)
        grad_diff = np.linalg.norm(gk - g(y_probe))
        point_diff = np.linalg.norm(x0 - y_probe)
        L_init = max(grad_diff / (point_diff + 1e-15), 1e-8)
    else:
        gk = g(x0)

    xk, yk = x0.copy(), x0.copy()
    Lk, mk = L_init, L_init

    history = {
        "grad_norms": [],
        "L_estimates": [],
        "m_estimates": [],
        "alphas": [],
        "iterations": [],
    } if track_history else None

    t = trange(maxit, desc="Training NAG-Free", unit="iter", disable=disable_tqdm)
    for iter_num in t:
        Lk = np.clip(Lk, 1e-10, L_max)
        xk_next = yk - (1 / Lk) * gk
        sqrt_L = np.sqrt(Lk)
        sqrt_m = np.sqrt(mk)
        ak = (sqrt_L - sqrt_m) / (sqrt_L + sqrt_m + 1e-15)
        ak = np.clip(ak, 0, 0.999)
        yk_next = xk_next + ak * (xk_next - xk)
        gk_next = g(yk_next)
        dy = np.linalg.norm(yk_next - yk)
        if dy > 1e-12:
            dg = np.linalg.norm(gk_next - gk)
            ck = dg / dy
            Lk = np.clip(max(Lk, ck), 1e-10, L_max)
            mk = min(mk, ck) if ck > 1e-12 else mk
        xk, yk, gk = xk_next, yk_next, gk_next
        err = np.linalg.norm(gk)

        if track_history:
            history["grad_norms"].append(err)
            history["L_estimates"].append(Lk)
            history["m_estimates"].append(mk)
            history["alphas"].append(ak)
            history["iterations"].append(iter_num)

        if err < tol:
            if not disable_tqdm:
                t.set_postfix({"‖∇f‖": f"{err:.3e}", "L": f"{Lk:.2e}", "μ": f"{mk:.2e}", "α": f"{ak:.3f}", "status": "✓"})
            break

        if not disable_tqdm:
            t.set_postfix({"‖∇f‖": f"{err:.3e}", "L": f"{Lk:.2e}", "μ": f"{mk:.2e}", "α": f"{ak:.3f}"})

    if track_history:
        for key in history:
            history[key] = np.array(history[key])
        return xk, history

    return xk
