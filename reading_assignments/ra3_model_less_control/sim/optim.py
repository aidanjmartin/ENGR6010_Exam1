"""Gradient descent, Newton and BFGS on the 2-D quadratic f(x) = 1/2 (x1^2 + gamma x2^2).

Every routine returns the full list of iterates so the acts can animate them.
BFGS also returns its curvature estimate B after every step.
"""
from __future__ import annotations

import numpy as np

# Start point used by Acts 1 and 2. Far along the shallow axis and a little off
# it, so gradient descent zigzags visibly and every iterate stays inside the
# [-10, 10]^2 plot for any gamma in [1, 50].
START = np.array([-8.8, 1.2])
TOL = 1e-3
MAX_STEPS = 200


def hessian(gamma: float) -> np.ndarray:
    return np.diag([1.0, float(gamma)])


def f(x: np.ndarray, gamma: float) -> float:
    return 0.5 * (x[0] ** 2 + gamma * x[1] ** 2)


def grad(x: np.ndarray, gamma: float) -> np.ndarray:
    return np.array([x[0], gamma * x[1]])


def gradient_descent(gamma: float, x0=START, tol=TOL, max_steps=MAX_STEPS):
    """Steepest descent with exact line search t = g'g / g'Hg."""
    H = hessian(gamma)
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    for _ in range(max_steps):
        if np.linalg.norm(x) < tol:
            break
        g = H @ x
        t = (g @ g) / (g @ H @ g)
        x = x - t * g
        path.append(x.copy())
    return np.array(path)


def newton(gamma: float, x0=START, tol=TOL, max_steps=MAX_STEPS):
    """x <- x - H^{-1} grad f. Exact in one step on a quadratic."""
    H = hessian(gamma)
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    for _ in range(max_steps):
        if np.linalg.norm(x) < tol:
            break
        x = x - np.linalg.solve(H, H @ x)
        path.append(x.copy())
    return np.array(path)


def bfgs_update(B: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Standard BFGS update of the Hessian estimate. Satisfies B_new s = y."""
    sy = s @ y
    if sy <= 1e-12:
        return B
    Bs = B @ s
    return B - np.outer(Bs, Bs) / (s @ Bs) + np.outer(y, y) / sy


def bfgs(gamma: float, x0=START, tol=TOL, max_steps=MAX_STEPS):
    """BFGS with exact line search along p = -B^{-1} grad f, B0 = I.

    Returns (path, Bs, ss, ys): iterates, the estimate B_k used at each iterate,
    and the step / gradient-change pairs that drove each update.
    """
    H = hessian(gamma)
    x = np.array(x0, dtype=float)
    B = np.eye(2)
    path, Bs, ss, ys = [x.copy()], [B.copy()], [], []
    for _ in range(max_steps):
        if np.linalg.norm(x) < tol:
            break
        g = H @ x
        p = -np.linalg.solve(B, g)
        t = -(g @ p) / (p @ H @ p)
        s = t * p
        x_new = x + s
        y = H @ x_new - g
        B = bfgs_update(B, s, y)
        x = x_new
        path.append(x.copy())
        Bs.append(B.copy())
        ss.append(s)
        ys.append(y)
    return np.array(path), Bs, ss, ys


def steps_to_converge(path: np.ndarray) -> int:
    return len(path) - 1


def gd_rate(gamma: float) -> float:
    """Worst-case per-step error contraction of exact-line-search descent."""
    return (gamma - 1.0) / (gamma + 1.0)
