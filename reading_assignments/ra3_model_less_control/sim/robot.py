"""Planar, quasi-static, tendon-driven continuum instrument leaving a bendable endoscope.

All lengths are millimetres, angles radians. World frame: x right, y up, origin at
the centre of a robot panel.

Actuators y = [y_ins, y_L, y_R]
    free length   l     = L0 + y_ins
    bend angle    theta = g * (y_L - y_R) / (2 d)
The model assumes g = 1 and an endoscope bend phi = 0. The true plant uses g (tendon
friction / slack) and the real phi.
"""
from __future__ import annotations

import math

import numpy as np

L0 = 20.0            # free length at zero insertion
D = 5.0              # tendon offset from the backbone
TAU0 = 0.20          # tendon tension at zero pull [N]
K_TENSION = 0.08     # tension stiffness [N / mm]   (paper eq. 8)
TAU_MIN = 0.30       # minimum allowed tension [N]  (paper uses 0.3 N)
G_TRUE = 0.85        # default tendon effectiveness of the true plant

Y_LO = np.array([0.0, 0.0, 0.0])
Y_HI = np.array([100.0, 30.0, 30.0])
Y_SLACK = (TAU_MIN - TAU0) / K_TENSION     # pull at which a tendon sits at tau_min
Y_HOME = np.array([40.0, Y_SLACK, Y_SLACK])

ACT_TAU = 0.035       # actuator first-order lag time constant [s]

# Endoscope: a straight shaft rising from the bottom of the panel, then a bending
# section of length LB whose total bend is phi. PSI0 is the unbent exit direction.
SHAFT_X = 40.0
BEND_START = np.array([SHAFT_X, -20.0])
LB = 28.0
PSI0 = math.pi / 2
ENDO_RADIUS = 6.5    # drawn radius of the endoscope tube


def rot(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def _arc(theta: float):
    """(sin t / t, (1 - cos t) / t) and its derivative, Taylor-safe near 0."""
    t = theta
    if abs(t) < 1e-2:
        t2 = t * t
        a = 1 - t2 / 6 + t2 * t2 / 120
        b = t / 2 - t * t2 / 24 + t * t2 * t2 / 720
        da = -t / 3 + t * t2 / 30
        db = 0.5 - t2 / 8 + t2 * t2 / 144
    else:
        s, c = math.sin(t), math.cos(t)
        a = s / t
        b = (1 - c) / t
        da = (t * c - s) / (t * t)
        db = (t * s - (1 - c)) / (t * t)
    return a, b, da, db


def local_tip(length: float, theta: float) -> np.ndarray:
    a, b, _, _ = _arc(theta)
    return np.array([length * a, length * b])


def local_jacobian(length: float, theta: float) -> np.ndarray:
    """d p_local / d (l, theta)."""
    a, b, da, db = _arc(theta)
    return np.array([[a, length * da], [b, length * db]])


def exit_pose(phi: float):
    """Endoscope exit point E(phi) and exit direction psi."""
    a, b, _, _ = _arc(phi)
    chord = LB * np.array([a, b])
    return BEND_START + rot(PSI0) @ chord, PSI0 + phi


def bend_angle(y: np.ndarray, g: float) -> float:
    return g * (y[1] - y[2]) / (2 * D)


def tip(y: np.ndarray, phi: float, g: float) -> np.ndarray:
    E, psi = exit_pose(phi)
    return E + rot(psi) @ local_tip(L0 + y[0], bend_angle(y, g))


def jacobian(y: np.ndarray, phi: float, g: float) -> np.ndarray:
    """d tip / d y  (2 x 3) for given endoscope bend and tendon effectiveness."""
    length, theta = L0 + y[0], bend_angle(y, g)
    Jl = local_jacobian(length, theta)
    dtheta = g / (2 * D)
    cols = np.column_stack([Jl[:, 0], Jl[:, 1] * dtheta, -Jl[:, 1] * dtheta])
    _, psi = exit_pose(phi)
    return rot(psi) @ cols


def model_jacobian(y: np.ndarray) -> np.ndarray:
    """What the model-based controller believes: straight endoscope, ideal tendons."""
    return jacobian(y, 0.0, 1.0)


def tensions(y: np.ndarray) -> np.ndarray:
    return TAU0 + K_TENSION * np.array([y[1], y[2]])


def backbone(y: np.ndarray, phi: float, g: float, n: int = 48) -> np.ndarray:
    """Dense samples of the instrument's constant-curvature arc, exit to tip."""
    E, psi = exit_pose(phi)
    length, theta = L0 + y[0], bend_angle(y, g)
    R = rot(psi)
    pts = [E + R @ local_tip(length * k / (n - 1), theta * k / (n - 1)) for k in range(n)]
    return np.array(pts)


def endoscope_centerline(phi: float, bottom: float, n: int = 28) -> np.ndarray:
    """Shaft from below the panel up to BEND_START, then the bending section."""
    pts = [np.array([SHAFT_X, bottom]), BEND_START.copy()]
    for k in range(1, n):
        a, b, _, _ = _arc(phi * k / (n - 1))
        pts.append(BEND_START + rot(PSI0) @ (LB * k / (n - 1) * np.array([a, b])))
    return np.array(pts)


# ---- reachable set (for projecting clicked targets) ------------------------------

_REACH_L = np.linspace(L0 + 8.0, L0 + 92.0, 43)
_REACH_T = np.linspace(-2.1, 2.1, 85)


def _reach_grid(g: float) -> np.ndarray:
    pts = []
    for length in _REACH_L:
        for th in _REACH_T:
            pts.append(local_tip(length, th))
    return np.array(pts)


_REACH_CACHE: dict = {}


def project_reachable_local(p_local: np.ndarray, g: float = G_TRUE) -> np.ndarray:
    """Nearest comfortably reachable point in the exit frame."""
    key = round(g, 3)
    if key not in _REACH_CACHE:
        _REACH_CACHE[key] = _reach_grid(g)
    grid = _REACH_CACHE[key]
    i = int(np.argmin(np.sum((grid - p_local) ** 2, axis=1)))
    # Keep the click if it is already inside the sampled region.
    if np.linalg.norm(grid[i] - p_local) < 2.5:
        return np.array(p_local, dtype=float)
    return grid[i].copy()


def reach_outline_local(n: int = 60) -> np.ndarray:
    """Boundary of the comfortable workspace fan, exit frame."""
    lo, hi = _REACH_L[0], _REACH_L[-1]
    t0, t1 = _REACH_T[0], _REACH_T[-1]
    outer = [local_tip(hi, t) for t in np.linspace(t0, t1, n)]
    side1 = [local_tip(l, t1) for l in np.linspace(hi, lo, n // 3)]
    inner = [local_tip(lo, t) for t in np.linspace(t1, t0, n // 2)]
    side0 = [local_tip(l, t0) for l in np.linspace(lo, hi, n // 3)]
    return np.array(outer + side1 + inner + side0)


def world_from_local(p_local: np.ndarray, phi: float) -> np.ndarray:
    E, psi = exit_pose(phi)
    return E + rot(psi) @ p_local


def local_from_world(p: np.ndarray, phi: float) -> np.ndarray:
    E, psi = exit_pose(phi)
    return rot(psi).T @ (np.asarray(p) - E)


class Plant:
    """The real robot: actuators with first-order lag, true phi and g."""

    def __init__(self, g: float = G_TRUE):
        self.g = g
        self.phi = 0.0
        self.y = Y_HOME.copy()       # actual actuator positions (encoders)
        self.y_cmd = Y_HOME.copy()   # commanded positions

    def reset(self):
        self.y = Y_HOME.copy()
        self.y_cmd = Y_HOME.copy()

    def step(self, dt: float):
        a = 1.0 - math.exp(-dt / ACT_TAU)
        self.y = self.y + a * (self.y_cmd - self.y)

    def tip(self) -> np.ndarray:
        return tip(self.y, self.phi, self.g)

    def jacobian(self) -> np.ndarray:
        return jacobian(self.y, self.phi, self.g)
