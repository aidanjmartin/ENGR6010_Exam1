"""Model-based and model-less controllers for the continuum robot.

Both run the same control law (Yip and Camarillo 2014, eq. 9): pick the actuator
step that produces the desired tip motion with the lowest tendon tension, keeping
every tendon above tau_min. The only difference is where the Jacobian comes from:

    model-based : analytic Jacobian of the kinematic model (assumes phi = 0, g = 1)
    model-less  : wiggle initialisation, then Broyden updates (eq. 10)

The constrained solve is exact and closed-form. With 3 actuators and a 2-D tip,
A du = dx_d leaves a one-dimensional family du = u_p + s n (n spans the null space
of A). Every inequality is linear in s, so the feasible set is an interval, and the
quadratic tension cost is minimised by clamping its stationary point to it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sim import robot as rb

CONTROL_HZ = 30.0
PLANT_SUBSTEPS = 4
BETA = 0.30                              # fraction of the error to close per step
STEP_MM = 2.4                            # max desired tip motion per control step
DY_MAX = np.array([3.0, 1.6, 1.6])       # max actuator motion per control step [mm]
EPS_REG = 1e-6
C_MIN = 0.05                             # below this feasible fraction, use the fallback


@dataclass
class StepInfo:
    feasible: bool = True      # the eq. 9 problem had a solution for the full step
    scale: float = 1.0         # fraction of the desired motion that was feasible
    fallback: bool = False     # damped least squares was used
    at_limit: bool = False


def _null_vector(A: np.ndarray):
    n = np.cross(A[0], A[1])
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        return None
    return n / nn


def _constraints(w, y, tau, y_lo, y_hi, dy_max, tau_min):
    """Inequalities G du <= h in normalised actuator units (dy = w * du)."""
    G, h = [], []
    for i in range(3):
        e = np.zeros(3)
        e[i] = w[i]
        G += [e, -e, e, -e]
        h += [dy_max[i], dy_max[i], y_hi[i] - y[i], y[i] - y_lo[i]]
    for j, i in enumerate((1, 2)):
        e = np.zeros(3)
        e[i] = -rb.K_TENSION * w[i]
        G.append(e)
        h.append(tau[j] - tau_min)
    return np.array(G), np.array(h)


def _interval(G, h, up, n, c):
    lo, hi = -np.inf, np.inf
    Gn = G @ n
    rhs = h - c * (G @ up)
    for gi, ri in zip(Gn, rhs):
        if gi > 1e-12:
            hi = min(hi, ri / gi)
        elif gi < -1e-12:
            lo = max(lo, ri / gi)
        elif ri < -1e-9:
            return None
    if lo > hi + 1e-12:
        return None
    return lo, hi


def tension_step(A, w, y, dx_d, y_lo=rb.Y_LO, y_hi=rb.Y_HI, dy_max=DY_MAX,
                 tau_min=rb.TAU_MIN):
    """Solve paper eq. 9 for one control step.

    minimise ||tau + K dy||^2  s.t.  A du = dx_d,  tau + K dy >= tau_min,
                                     |dy| <= dy_max, y_lo <= y + dy <= y_hi
    with dy = w * du and A = J W. Returns (dy, StepInfo).
    """
    w = np.asarray(w, dtype=float)
    tau = rb.tensions(y)
    G, h = _constraints(w, y, tau, y_lo, y_hi, dy_max, tau_min)
    n = _null_vector(A)
    info = StepInfo()
    if n is not None:
        up = A.T @ np.linalg.solve(A @ A.T, dx_d)
        iv = _interval(G, h, up, n, 1.0)
        c = 1.0
        if iv is None:
            if _interval(G, h, up, n, 0.0) is None:
                c = None
            else:
                lo_c, hi_c = 0.0, 1.0
                for _ in range(20):
                    mid = 0.5 * (lo_c + hi_c)
                    if _interval(G, h, up, n, mid) is None:
                        hi_c = mid
                    else:
                        lo_c = mid
                c = lo_c
                iv = _interval(G, h, up, n, c)
                info.feasible = False
        if c is not None and c > C_MIN:
            M = np.zeros((2, 3))
            M[0, 1] = rb.K_TENSION * w[1]
            M[1, 2] = rb.K_TENSION * w[2]
            Mn = M @ n
            s = -(Mn @ (tau + c * (M @ up)) + EPS_REG * c * (n @ up)) / (Mn @ Mn + EPS_REG)
            s = min(max(s, iv[0]), iv[1])
            du = c * up + s * n
            info.scale = c
            dy = w * du
            info.at_limit = bool(np.any(y + dy <= y_lo + 1e-6) or np.any(y + dy >= y_hi - 1e-6))
            return dy, info
    # Fallback: bounded damped least squares, then restore tension by co-activating
    # both tendons (releasing a slack tendon becomes pulling its antagonist). It
    # keeps the robot moving when the estimate claims no feasible step exists, and
    # that motion is what lets the Jacobian update repair the estimate.
    info.feasible, info.fallback = False, True
    lam = 0.15
    du = A.T @ np.linalg.solve(A @ A.T + lam ** 2 * np.eye(2), dx_d)
    dy = np.clip(w * du, -dy_max, dy_max)
    need = max(0.0, (tau_min - rb.tensions(y + dy).min()) / rb.K_TENSION)
    dy[1:] += need
    dy = np.clip(dy, -dy_max - need, dy_max + need)
    dy = np.clip(y + dy, y_lo, y_hi) - y
    du = dy / w
    pred = A @ du
    info.scale = float(np.clip(pred @ dx_d / max(dx_d @ dx_d, 1e-12), 0.0, 1.0))
    info.at_limit = bool(np.any(y + dy <= y_lo + 1e-6) or np.any(y + dy >= y_hi - 1e-6))
    return dy, info


def broyden_update(A, du, dx, alpha=1.0):
    """Smallest Frobenius-norm change making (A + dA) du = dx (paper eq. 10),
    applied with smoothing factor alpha."""
    du = np.asarray(du, dtype=float)
    denom = du @ du
    if denom < 1e-12:
        return A.copy()
    dA = np.outer(dx - A @ du, du) / denom
    return A + alpha * dA


def column_weights(J):
    norms = np.linalg.norm(J, axis=0)
    return 1.0 / np.maximum(norms, 1e-6)


class Controller:
    """One robot's controller. mode is 'model' or 'modelless'."""

    def __init__(self, mode: str):
        self.mode = mode
        self.alpha = 0.5
        self.threshold = 0.5     # tip movement needed before a Jacobian update [mm]
        self.du_min = 0.25       # actuator movement needed as well (normalised)
        self.ratio_max = 2.0     # reject motion the actuators could not have caused
        self.w = np.ones(3)
        self.A = np.zeros((2, 3))
        self.anchor_x = None
        self.anchor_u = None
        self.updates = 0
        self.last_info = StepInfo()
        self.last_dx_d = np.zeros(2)

    # -- Jacobian source --------------------------------------------------------
    def init_model(self, y):
        J = rb.model_jacobian(rb.Y_HOME)
        self.w = column_weights(J)
        self.A = rb.model_jacobian(y) * self.w

    def init_from_columns(self, J_est):
        """Wiggle initialisation: J_est[:, i] = dx / dy_i."""
        self.w = column_weights(J_est)
        self.A = J_est * self.w
        self.anchor_x = None
        self.anchor_u = None
        self.updates = 0

    @staticmethod
    def wiggle_columns(plant: rb.Plant, deltas=(3.0, 2.0, 2.0)):
        """Move each actuator a little from rest and record the tip displacement."""
        y0 = plant.y.copy()
        x0 = rb.tip(y0, plant.phi, plant.g)
        J = np.zeros((2, 3))
        for i, d in enumerate(deltas):
            y1 = y0.copy()
            y1[i] += d
            J[:, i] = (rb.tip(y1, plant.phi, plant.g) - x0) / d
        return J

    def jacobian_estimate(self):
        """Current estimate of J (mm per mm), for drawing."""
        return self.A / self.w

    # -- control ------------------------------------------------------------------
    def update_estimate(self, x_meas, y_act):
        if self.mode != "modelless":
            return False
        u = y_act / self.w
        if self.anchor_x is None:
            self.anchor_x, self.anchor_u = x_meas.copy(), u.copy()
            return False
        dx = x_meas - self.anchor_x
        du = u - self.anchor_u
        ndx, ndu = np.linalg.norm(dx), np.linalg.norm(du)
        if ndx <= self.threshold:
            return False
        if ndu < self.du_min or ndx > self.ratio_max * ndu * max(np.linalg.norm(self.A, 2), 0.5):
            # The tip moved without the actuators moving enough to explain it
            # (the endoscope bent under us): restart the difference window.
            self.anchor_x, self.anchor_u = x_meas.copy(), u.copy()
            return False
        self.A = broyden_update(self.A, du, dx, self.alpha)
        self.anchor_x, self.anchor_u = x_meas.copy(), u.copy()
        self.updates += 1
        return True

    def control(self, plant: rb.Plant, x_meas, target):
        if self.mode == "model":
            self.A = rb.model_jacobian(plant.y) * self.w
        else:
            self.update_estimate(x_meas, plant.y)
        e = target - x_meas
        dx_d = BETA * e
        nrm = np.linalg.norm(dx_d)
        if nrm > STEP_MM:
            dx_d *= STEP_MM / nrm
        self.last_dx_d = dx_d
        dy, info = tension_step(self.A, self.w, plant.y_cmd, dx_d)
        plant.y_cmd = np.clip(plant.y_cmd + dy, rb.Y_LO, rb.Y_HI)
        self.last_info = info
        return info


@dataclass
class Rig:
    """A plant, its controller, and its camera."""
    mode: str
    plant: rb.Plant = field(default_factory=rb.Plant)
    ctrl: Controller = None
    x_meas: np.ndarray = None
    err_hist: list = field(default_factory=list)
    trail: list = field(default_factory=list)
    positive_feedback: float = 0.0   # smoothed indicator: tip moving away from target

    def __post_init__(self):
        self.ctrl = Controller(self.mode)
        self.reset()

    def reset(self, init=True):
        self.plant.reset()
        self.x_meas = self.plant.tip()
        self.err_hist = []
        self.trail = []
        self.positive_feedback = 0.0
        if init:
            if self.mode == "model":
                self.ctrl.init_model(self.plant.y)
            else:
                self.ctrl.init_from_columns(Controller.wiggle_columns(self.plant))


class TwinSim:
    """Both robots side by side, sharing the endoscope bend, targets and noise."""

    def __init__(self, seed=3):
        self.rigs = {"model": Rig("model"), "modelless": Rig("modelless")}
        self.rng = np.random.default_rng(seed)
        self.sigma = 0.0
        self.phi = 0.0
        self.g = rb.G_TRUE
        self.target_local = None
        self.traj_on = False
        self.traj_t = 0.0
        self.traj_speed = 16.0       # mm/s along the square
        self.square_center = np.array([55.0, 0.0])
        self.square_half = 16.0
        self.running = True
        self.time = 0.0
        self._acc = 0.0
        self._sub = 0
        self._phi_tick = 0.0
        self.set_target_local(rb.local_tip(rb.L0 + rb.Y_HOME[0], 0.0))

    # -- scene state --------------------------------------------------------------
    def reset(self):
        for r in self.rigs.values():
            r.plant.phi = self.phi
            r.plant.g = self.g
            r.reset()
        self.traj_on = False
        self.traj_t = 0.0
        self.set_target_local(rb.local_tip(rb.L0 + rb.Y_HOME[0], 0.0))

    def set_phi(self, phi):
        self.phi = phi
        for r in self.rigs.values():
            r.plant.phi = phi

    def set_g(self, g):
        self.g = g
        for r in self.rigs.values():
            r.plant.g = g

    def set_target_local(self, p_local):
        self.target_local = np.array(p_local, dtype=float)

    def click_world(self, p_world):
        loc = rb.local_from_world(p_world, self.phi)
        self.set_target_local(rb.project_reachable_local(loc))
        self.traj_on = False

    def target_world(self):
        return rb.world_from_local(self.target_local, self.phi)

    def square_local(self, t):
        """Point on the square trajectory (exit frame) after travelling t seconds."""
        h = self.square_half
        corners = [np.array(c) for c in ((-h, -h), (h, -h), (h, h), (-h, h))]
        side = 2 * h
        s = (t * self.traj_speed) % (4 * side)
        k = int(s // side)
        f = (s - k * side) / side
        a, b = corners[k], corners[(k + 1) % 4]
        return self.square_center + a + f * (b - a)

    def square_outline_local(self):
        h = self.square_half
        return [self.square_center + np.array(c) for c in ((-h, -h), (h, -h), (h, h), (-h, h))]

    def toggle_trajectory(self):
        self.traj_on = not self.traj_on
        if self.traj_on:
            self.traj_t = 0.0

    # -- stepping -----------------------------------------------------------------
    def advance(self, dt):
        if not self.running:
            return
        dt = min(dt, 0.1)
        self._acc += dt
        h = 1.0 / (CONTROL_HZ * PLANT_SUBSTEPS)
        while self._acc >= h:
            self._acc -= h
            self.time += h
            for r in self.rigs.values():
                r.plant.step(h)
            self._sub += 1
            if self._sub >= PLANT_SUBSTEPS:
                self._sub = 0
                self._control_tick(1.0 / CONTROL_HZ)

    def _control_tick(self, dt):
        # The square pauses while the endoscope is bending: the surgeon holds the
        # instrument still, so the target stays put in the exit frame.
        bending = abs(self.phi - self._phi_tick) > 1e-5
        self._phi_tick = self.phi
        if self.traj_on and not bending:
            self.traj_t += dt
            self.set_target_local(self.square_local(self.traj_t))
        target = self.target_world()
        for r in self.rigs.values():
            x_true = r.plant.tip()
            noise = self.rng.normal(0.0, self.sigma, 2) if self.sigma > 0 else np.zeros(2)
            prev = r.x_meas
            r.x_meas = x_true + noise
            r.ctrl.control(r.plant, r.x_meas, target)
            err = float(np.linalg.norm(target - x_true))
            r.err_hist.append(err)
            if len(r.err_hist) > 360:
                del r.err_hist[:-360]
            r.trail.append(x_true.copy())
            if len(r.trail) > 75:
                del r.trail[:-75]
            # Positive feedback: tip moving away from the target it was commanded toward.
            moved = r.x_meas - prev
            want = r.ctrl.last_dx_d
            pf = 0.0
            if err > 6.0 and np.linalg.norm(moved) > 0.2 and np.linalg.norm(want) > 0.2:
                pf = 1.0 if (moved @ want) < 0 else 0.0
            if r.ctrl.last_info.at_limit and err > 10.0:
                pf = max(pf, 0.7)
            r.positive_feedback += 0.08 * (pf - r.positive_feedback)

    def error(self, key):
        r = self.rigs[key]
        return float(np.linalg.norm(self.target_world() - r.plant.tip()))
