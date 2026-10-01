import time

import numpy as np
import pytest

from sim import controller as C
from sim import robot as rb
from sim.controller import TwinSim

RNG = np.random.default_rng(0)


# ---- Broyden update (paper eq. 10) -------------------------------------------------

def test_broyden_satisfies_secant_condition():
    for _ in range(50):
        A = RNG.normal(size=(2, 3))
        du = RNG.normal(size=3)
        dx = RNG.normal(size=2)
        A_new = C.broyden_update(A, du, dx, alpha=1.0)
        assert np.allclose(A_new @ du, dx, atol=1e-12)


def test_broyden_is_smallest_change():
    for _ in range(20):
        A = RNG.normal(size=(2, 3))
        du = RNG.normal(size=3)
        dx = RNG.normal(size=2)
        dA = C.broyden_update(A, du, dx) - A
        best = np.linalg.norm(dA)
        # Any other correction satisfying (A + E) du = dx is dA + Z with Z du = 0.
        P = np.eye(3) - np.outer(du, du) / (du @ du)
        for _ in range(200):
            Z = RNG.normal(size=(2, 3)) @ P
            assert np.allclose((A + dA + Z) @ du, dx)
            assert np.linalg.norm(dA + Z) >= best - 1e-12


def test_broyden_alpha_smoothing():
    A = RNG.normal(size=(2, 3))
    du, dx = RNG.normal(size=3), RNG.normal(size=2)
    full = C.broyden_update(A, du, dx, 1.0)
    half = C.broyden_update(A, du, dx, 0.5)
    assert np.allclose(half, 0.5 * (A + full))


# ---- tension-minimising control step (paper eq. 9) --------------------------------

def _random_state():
    y = np.array([RNG.uniform(10, 90), RNG.uniform(1.3, 20), RNG.uniform(1.3, 20)])
    return y


def test_tension_step_constraints():
    solved = 0
    for _ in range(300):
        y = _random_state()
        J = rb.jacobian(y, RNG.uniform(-2, 2), RNG.uniform(0.6, 1.0))
        w = C.column_weights(J)
        A = J * w
        dx_d = RNG.normal(size=2) * 0.8
        dy, info = C.tension_step(A, w, y, dx_d)
        tau_new = rb.tensions(y + dy)
        if info.feasible:
            solved += 1
            assert np.allclose(A @ (dy / w), dx_d, atol=1e-6)
            assert np.all(tau_new >= rb.TAU_MIN - 1e-9)
            assert np.all(np.abs(dy) <= C.DY_MAX + 1e-9)
        assert np.all(y + dy >= rb.Y_LO - 1e-9) and np.all(y + dy <= rb.Y_HI + 1e-9)
    assert solved > 250


def test_tension_step_minimises_tension():
    # Along the null space of A, any other feasible step has at least as much tension.
    y = np.array([40.0, 8.0, 6.0])
    J = rb.jacobian(y, 0.3, 0.85)
    w = C.column_weights(J)
    A = J * w
    dx_d = np.array([0.5, -0.4])
    dy, info = C.tension_step(A, w, y, dx_d)
    assert info.feasible
    cost = np.sum(rb.tensions(y + dy) ** 2)
    n = np.cross(A[0], A[1])
    n /= np.linalg.norm(n)
    for s in np.linspace(-2, 2, 81):
        dy2 = dy + w * (s * n)
        if (np.all(rb.tensions(y + dy2) >= rb.TAU_MIN) and np.all(np.abs(dy2) <= C.DY_MAX)):
            assert np.sum(rb.tensions(y + dy2) ** 2) >= cost - 1e-9
    # Minimum tension relaxes the tendons as far as one step allows: either a
    # tendon reaches tau_min or the relaxation hits the per-step bound.
    at_min = np.isclose(rb.tensions(y + dy).min(), rb.TAU_MIN, atol=1e-6)
    at_step = np.any(np.isclose(dy[1:], -C.DY_MAX[1:], atol=1e-6))
    assert at_min or at_step


def test_tension_step_is_fast():
    y = rb.Y_HOME.copy()
    J = rb.model_jacobian(y)
    w = C.column_weights(J)
    t0 = time.perf_counter()
    for _ in range(200):
        C.tension_step(J * w, w, y, np.array([1.0, 0.5]))
    assert (time.perf_counter() - t0) / 200 < 2e-3


def test_jacobian_matches_finite_difference():
    y = np.array([35.0, 9.0, 3.0])
    for phi in (0.0, 1.0, 2.1):
        J = rb.jacobian(y, phi, 0.85)
        h = 1e-5
        for i in range(3):
            e = np.zeros(3)
            e[i] = h
            fd = (rb.tip(y + e, phi, 0.85) - rb.tip(y - e, phi, 0.85)) / (2 * h)
            assert np.allclose(J[:, i], fd, atol=1e-6)


# ---- closed-loop behaviour -------------------------------------------------------

TARGETS = [(80, 20), (50, -25), (95, 5), (40, 30), (70, -35), (100, -10)]


def _bend_then_reach(phi_deg, target, T=6.0):
    """Initialise at phi = 0, bend while holding still, then reach for a target."""
    s = TwinSim()
    s.reset()
    t0 = s.time
    while s.time < t0 + 2.5:
        f = min(1.0, (s.time - t0) / 2.0)
        s.set_phi(np.radians(phi_deg) * f * f * (3 - 2 * f))
        s.advance(1 / 60)
    s.set_target_local(np.array(target, float))
    t1 = s.time
    reach = {k: None for k in s.rigs}
    final = {}
    while s.time < t1 + T:
        s.advance(1 / 60)
        for k in s.rigs:
            e = s.error(k)
            if e < 1.0 and reach[k] is None:
                reach[k] = s.time - t1
            if e >= 1.0:
                reach[k] = None
            final[k] = e
    return reach, final


@pytest.mark.parametrize("target", TARGETS)
def test_straight_endoscope_both_reach(target):
    reach, final = _bend_then_reach(0, target)
    for k in ("model", "modelless"):
        assert reach[k] is not None and reach[k] < 2.0
        assert final[k] < 0.5


@pytest.mark.parametrize("target", TARGETS)
def test_bend_60_model_slower_modelless_converges(target):
    reach0, _ = _bend_then_reach(0, target)
    reach, final = _bend_then_reach(60, target)
    assert reach["model"] is not None and reach["model"] > reach0["model"]
    assert reach["modelless"] is not None and reach["modelless"] < 2.0
    assert final["modelless"] < 0.5


@pytest.mark.parametrize("target", TARGETS)
def test_bend_120_model_fails_modelless_converges(target):
    reach, final = _bend_then_reach(120, target)
    assert reach["model"] is None and final["model"] > 20.0
    assert reach["modelless"] is not None and reach["modelless"] < 3.0


def test_model_based_is_positive_feedback_at_120():
    # Holding still, the model-based loop at 120 deg is unstable: it runs away and
    # stalls at an actuator limit.
    s = TwinSim()
    s.set_phi(np.radians(120))
    s.reset()
    s.set_target_local(np.array([70.0, 10.0]))
    while s.time < 6.0:
        s.advance(1 / 60)
    r = s.rigs["model"]
    assert s.error("model") > 20.0
    assert np.any(r.plant.y_cmd <= rb.Y_LO + 1e-6) or np.any(r.plant.y_cmd >= rb.Y_HI - 1e-6)
    assert r.positive_feedback > 0.3


def test_square_trajectory_through_bend():
    s = TwinSim()
    s.reset()
    s.toggle_trajectory()
    errs = {"model": [], "modelless": []}
    while s.time < 16.0:
        if s.time > 2.0:
            f = min(1.0, (s.time - 2.0) / 3.0)
            s.set_phi(np.radians(120) * f * f * (3 - 2 * f))
        s.advance(1 / 60)
        if s.time > 8.0:
            for k in errs:
                errs[k].append(s.error(k))
    assert max(errs["modelless"]) < 5.0
    assert np.mean(errs["model"]) > 20.0
