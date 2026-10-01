import numpy as np
import pytest

from sim import optim


@pytest.mark.parametrize("gamma", [2.0, 5.0, 25.0, 50.0])
def test_gd_exact_line_search_rate(gamma):
    # From the worst-case start (gamma, 1) the error shrinks by exactly
    # (gamma - 1) / (gamma + 1) every step (Boyd and Vandenberghe, ch. 9).
    x0 = np.array([gamma, 1.0])
    path = optim.gradient_descent(gamma, x0=x0, max_steps=12)
    H = optim.hessian(gamma)
    fvals = np.array([0.5 * x @ H @ x for x in path])
    ratios = np.sqrt(fvals[1:] / fvals[:-1])
    assert np.allclose(ratios, optim.gd_rate(gamma), rtol=1e-9)


@pytest.mark.parametrize("gamma", [1.0, 3.0, 25.0, 50.0])
def test_newton_one_step(gamma):
    path = optim.newton(gamma)
    assert len(path) == 2
    assert np.linalg.norm(path[-1]) < 1e-12


@pytest.mark.parametrize("gamma", [1.0, 2.0, 10.0, 25.0, 50.0])
def test_bfgs_converges_in_three(gamma):
    path, Bs, ss, ys = optim.bfgs(gamma)
    assert optim.steps_to_converge(path) <= 3
    assert np.linalg.norm(path[-1]) < optim.TOL
    for B, s, y in zip(Bs[1:], ss, ys):
        assert np.allclose(B @ s, y)          # secant condition after every update


def test_demo_start_is_interesting():
    # The ravine beat needs many zigzag steps; everything must stay on the plot.
    assert optim.steps_to_converge(optim.gradient_descent(25.0)) > 30
    for gamma in (1.0, 5.0, 25.0, 50.0):
        for path in (optim.gradient_descent(gamma), optim.bfgs(gamma)[0]):
            assert np.abs(path).max() <= 10.0
