"""
The GrabCut GMM, written the way the paper describes it.

This is deliberately *not* a soft-EM mixture model. Rother et al. introduce an
extra variable k_n - "the GMM component that pixel n comes from" - into the
energy itself, which turns the mixture fit into hard assignment:

    1. k_n  <- argmin_k  D(alpha_n, k, theta, z_n)      (assign)
    2. theta <- mean/cov/weight of the pixels in each k  (refit)

Both steps can only lower E, which is the whole reason the outer loop is
guaranteed to converge. OpenCV's grabcut.cpp does exactly this too.

D(alpha_n, k_n, theta, z_n) = -log pi_k
                              + (1/2) log det Sigma_k
                              + (1/2) (z - mu_k)^T Sigma_k^-1 (z - mu_k)
"""

import numpy as np

REG = 1e-4          # covariance ridge; RGB is scaled to [0,1] so this is small


def _kmeanspp(X, K, rng):
    """k-means++ seeding - keeps the 5 components from collapsing onto each other."""
    centers = [X[rng.integers(len(X))]]
    for _ in range(1, K):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        total = d2.sum()
        if total <= 0:
            centers.append(X[rng.integers(len(X))])
        else:
            centers.append(X[rng.choice(len(X), p=d2 / total)])
    return np.array(centers)


def kmeans(X, K, rng, iters=25):
    C = _kmeanspp(X, K, rng)
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        new = d2.argmin(1)
        if np.array_equal(new, lab):
            break
        lab = new
        for k in range(K):
            m = lab == k
            if m.any():
                C[k] = X[m].mean(0)
            else:                                    # revive a dead component
                C[k] = X[rng.integers(len(X))]
    return lab


class GrabCutGMM:
    """K full-covariance Gaussians in RGB, fit by hard assignment."""

    def __init__(self, K=5, seed=0):
        self.K = K
        self.rng = np.random.default_rng(seed)
        self.weights = np.full(K, 1.0 / K)
        self.means = np.zeros((K, 3))
        self.covs = np.tile(np.eye(3) * 0.05, (K, 1, 1))
        self._refresh()

    # -- internals ----------------------------------------------------------
    def _refresh(self):
        self.inv = np.linalg.inv(self.covs)
        sign, logdet = np.linalg.slogdet(self.covs)
        self.logdet = logdet
        self.logw = np.log(np.maximum(self.weights, 1e-12))

    def _fit_components(self, X, lab):
        K, d = self.K, X.shape[1]
        w = np.zeros(K)
        mu = np.zeros((K, d))
        cov = np.zeros((K, d, d))
        for k in range(K):
            m = lab == k
            n = int(m.sum())
            if n < d + 1:                            # too small to estimate a cov
                w[k] = max(n, 1) / max(len(X), 1)
                mu[k] = X[m].mean(0) if n else X.mean(0)
                cov[k] = np.eye(d) * 0.02
                continue
            Xk = X[m]
            w[k] = n / len(X)
            mu[k] = Xk.mean(0)
            cov[k] = np.cov(Xk.T) + np.eye(d) * REG
        w = np.maximum(w, 1e-6)
        self.weights, self.means, self.covs = w / w.sum(), mu, cov
        self._refresh()

    # -- public -------------------------------------------------------------
    def init_fit(self, X):
        """Cold start: k-means to get the initial component assignment."""
        X = np.asarray(X, float)
        if len(X) < self.K:
            X = np.repeat(X, self.K, axis=0)
        self._fit_components(X, kmeans(X, self.K, self.rng))
        return self

    def component_costs(self, X):
        """(N, K) array of D(., k, theta, z) - the per-component data cost."""
        X = np.asarray(X, float)
        d = X[:, None, :] - self.means[None]                     # (N, K, 3)
        maha = np.einsum("nki,kij,nkj->nk", d, self.inv, d)
        return -self.logw[None] + 0.5 * self.logdet[None] + 0.5 * maha

    def assign(self, X):
        """Step 1: k_n <- argmin_k D."""
        return self.component_costs(X).argmin(1)

    def refit(self, X, lab=None):
        """Step 2: theta <- ML parameters of each component's pixels."""
        X = np.asarray(X, float)
        if len(X) == 0:
            return self
        self._fit_components(X, self.assign(X) if lab is None else lab)
        return self

    def cost(self, X):
        """min_k D - the pixel's contribution to U for this label."""
        if len(X) == 0:
            return np.zeros(0)
        return self.component_costs(X).min(1)

    def params(self):
        return self.weights.copy(), self.means.copy(), self.covs.copy()

    def set_params(self, w, mu, cov):
        self.weights, self.means, self.covs = np.array(w), np.array(mu), np.array(cov)
        self._refresh()
        return self


def blend_params(a, b, t):
    """
    Interpolate between two GMM parameter sets so refits can be *animated*.

    Components are matched greedily by mean distance first, otherwise ellipse k
    in frame 1 would fly across the plot to become an unrelated ellipse k in
    frame 2 just because the fit renumbered them.
    """
    wa, mua, ca = a
    wb, mub, cb = b
    K = len(wa)
    cost = ((mua[:, None, :] - mub[None]) ** 2).sum(-1)
    order, used = [], set()
    for i in range(K):
        j = min((j for j in range(K) if j not in used), key=lambda j: cost[i, j])
        order.append(j)
        used.add(j)
    order = np.array(order)
    w = wa + (wb[order] - wa) * t
    mu = mua + (mub[order] - mua) * t
    cov = ca + (cb[order] - ca) * t
    return w, mu, cov, order
