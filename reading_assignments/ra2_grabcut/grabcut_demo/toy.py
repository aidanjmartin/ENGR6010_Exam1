"""
The toy instance: a real GrabCut, small enough to draw every node and edge.

An 8x8 image gives 64 pixel nodes plus source and sink, which is few enough to
lay out on screen as an honest graph. The max-flow here is a plain Edmonds-Karp
BFS - at 66 nodes that is instant, and unlike a heuristic it returns the *exact*
min cut, which is what makes the monotonically-decreasing energy plot a real
result rather than a drawing.

Labels follow the paper: alpha_n = 1 is foreground, alpha_n = 0 is background.
The source terminal S is the foreground terminal; pixels still connected to S
after the cut are the foreground.
"""

from collections import deque

import numpy as np

from .gmm import GrabCutGMM

N_SIDE = 8
GAMMA = 3.5            # smoothness weight; tuned for [0,1] colors + 4-neighbours
HARD = 1e4             # t-link weight that pins a definite-background pixel
K_TOY = 2              # components per class in the toy (the paper uses K = 5)
NOISE = 0.04
RAMP = 0.22            # softness of the object's edge, in pixels

# Colors are chosen so the object is warm and the ground is cool - readable at
# the back of a room - but they are ordinary image data, not part of the UI
# palette, so they are allowed to be saturated.
C_LIT = np.array([0.91, 0.62, 0.28])     # object, lit side
C_SHADE = np.array([0.62, 0.38, 0.20])   # object, shaded side
C_GROUND = np.array([0.20, 0.30, 0.44])  # background


# --- the image -------------------------------------------------------------
def _membership(n=N_SIDE, cy=3.4, cx=3.5, ry=2.3, rx=2.0, ramp=RAMP):
    """Soft 0..1 objectness field. The soft edge is the point: a hard-edged blob
    is solved in a single cut and there is no iteration left to show."""
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    r = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)
    return 1.0 / (1.0 + np.exp((r - 1.0) / ramp)), yy, cy


def true_mask(n=N_SIDE):
    w, _, _ = _membership(n)
    return w > 0.5


def make_toy(seed=7, noise=NOISE):
    """
    A shaded object on a flat ground, with a soft boundary.

    The object is deliberately given two tones, so both of the foreground GMM's
    K = 2 components are spent on the object itself and none is left over to
    model the halo - which is what makes the halo peel away over successive
    iterations instead of sticking after the first cut.
    """
    rng = np.random.default_rng(seed)
    w, yy, cy = _membership(n=N_SIDE)
    shade = np.clip((yy - cy) / 4.0 + 0.5, 0, 1)[..., None]
    core = C_LIT * (1 - shade) + C_SHADE * shade
    img = C_GROUND + (core - C_GROUND) * w[..., None]
    img = np.clip(img + rng.normal(0, noise, img.shape), 0, 1)
    return img, w > 0.5


def init_rect(n=N_SIDE):
    """The presenter's bounding box: deliberately looser than the object."""
    return (1, 1, n - 2, n - 2)                       # r0, c0, r1, c1 inclusive


def trimap(rect, n=N_SIDE):
    """0 = definite background (outside the box), 1 = unknown (inside)."""
    r0, c0, r1, c1 = rect
    tm = np.zeros((n, n), dtype=np.uint8)
    tm[r0:r1 + 1, c0:c1 + 1] = 1
    return tm


# --- graph -----------------------------------------------------------------
def neighbours(n=N_SIDE):
    """4-connected pairs. The paper uses 8; 4 is legible as a drawing."""
    out = []
    for r in range(n):
        for c in range(n):
            i = r * n + c
            if c + 1 < n:
                out.append((i, i + 1))
            if r + 1 < n:
                out.append((i, i + n))
    return out


def beta_of(img, pairs=None):
    """beta = 1 / (2 <||z_m - z_n||^2>) - the image's own average contrast."""
    n = img.shape[0]
    z = img.reshape(-1, 3)
    pairs = pairs or neighbours(n)
    d2 = np.array([((z[a] - z[b]) ** 2).sum() for a, b in pairs])
    return 1.0 / (2.0 * d2.mean()), d2


def nlink_weights(img, gamma=GAMMA):
    """V's per-edge weight: gamma * exp(-beta ||z_m - z_n||^2)."""
    pairs = neighbours(img.shape[0])
    beta, d2 = beta_of(img, pairs)
    return pairs, gamma * np.exp(-beta * d2), beta


# --- max flow --------------------------------------------------------------
def max_flow_min_cut(cap):
    """
    Edmonds-Karp. `cap` is a dense (V, V) capacity matrix; the last two indices
    are source and sink. Returns (flow value, boolean mask of the source side).
    """
    cap = cap.astype(float).copy()
    V = cap.shape[0]
    s, t = V - 2, V - 1
    flow = 0.0
    while True:
        parent = np.full(V, -1)
        parent[s] = s
        q = deque([s])
        while q and parent[t] < 0:
            u = q.popleft()
            for v in np.nonzero(cap[u] > 1e-12)[0]:
                if parent[v] < 0:
                    parent[v] = u
                    q.append(v)
        if parent[t] < 0:
            break
        # bottleneck along the augmenting path
        path, v = [], t
        while v != s:
            path.append((parent[v], v))
            v = parent[v]
        b = min(cap[u, v] for u, v in path)
        for u, v in path:
            cap[u, v] -= b
            cap[v, u] += b
        flow += b
    # residual reachability from s = source side of the min cut
    seen = np.zeros(V, bool)
    seen[s] = True
    q = deque([s])
    while q:
        u = q.popleft()
        for v in np.nonzero(cap[u] > 1e-12)[0]:
            if not seen[v]:
                seen[v] = True
                q.append(v)
    return flow, seen


def build_capacity(img, tm, fg_gmm, bg_gmm, gamma=GAMMA):
    """
    t-links carry the data term, n-links the smoothness term.

    Following OpenCV's grabcut.cpp: the capacity from S is the pixel's cost
    under the *background* model and the capacity to T is its cost under the
    *foreground* model, so a pixel that looks like background has a cheap S-link
    and gets cut away from S. Costs are negative log likelihoods and can be
    negative, so both are shifted by one shared constant - which adds N*offset
    to every possible cut and therefore leaves the argmin untouched.
    """
    n = img.shape[0]
    z = img.reshape(-1, 3)
    N = n * n
    pairs, w, beta = nlink_weights(img, gamma)

    d_fg = fg_gmm.cost(z)
    d_bg = bg_gmm.cost(z)
    offset = max(0.0, -min(d_fg.min(), d_bg.min())) + 1.0

    cap = np.zeros((N + 2, N + 2))
    S, Tk = N, N + 1
    unknown = tm.reshape(-1) == 1
    cap[S, :N] = np.where(unknown, d_bg + offset, 0.0)
    cap[:N, Tk] = np.where(unknown, d_fg + offset, HARD)
    for (a, b), wv in zip(pairs, w):
        cap[a, b] += wv
        cap[b, a] += wv
    return cap, pairs, w, beta, d_fg, d_bg


def energy(img, alpha, fg_gmm, bg_gmm, gamma=GAMMA):
    """E = U + V, evaluated honestly from the unshifted negative log likelihoods."""
    z = img.reshape(-1, 3)
    a = alpha.reshape(-1).astype(bool)
    U = 0.0
    if a.any():
        U += float(fg_gmm.cost(z[a]).sum())
    if (~a).any():
        U += float(bg_gmm.cost(z[~a]).sum())
    pairs, w, _ = nlink_weights(img, gamma)
    V = float(sum(wv for (p, q), wv in zip(pairs, w) if a[p] != a[q]))
    return U + V, U, V


# --- the alternating loop --------------------------------------------------
def run_toy_grabcut(iters=6, K=K_TOY, gamma=GAMMA, seed=7):
    """
    Returns one record per iteration with everything the visuals need:
    GMM parameters before and after the refit, the cut, the mask, and E.
    """
    img, gt = make_toy(seed)
    n = img.shape[0]
    z = img.reshape(-1, 3)
    rect = init_rect(n)
    tm = trimap(rect, n)

    alpha = (tm == 1).astype(np.uint8)               # step 0: box interior is FG
    fg = GrabCutGMM(K, seed=seed).init_fit(z[alpha.reshape(-1) == 1])
    bg = GrabCutGMM(K, seed=seed + 1).init_fit(z[alpha.reshape(-1) == 0])

    history = []
    for it in range(1, iters + 1):
        before = (fg.params(), bg.params())

        # step 1+2: reassign components and refit theta to the current labels
        a = alpha.reshape(-1) == 1
        if a.any():
            fg.refit(z[a])
        if (~a).any():
            bg.refit(z[~a])
        after = (fg.params(), bg.params())

        # step 3: global min cut for the new theta
        cap, pairs, w, beta, d_fg, d_bg = build_capacity(img, tm, fg, bg, gamma)
        flow, src_side = max_flow_min_cut(cap)
        new_alpha = src_side[:n * n].reshape(n, n).astype(np.uint8)
        new_alpha[tm == 0] = 0                       # box exterior stays background

        E, U, V = energy(img, new_alpha, fg, bg, gamma)
        cut_edges = [(p, q) for (p, q) in pairs
                     if new_alpha.reshape(-1)[p] != new_alpha.reshape(-1)[q]]
        changed = int((new_alpha != alpha).sum())

        history.append(dict(it=it, gmm_before=before, gmm_after=after,
                            alpha_before=alpha.copy(), alpha=new_alpha.copy(),
                            pairs=pairs, nweights=w, beta=beta, flow=flow,
                            cut_edges=cut_edges, d_fg=d_fg, d_bg=d_bg,
                            E=E, U=U, V=V, changed=changed))
        alpha = new_alpha
        if changed == 0 and it >= 3:
            break

    return dict(img=img, gt=gt, rect=rect, trimap=tm, history=history,
                fg=fg, bg=bg, gamma=gamma)
