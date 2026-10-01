"""
The real-photo side of the demo: loading, GMM fitting, and a stepped GrabCut.

`cv2.grabCut` is called with `iterCount=1` inside a loop rather than run to
convergence in one shot, so each turn of the alternating loop from Part A can be
put on screen as it happens. OpenCV does not report its energy, so E is
recomputed here from the current mask using the same GMM and the same
smoothness term as the toy - labelled honestly on screen as a re-evaluation.

Color space note (Ibraheem et al.): the GMMs below live in RGB because that is
what the paper and `cv2.grabCut` use. RGB entangles brightness with hue, so a
shadowed patch of an object can land far from its lit patch and cost a whole
mixture component; Lab or HSV would separate those better.
"""

import numpy as np

try:
    import cv2
except ImportError:                                    # pragma: no cover
    cv2 = None

from .gmm import GrabCutGMM

K_REAL = 5              # the paper's K
GAMMA_REAL = 50.0       # the paper's gamma, for 8-connectivity on 0..255 colors
MAX_SIDE = 520          # working resolution; keeps one iteration well under a second

# Which two of the three channels the color-space scatter is drawn in.
PROJ = (0, 2)
PROJ_NAMES = ("R", "B")

# `astronaut` is the default because its two classes start badly overlapped in
# RGB and pull cleanly apart as the loop runs - which is the thing worth
# watching. (Fisher separability in the R-B plane goes from 0.04 at the initial
# box to 0.24 at convergence; for `coffee` it barely moves.)
SAMPLE = "astronaut"


# --- loading ---------------------------------------------------------------
def load_image(path=None, sample=SAMPLE):
    """Bundled scikit-image sample by default; any file path if one is given."""
    if path:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        from skimage import data
        img = getattr(data, sample)()
        if img.ndim == 2:
            img = np.dstack([img] * 3)
        img = np.ascontiguousarray(img[..., :3])
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                         interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img.astype(np.uint8))


def default_rect(img, fx=0.16, fy=0.05):
    """
    A plausible box for --export and for the pre-drag state.

    Insets more on the sides than top and bottom: a box a presenter actually
    draws hugs the subject horizontally but usually runs off the top and bottom
    of it, and a symmetric inset lops the head and shoulders off the sample.
    """
    h, w = img.shape[:2]
    x, y = int(w * fx), int(h * fy)
    return (x, y, int(w * (1 - 2 * fx)), int(h * (1 - 2 * fy)))     # x, y, w, h


# --- sampling and GMMs -----------------------------------------------------
def sample_pixels(img, mask_fg, n=2400, seed=0):
    """Subsample foreground and background pixels for the scatter plot."""
    rng = np.random.default_rng(seed)
    z = img.reshape(-1, 3).astype(float) / 255.0
    f = np.flatnonzero(mask_fg.reshape(-1))
    b = np.flatnonzero(~mask_fg.reshape(-1))
    take = lambda idx: rng.choice(idx, min(n, len(idx)), replace=False) if len(idx) else idx
    return z[take(f)], z[take(b)]


def fit_pair(zf, zb, K=K_REAL, seed=3):
    fg = GrabCutGMM(K, seed=seed).init_fit(zf) if len(zf) else GrabCutGMM(K, seed=seed)
    bg = GrabCutGMM(K, seed=seed + 1).init_fit(zb) if len(zb) else GrabCutGMM(K, seed=seed + 1)
    return fg, bg


def rect_trimap(img, rect):
    """True inside the box (unknown -> initially foreground), False outside."""
    x, y, w, h = rect
    m = np.zeros(img.shape[:2], bool)
    m[max(y, 0):y + h, max(x, 0):x + w] = True
    return m


# --- energy ----------------------------------------------------------------
def _pair_slices(dy, dx):
    """Index pair (p, q) for every valid p and q = p + (dy, dx)."""
    ra = slice(max(0, -dy), None if dy <= 0 else -dy)
    rb = slice(max(0, dy), None if dy >= 0 else dy)
    ca = slice(max(0, -dx), None if dx <= 0 else -dx)
    cb = slice(max(0, dx), None if dx >= 0 else dx)
    return (ra, ca), (rb, cb)


# The four offsets that cover an 8-neighbourhood without counting an edge twice.
_OFFSETS = ((0, 1, 1.0), (1, 0, 1.0),
            (1, 1, 1 / np.sqrt(2)), (1, -1, 1 / np.sqrt(2)))


def _beta_full(z):
    """beta = 1 / (2 <||z_m - z_n||^2>) over the 8-neighbourhood, as in OpenCV."""
    acc, cnt = 0.0, 0
    for dy, dx, _ in _OFFSETS:
        A, B = _pair_slices(dy, dx)
        d = z[A] - z[B]
        acc += float((d * d).sum())
        cnt += d.shape[0] * d.shape[1]
    mean = acc / max(cnt, 1)
    return 1.0 / (2.0 * mean) if mean > 0 else 0.0


def energy_full(img, alpha, fg, bg, gamma=GAMMA_REAL):
    """E = U + V over the whole image, vectorised over the 8-neighbourhood."""
    z = img.astype(float)
    zn = z.reshape(-1, 3) / 255.0
    a = alpha.reshape(-1).astype(bool)
    U = 0.0
    if a.any():
        U += float(fg.cost(zn[a]).sum())
    if (~a).any():
        U += float(bg.cost(zn[~a]).sum())

    beta = _beta_full(z)
    V = 0.0
    A_ = alpha.astype(bool)
    for dy, dx, dist in _OFFSETS:
        P, Q = _pair_slices(dy, dx)
        d2 = ((z[P] - z[Q]) ** 2).sum(-1)
        V += float((gamma * dist * np.exp(-beta * d2) * (A_[P] != A_[Q])).sum())
    return U + V, U, V, beta


# --- morphology ------------------------------------------------------------
def clean_mask(mask, k=5):
    """
    The ImageJ-style tidy-up: open to drop speckle, close to fill pinholes.
    Deliberately a separate, toggleable pass - it is not part of GrabCut.
    """
    m = (mask.astype(np.uint8)) * 255
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, se)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se)
    return m > 127


# --- stepped session -------------------------------------------------------
class GrabCutSession:
    """One bounding box's worth of GrabCut, advanced one iteration at a time."""

    def __init__(self, img, rect, gamma=GAMMA_REAL, K=K_REAL):
        self.img = img
        self.rect = tuple(int(v) for v in rect)
        self.gamma, self.K = gamma, K
        self.bgd = np.zeros((1, 65), np.float64)
        self.fgd = np.zeros((1, 65), np.float64)
        self.mask = np.zeros(img.shape[:2], np.uint8)
        self.iteration = 0
        self.history = []          # dicts: it, E, U, V, changed, gmm params
        self._started = False
        self._prev = None
        self._fg = self._bg = None      # persistent, warm-started across steps

    @property
    def fg_mask(self):
        return (self.mask == cv2.GC_FGD) | (self.mask == cv2.GC_PR_FGD)

    def step(self):
        """One turn of the loop: refit theta, then one global min cut."""
        mode = cv2.GC_INIT_WITH_RECT if not self._started else cv2.GC_EVAL
        if not self._started:
            self.mask[:] = 0
        cv2.grabCut(self.img, self.mask, self.rect, self.bgd, self.fgd,
                    1, mode)
        self._started = True
        self.iteration += 1

        alpha = self.fg_mask
        changed = int((alpha != self._prev).sum()) if self._prev is not None else int(alpha.sum())
        self._prev = alpha.copy()

        # Warm-start the shadow GMMs instead of re-seeding k-means every step,
        # so the energy curve tracks the segmentation rather than the RNG.
        zf, zb = sample_pixels(self.img, alpha, n=2000, seed=0)
        if self._fg is None:
            self._fg, self._bg = fit_pair(zf, zb, self.K, seed=3)
        else:
            if len(zf):
                self._fg.refit(zf)
            if len(zb):
                self._bg.refit(zb)
        fg, bg = self._fg, self._bg
        E, U, V, beta = energy_full(self.img, alpha, fg, bg, self.gamma)

        rec = dict(it=self.iteration, E=E, U=U, V=V, beta=beta, changed=changed,
                   alpha=alpha.copy(), zf=zf, zb=zb,
                   fg_params=fg.params(), bg_params=bg.params())
        self.history.append(rec)
        return rec

    def converged(self, tol=None):
        if len(self.history) < 2:
            return False
        tol = tol or max(30, int(0.0006 * self.img.shape[0] * self.img.shape[1]))
        return self.history[-1]["changed"] <= tol
