"""
Reusable drawing primitives, so every screen is literally made of the same parts.

Everything here follows the same contract as Scene: build the artists once,
then mutate them. Nothing re-plots inside an animation frame.
"""

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Rectangle

from . import theme as T


# --- equations -------------------------------------------------------------
class EqStack:
    """
    A column of equation/prose lines that reveal one at a time and can be
    highlighted individually. Lines rise a few points as they fade in, which
    reads as 'settling into place' rather than 'blinking on'.
    """

    def __init__(self, ax, lines, x=0.5, y0=0.72, dy=0.13, ha="center",
                 sizes=None, colors=None):
        self.ax = ax
        self.y0, self.dy = y0, dy
        self.n = len(lines)
        self.texts = []
        for i, s in enumerate(lines):
            size = (sizes[i] if sizes else 22)
            col = (colors[i] if colors else T.TEXT)
            t = ax.text(x, y0 - i * dy, s, transform=ax.transAxes, ha=ha,
                        va="center", fontsize=size, color=col, alpha=0.0)
            self.texts.append(t)
        self.base_colors = [t.get_color() for t in self.texts]
        self.base_y = [y0 - i * dy for i in range(self.n)]

    def reveal(self, count, t=1.0, rise=0.022):
        """Show `count` lines; the last one animates in with progress `t`."""
        for i, txt in enumerate(self.texts):
            if i < count - 1:
                a, off = 1.0, 0.0
            elif i == count - 1:
                e = T.smooth(t)
                a, off = e, (1 - e) * rise
            else:
                a, off = 0.0, rise
            txt.set_alpha(a)
            txt.set_position((txt.get_position()[0], self.base_y[i] - off))

    def reveal_all(self, t=1.0):
        for i, txt in enumerate(self.texts):
            e = T.stagger(t, i, self.n)
            txt.set_alpha(e)
            txt.set_position((txt.get_position()[0], self.base_y[i] - (1 - e) * 0.022))

    def emphasise(self, index, t=1.0, on=T.GOLD, off=T.DIM):
        """Pull one line forward and push the rest back."""
        e = T.smooth(t)
        for i, txt in enumerate(self.texts):
            if txt.get_alpha() == 0:
                continue
            target = on if i == index else off
            txt.set_color(_mix(self.base_colors[i], target, e))

    def reset_colors(self):
        for txt, c in zip(self.texts, self.base_colors):
            txt.set_color(c)


def _mix(c1, c2, t):
    from matplotlib.colors import to_rgb
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple(a + (b - a) * float(np.clip(t, 0, 1)))


def mix(c1, c2, t):
    return _mix(c1, c2, t)


class DefList:
    """
    Rows of (equation, caption) with real vertical rhythm.

    Mathtext is much taller than its nominal font size, so a caption placed at a
    fixed small offset below an equation collides with its descenders. Each row
    here gets its own generous block instead.
    """

    def __init__(self, ax, rows, x=0.0, y0=0.88, row_h=0.30, gap=0.105,
                 eq_size=27, cap_size=14.5, ha="left"):
        self.ax, self.rows = ax, rows
        self.eqs, self.caps = [], []
        for i, (eq, cap) in enumerate(rows):
            y = y0 - i * row_h
            self.eqs.append(ax.text(x, y, eq, transform=ax.transAxes, ha=ha,
                                    va="center", fontsize=eq_size, color=T.TEXT,
                                    alpha=0.0))
            self.caps.append(ax.text(x, y - gap, cap, transform=ax.transAxes,
                                     ha=ha, va="top", fontsize=cap_size,
                                     color=T.MUTED, alpha=0.0, linespacing=1.62))
        self.base_y = [y0 - i * row_h for i in range(len(rows))]
        self.gap = gap

    def reveal(self, count, t=1.0, rise=0.02):
        for i in range(len(self.rows)):
            if i < count - 1:
                a, off = 1.0, 0.0
            elif i == count - 1:
                e = T.smooth(t)
                a, off = e, (1 - e) * rise
            else:
                a, off = 0.0, rise
            self.eqs[i].set_alpha(a)
            self.caps[i].set_alpha(a * 0.95)
            x = self.eqs[i].get_position()[0]
            self.eqs[i].set_position((x, self.base_y[i] - off))
            self.caps[i].set_position((x, self.base_y[i] - self.gap - off))

    def focus(self, index):
        """Bring one row forward, push the others back."""
        for i in range(len(self.rows)):
            if self.eqs[i].get_alpha() == 0:
                continue
            near = (i == index)
            self.eqs[i].set_color(T.TEXT if near else mix(T.TEXT, T.BG, 0.55))
            self.caps[i].set_color(T.MUTED if near else mix(T.MUTED, T.BG, 0.5))


def tag(ax, x, y, text, color=T.GOLD, size=12.5, ha="center"):
    """A small rounded chip - used to label a term under an equation."""
    t = ax.text(x, y, text, transform=ax.transAxes, ha=ha, va="center",
                fontsize=size, color=color, alpha=0.0, zorder=5,
                bbox=dict(boxstyle="round,pad=0.42", facecolor=T.PANEL,
                          edgecolor=color, linewidth=1.0, alpha=0.95))
    return t


def brace(ax, x0, x1, y, color=T.GOLD, lw=1.6, depth=0.022):
    """An underline with turned-down ends: a poor man's \\underbrace."""
    xs = [x0, x0, x1, x1]
    ys = [y + depth, y, y, y + depth]
    (ln,) = ax.plot(xs, ys, transform=ax.transAxes, color=color, lw=lw,
                    solid_joinstyle="round", alpha=0.0, zorder=4)
    return ln


# --- gaussians -------------------------------------------------------------
def ellipse_geom(mean2, cov2, nsig=2.0):
    """(width, height, angle_deg) of the n-sigma ellipse of a 2x2 covariance."""
    vals, vecs = np.linalg.eigh(cov2)
    vals = np.maximum(vals, 1e-9)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * nsig * np.sqrt(vals)
    return float(w), float(h), float(ang)


class EllipseSet:
    """
    K covariance ellipses in a 2-D color plane, drawn at two sigma levels so a
    component reads as a soft cloud rather than a hard ring.
    """

    def __init__(self, ax, K, color, proj=(0, 2), zorder=6):
        self.ax, self.K, self.color, self.proj = ax, K, color, proj
        self.rings = []
        self.cores = []
        for _ in range(K):
            outer = Ellipse((0, 0), 1, 1, facecolor="none", edgecolor=color,
                            lw=1.7, alpha=0.0, zorder=zorder, clip_on=True)
            inner = Ellipse((0, 0), 1, 1, facecolor=color, edgecolor="none",
                            alpha=0.0, zorder=zorder - 1, clip_on=True)
            ax.add_patch(outer)
            ax.add_patch(inner)
            self.rings.append(outer)
            self.cores.append(inner)
        self.dots = ax.scatter(np.zeros(K), np.zeros(K), s=18, c=color,
                               edgecolors=T.BG, linewidths=0.8, zorder=zorder + 1,
                               alpha=0.0)

    def update(self, weights, means, covs, alpha=1.0, nsig=1.6):
        i, j = self.proj
        idx = np.ix_([i, j], [i, j])
        cen = np.zeros((self.K, 2))
        for k in range(self.K):
            m2 = np.array([means[k][i], means[k][j]])
            c2 = np.asarray(covs[k])[idx]
            w, h, a = ellipse_geom(m2, c2, nsig)
            wt = float(np.clip(weights[k] * self.K, 0.25, 2.0))
            for patch, scale, base in ((self.rings[k], 1.0, 0.95),
                                       (self.cores[k], 0.60, 0.10)):
                patch.set_center(m2)
                patch.set_width(w * scale)
                patch.set_height(h * scale)
                patch.set_angle(a)
                patch.set_alpha(alpha * base * min(1.0, 0.45 + 0.55 * wt))
            self.rings[k].set_linewidth(1.0 + 1.3 * min(wt, 1.6))
            cen[k] = m2
        self.dots.set_offsets(cen)
        self.dots.set_alpha(alpha)

    def set_alpha(self, a):
        for p in self.rings + self.cores:
            p.set_alpha(p.get_alpha() if a is None else a)
        self.dots.set_alpha(a)


# --- toy image / mask ------------------------------------------------------
class ImageGrid:
    """
    The 8x8 toy drawn as discrete swatches with visible gaps, so it reads as
    'a grid of pixels' and lines up one-for-one with the graph's nodes.
    """

    def __init__(self, ax, img, gap=0.10, zorder=2):
        self.ax, self.img = ax, img
        self.n = img.shape[0]
        self.patches = []
        n = self.n
        for r in range(n):
            for c in range(n):
                p = Rectangle((c + gap / 2, (n - 1 - r) + gap / 2),
                              1 - gap, 1 - gap,
                              facecolor=img[r, c], edgecolor="none",
                              zorder=zorder)
                ax.add_patch(p)
                self.patches.append(p)
        self.overlays = []
        for r in range(n):
            for c in range(n):
                p = Rectangle((c + gap / 2, (n - 1 - r) + gap / 2),
                              1 - gap, 1 - gap,
                              facecolor=T.BG, edgecolor="none", alpha=0.0,
                              zorder=zorder + 1)
                ax.add_patch(p)
                self.overlays.append(p)
        ax.set_xlim(-0.35, n + 0.35)
        ax.set_ylim(-0.35, n + 0.35)
        ax.set_aspect("equal")
        T.blank(ax)

    def set_mask(self, alpha_mask, strength=1.0, bg_color=T.BG, bg_alpha=0.74):
        """Veil the background pixels; foreground pixels keep full color."""
        a = np.asarray(alpha_mask).reshape(-1).astype(bool)
        for i, p in enumerate(self.overlays):
            p.set_facecolor(bg_color)
            p.set_alpha(0.0 if a[i] else bg_alpha * strength)

    def clear_mask(self):
        for p in self.overlays:
            p.set_alpha(0.0)

    def set_image_alpha(self, a):
        for p in self.patches:
            p.set_alpha(a)


def mask_outline(ax, mask, color=T.GOLD, lw=2.4, zorder=8):
    """Trace the boundary of a boolean mask on an ImageGrid's axes."""
    n = mask.shape[0]
    segs = []
    m = np.asarray(mask).astype(bool)
    for r in range(n):
        for c in range(n):
            if not m[r, c]:
                continue
            y = n - 1 - r
            if r == 0 or not m[r - 1, c]:
                segs.append([(c, y + 1), (c + 1, y + 1)])
            if r == n - 1 or not m[r + 1, c]:
                segs.append([(c, y), (c + 1, y)])
            if c == 0 or not m[r, c - 1]:
                segs.append([(c, y), (c, y + 1)])
            if c == n - 1 or not m[r, c + 1]:
                segs.append([(c + 1, y), (c + 1, y + 1)])
    lc = LineCollection(segs, colors=color, linewidths=lw, zorder=zorder,
                        capstyle="round")
    ax.add_collection(lc)
    return lc


def set_outline(lc, mask):
    n = mask.shape[0]
    m = np.asarray(mask).astype(bool)
    segs = []
    for r in range(n):
        for c in range(n):
            if not m[r, c]:
                continue
            y = n - 1 - r
            if r == 0 or not m[r - 1, c]:
                segs.append([(c, y + 1), (c + 1, y + 1)])
            if r == n - 1 or not m[r + 1, c]:
                segs.append([(c, y), (c + 1, y)])
            if c == 0 or not m[r, c - 1]:
                segs.append([(c, y), (c, y + 1)])
            if c == n - 1 or not m[r, c + 1]:
                segs.append([(c + 1, y), (c + 1, y + 1)])
    lc.set_segments(segs)


def _compact(v):
    """-1744474 -> '-1.74 M'; keeps the y axis from eating the plot area."""
    a = abs(v)
    for scale, suffix in ((1e9, " B"), (1e6, " M"), (1e3, " k")):
        if a >= scale:
            return f"{v / scale:,.2f}{suffix}"
    return f"{v:,.1f}"


# --- charts ----------------------------------------------------------------
class EnergyChart:
    """E against iteration, revealed point by point."""

    def __init__(self, ax, n_max, color=T.GOLD, label=r"$E$"):
        self.ax, self.n_max = ax, n_max
        T.chart(ax)
        (self.line,) = ax.plot([], [], color=color, lw=2.2, zorder=4,
                               solid_capstyle="round")
        self.dots = ax.scatter([], [], s=42, c=color, edgecolors=T.BG,
                               linewidths=1.4, zorder=5)
        self.halo = ax.scatter([], [], s=150, c=color, alpha=0.16, zorder=3)
        self.xs, self.ys = [], []
        ax.set_xlabel("iteration", color=T.MUTED, fontsize=11, labelpad=6)
        self.xlabel = ax.xaxis.label
        self.vlabel = ax.text(0.0, 1.06, label, transform=ax.transAxes,
                              color=T.MUTED, fontsize=13, va="bottom")

    def set_data(self, xs, ys, frac=1.0):
        """`frac` lets the newest segment grow in rather than pop in."""
        self.xs, self.ys = list(xs), list(ys)
        if not xs:
            self.line.set_data([], [])
            self.dots.set_offsets(np.empty((0, 2)))
            self.halo.set_offsets(np.empty((0, 2)))
            self.ax.set_yticks([])                  # otherwise a reset keeps the
            self.ax.set_xticks([])                  # previous run's axis on screen
            self.ax.set_xlim(0.45, max(self.n_max, 2) + 0.55)
            self.ax.set_ylim(0, 1)
            return
        px, py = list(xs), list(ys)
        if frac < 1.0 and len(xs) >= 2:
            e = T.smooth(frac)
            px = xs[:-1] + [xs[-2] + (xs[-1] - xs[-2]) * e]
            py = ys[:-1] + [ys[-2] + (ys[-1] - ys[-2]) * e]
        self.line.set_data(px, py)
        self.dots.set_offsets(np.c_[px, py])
        self.halo.set_offsets(np.c_[px[-1:], py[-1:]])
        self._autoscale()

    def _autoscale(self):
        ax = self.ax
        ax.set_xlim(0.45, max(self.n_max, 2) + 0.55)
        lo, hi = min(self.ys), max(self.ys)
        pad = max((hi - lo) * 0.28, abs(hi) * 1e-3, 1e-6)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(range(1, max(self.n_max, 2) + 1))
        ax.set_yticks([lo, hi])
        ax.set_yticklabels([_compact(lo), _compact(hi)])
        ax.tick_params(labelsize=9.5)


# --- misc ------------------------------------------------------------------
def card(fig, rect, color=T.PANEL, edge=T.PANEL_EDGE, alpha=1.0, r=0.012, z=0):
    """A rounded background card placed in figure coordinates."""
    x, y, w, h = rect
    p = FancyBboxPatch((x, y), w, h, transform=fig.transFigure,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=color, edgecolor=edge, linewidth=1.0,
                       alpha=alpha, zorder=z)
    fig.add_artist(p)
    return p


def legend_dot(ax, x, y, color, label, size=11.5, r=0.011):
    d = Circle((x, y), r, transform=ax.transAxes, facecolor=color,
               edgecolor="none", zorder=6, clip_on=False)
    ax.add_patch(d)
    t = ax.text(x + 0.032, y, label, transform=ax.transAxes, va="center",
                ha="left", fontsize=size, color=T.MUTED)
    return d, t
