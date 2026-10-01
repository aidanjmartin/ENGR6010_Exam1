"""
The toy image drawn as an explicit s-t graph.

One node per pixel, laid out in a grid and *filled with that pixel's own color*,
so the graph and the image are visibly the same object. Source S (foreground)
sits above the grid, sink T (background) below. Edge opacity and width encode
weight, which is the only honest way to show that the cut goes through the thin
edges - the places where neighbouring colors disagree.
"""

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from . import theme as T
from .draw import mix


def _norm(v, lo=None, hi=None):
    v = np.asarray(v, float)
    lo = v.min() if lo is None else lo
    hi = v.max() if hi is None else hi
    return np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)


class GraphView:
    def __init__(self, ax, img, pairs, node_s=210.0, grid_h=0.615):
        self.ax, self.img = ax, img
        self.n = img.shape[0]
        self.N = self.n * self.n
        self.pairs = list(pairs)

        T.blank(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # square cells regardless of how wide the axes ends up
        fig = ax.figure
        fw, fh = fig.get_size_inches()
        p = ax.get_position()
        aw, ah = p.width * fw, p.height * fh
        step_y = grid_h / (self.n - 1)
        step_x = step_y * ah / aw
        self.step_x, self.step_y = step_x, step_y

        x0 = 0.5 - step_x * (self.n - 1) / 2
        y_top = 0.5 + grid_h / 2
        self.pos = np.zeros((self.N, 2))
        for r in range(self.n):
            for c in range(self.n):
                self.pos[r * self.n + c] = (x0 + c * step_x, y_top - r * step_y)

        self.S = np.array([0.5, 0.945])
        self.Tk = np.array([0.5, 0.055])

        # --- t-links (drawn first, they sit behind everything) -------------
        self.s_lc = LineCollection([[tuple(self.S), tuple(q)] for q in self.pos],
                                   colors=T.BLUE, linewidths=0.6, zorder=2)
        self.t_lc = LineCollection([[tuple(q), tuple(self.Tk)] for q in self.pos],
                                   colors=T.SLATE, linewidths=0.6, zorder=2)
        ax.add_collection(self.s_lc)
        ax.add_collection(self.t_lc)

        # --- n-links -------------------------------------------------------
        self.n_lc = LineCollection(
            [[tuple(self.pos[a]), tuple(self.pos[b])] for a, b in self.pairs],
            colors=T.SLATE, linewidths=1.0, zorder=4)
        ax.add_collection(self.n_lc)

        # --- the cut -------------------------------------------------------
        self.cut_lc = LineCollection([], colors=T.GOLD, linewidths=2.0,
                                     zorder=3, capstyle="round", alpha=0.35)
        ax.add_collection(self.cut_lc)
        self.stroke_lc = LineCollection([], colors=T.GOLD, linewidths=3.2,
                                        zorder=8, capstyle="round")
        ax.add_collection(self.stroke_lc)

        # --- nodes ---------------------------------------------------------
        self.node_face = img.reshape(-1, 3).copy()
        self.nodes = ax.scatter(self.pos[:, 0], self.pos[:, 1], s=node_s,
                                c=self.node_face, edgecolors=T.BG,
                                linewidths=1.4, zorder=6)
        self.halo = ax.scatter(self.pos[:, 0], self.pos[:, 1], s=node_s * 2.5,
                               c=T.GOLD, alpha=0.0, zorder=5)

        # --- terminals -----------------------------------------------------
        self.s_patch = Circle(self.S, 0.030, facecolor=T.BLUE, edgecolor="none",
                              zorder=9)
        self.t_patch = Circle(self.Tk, 0.030, facecolor=T.SLATE, edgecolor="none",
                              zorder=9)
        ax.add_patch(self.s_patch)
        ax.add_patch(self.t_patch)
        self.s_txt = ax.text(self.S[0], self.S[1], "S", ha="center", va="center",
                             fontsize=15, color=T.BG, zorder=10, fontweight="bold")
        self.t_txt = ax.text(self.Tk[0], self.Tk[1], "T", ha="center", va="center",
                             fontsize=15, color=T.BG, zorder=10, fontweight="bold")
        self.s_lab = ax.text(self.S[0] + 0.055, self.S[1], "foreground terminal",
                             ha="left", va="center", fontsize=11.5, color=T.BLUE,
                             zorder=10)
        self.t_lab = ax.text(self.Tk[0] + 0.055, self.Tk[1], "background terminal",
                             ha="left", va="center", fontsize=11.5, color=T.SLATE,
                             zorder=10)

        # sane default weights so alpha controls are usable before a real fit
        self.set_tlink_weights(np.ones(self.N), np.ones(self.N))
        self.set_nlink_weights(np.ones(len(self.pairs)))

        self.set_terminal_alpha(0.0)
        self.set_tlink_alpha(0.0)
        self.set_nlink_alpha(0.0)
        self.set_node_alpha(0.0)

    # --- alpha controls ----------------------------------------------------
    def set_node_alpha(self, a, subset=None):
        base = np.full(self.N, float(a))
        if subset is not None:
            base[:] = 0.12 * a
            base[list(subset)] = a
        self.nodes.set_alpha(base)

    def set_terminal_alpha(self, a):
        for art in (self.s_patch, self.t_patch, self.s_txt, self.t_txt,
                    self.s_lab, self.t_lab):
            art.set_alpha(a)

    def set_tlink_alpha(self, a, subset=None):
        for lc, w in ((self.s_lc, self._sw), (self.t_lc, self._tw)):
            al = w * a
            if subset is not None:
                keep = np.zeros(self.N, bool)
                keep[list(subset)] = True
                al = np.where(keep, al, 0.0)
            lc.set_alpha(al)

    def set_nlink_alpha(self, a, subset=None):
        al = self._nw_alpha * a
        if subset is not None:
            keep = np.zeros(len(self.pairs), bool)
            keep[list(subset)] = True
            al = np.where(keep, al, 0.0)
        self.n_lc.set_alpha(al)

    # --- weights -----------------------------------------------------------
    def set_tlink_weights(self, w_source, w_sink, gain=1.0):
        s = _norm(w_source)
        t = _norm(w_sink)
        self._sw = 0.10 + 0.80 * s
        self._tw = 0.10 + 0.80 * t
        self.s_lc.set_linewidths(0.35 + 1.7 * s * gain)
        self.t_lc.set_linewidths(0.35 + 1.7 * t * gain)

    def set_nlink_weights(self, w):
        v = _norm(w)
        # most neighbour pairs in a flat image sit near w_max, so the raw
        # range is useless as a visual signal; push it through a power curve
        self._nw_alpha = 0.10 + 0.88 * v ** 1.3
        self.n_lc.set_linewidths(0.35 + 4.2 * v ** 1.6)
        # weak links recede toward the page, strong links come forward
        self.n_lc.set_colors([mix(T.SLATE_DEEP, T.SLATE, x ** 1.3) for x in v])

    # --- the cut -----------------------------------------------------------
    def cut_order(self, cut_pairs):
        """Order cut edges by angle around the mask centroid, so the highlight
        sweeps around the boundary as one continuous motion."""
        mids = np.array([(self.pos[a] + self.pos[b]) / 2 for a, b in cut_pairs])
        if not len(mids):
            return np.array([], int)
        c = mids.mean(0)
        ang = np.arctan2(mids[:, 1] - c[1], mids[:, 0] - c[0])
        return np.argsort(-ang)

    def show_cut(self, cut_pairs, progress=1.0, stroke=True):
        """Highlight the cut n-links; `progress` sweeps them in around the loop."""
        if not cut_pairs:
            self.cut_lc.set_segments([])
            self.stroke_lc.set_segments([])
            return
        order = self.cut_order(cut_pairs)
        k = int(round(T.smooth(progress) * len(order)))
        take = [cut_pairs[i] for i in order[:k]]
        self.cut_lc.set_segments(
            [[tuple(self.pos[a]), tuple(self.pos[b])] for a, b in take])
        if stroke:
            segs = []
            for a, b in take:
                pa, pb = self.pos[a], self.pos[b]
                m = (pa + pb) / 2
                d = pb - pa
                perp = np.array([-d[1], d[0]])
                nrm = np.linalg.norm(perp)
                half = (self.step_x if abs(d[1]) > abs(d[0]) else self.step_y) / 2
                perp = perp / (nrm + 1e-12) * half
                segs.append([tuple(m - perp), tuple(m + perp)])
            self.stroke_lc.set_segments(segs)
        else:
            self.stroke_lc.set_segments([])

    def show_labels(self, alpha_mask, strength=1.0):
        """Recolor nodes by their label: foreground keeps its color, background
        is desaturated toward the page."""
        a = np.asarray(alpha_mask).reshape(-1).astype(bool)
        from matplotlib.colors import to_rgb
        page = np.array(to_rgb(T.BG))
        cols = self.node_face.copy()
        k = 0.72 * strength
        cols[~a] = cols[~a] * (1 - k) + page * k
        self.nodes.set_facecolor(cols)
        ring = np.where(a[:, None], np.array(to_rgb(T.BLUE)),
                        np.array(to_rgb(T.SLATE_DEEP)))
        self.nodes.set_edgecolor(ring)

    def clear_labels(self):
        self.nodes.set_facecolor(self.node_face)
        self.nodes.set_edgecolor(T.BG)

    def highlight_nodes(self, idx, a=1.0):
        al = np.zeros(self.N)
        al[list(idx)] = a
        self.halo.set_alpha(al)

    def node_xy(self, r, c):
        return self.pos[r * self.n + c]

    def pair_index(self, p, q):
        for i, (a, b) in enumerate(self.pairs):
            if (a, b) == (p, q) or (a, b) == (q, p):
                return i
        return None
