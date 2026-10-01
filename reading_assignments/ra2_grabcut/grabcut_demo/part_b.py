"""
Part B - live interactive segmentation on a real photograph.

Same screen language as Part A: the mask panel replaces the toy grid, the colour
model panel is the same scatter with the same ellipses, and the energy chart is
the same chart. `cv2.grabCut` is driven one iteration at a time so each turn of
the loop from Part A is visible as it happens.
"""

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

try:
    import cv2
except ImportError:                                    # pragma: no cover
    cv2 = None

from . import draw as D
from . import realimg as RI
from . import theme as T
from .scene import Scene

L, R_, TOPY = 0.042, 0.958, 0.84


class LiveScene(Scene):
    title = "Live: draw a box, watch it converge"
    subtitle = "cv2.grabCut, stepped one iteration at a time"
    interactive = True
    n_beats = 1
    beat_seconds = 0.75

    def __init__(self, image_path=None, sample=None):
        super().__init__()
        self.image_path = image_path
        self.sample = sample or RI.SAMPLE
        self.img = None
        self.session = None
        self.state = "idle"        # idle -> drawing -> running -> converged -> final
        self.morph = False
        self.drag0 = None
        self.rect = None
        self._prev_veil = None
        self._veil = None

    # ------------------------------------------------------------------ build
    def build(self):
        if self.img is None:
            self.img = RI.load_image(self.image_path, self.sample)
        img = self.img
        h, w = img.shape[:2]

        # --- the photo ----------------------------------------------------
        iax = self.add_axes([L, 0.095, 0.545, 0.725])
        T.blank(iax)
        iax.imshow(img, interpolation="bilinear")
        iax.set_xlim(0, w)
        iax.set_ylim(h, 0)
        iax.set_aspect("equal")
        self.iax = iax

        page = np.array(to_rgb(T.BG))
        self.veil = iax.imshow(np.zeros((h, w, 4)), interpolation="nearest",
                               zorder=4)
        self._page = page
        self.contour = LineCollection([], colors=T.GOLD, linewidths=2.2,
                                      zorder=6, capstyle="round")
        iax.add_collection(self.contour)
        self.band = Rectangle((0, 0), 0, 0, facecolor="none", edgecolor=T.GOLD,
                              lw=2.0, zorder=7, visible=False)
        iax.add_patch(self.band)
        self.prompt = iax.text(0.5, 0.5,
                               "drag a box around the object",
                               transform=iax.transAxes, ha="center", va="center",
                               fontsize=19, color=T.TEXT, zorder=9,
                               bbox=dict(boxstyle="round,pad=0.8",
                                         facecolor=T.BG, edgecolor=T.GOLD,
                                         linewidth=1.4, alpha=0.93))
        self.imgcap = iax.text(0.0, 1.028, "", transform=iax.transAxes,
                               ha="left", va="bottom", fontsize=11.5, color=T.DIM)

        # --- status block --------------------------------------------------
        sx = 0.625
        ax = self.add_axes([sx, 0.63, R_ - sx, 0.19])
        T.blank(ax)
        self.sax_txt = ax
        self.stage = ax.text(0.0, 0.94, "READY", transform=ax.transAxes,
                             va="center", fontsize=12.5, color=T.GOLD)
        # one baseline for the three numbers, one for the three captions
        NUM_Y, CAP_Y = 0.40, 0.31
        self.bignum = ax.text(0.0, NUM_Y, "–", transform=ax.transAxes,
                              va="bottom", fontsize=40, color=T.TEXT)
        self.bigcap = ax.text(0.0, CAP_Y, "iterations run", transform=ax.transAxes,
                              va="top", fontsize=12.5, color=T.MUTED)
        self.stat1 = ax.text(0.40, NUM_Y, "", transform=ax.transAxes, va="bottom",
                             fontsize=27, color=T.TEXT)
        self.stat1c = ax.text(0.40, CAP_Y, "pixels changed", transform=ax.transAxes,
                              va="top", fontsize=12.5, color=T.MUTED)
        self.stat2 = ax.text(0.71, NUM_Y, "", transform=ax.transAxes, va="bottom",
                             fontsize=27, color=T.GOLD)
        self.stat2c = ax.text(0.71, CAP_Y, "energy  E", transform=ax.transAxes,
                              va="top", fontsize=12.5, color=T.MUTED)
        for a in (self.stat1c, self.stat2c):
            a.set_alpha(0)

        # --- colour model ---------------------------------------------------
        i, j = RI.PROJ
        cax = self.add_axes([sx, 0.355, R_ - sx, 0.225])
        T.panel(cax, "colour model  ·  K = 5 per class, in RGB")
        cax.set_xlim(0, 1)
        cax.set_ylim(0, 1)
        self.cax = cax
        self.sc_b = cax.scatter([], [], s=5, c=T.SLATE, linewidths=0, zorder=3,
                                alpha=0.55)
        self.sc_f = cax.scatter([], [], s=5, c=T.BLUE, linewidths=0, zorder=4,
                                alpha=0.75)
        self.ell_f = D.EllipseSet(cax, RI.K_REAL, T.BLUE, proj=(i, j), zorder=8)
        self.ell_b = D.EllipseSet(cax, RI.K_REAL, T.SLATE, proj=(i, j), zorder=6)
        cax.text(0.975, 0.05, f"{RI.PROJ_NAMES[0]} →", transform=cax.transAxes,
                 ha="right", va="bottom", fontsize=11, color=T.DIM,
                 family="DejaVu Sans")
        cax.text(0.055, 0.05, f"{RI.PROJ_NAMES[1]} →", transform=cax.transAxes,
                 ha="left", va="bottom", rotation=90, fontsize=11, color=T.DIM,
                 family="DejaVu Sans")

        # --- energy ----------------------------------------------------------
        eax = self.add_axes([sx, 0.105, R_ - sx, 0.175])
        self.chart = D.EnergyChart(eax, 6, label="")
        eax.set_title("ENERGY  E,  RE-EVALUATED FROM EACH MASK", color=T.MUTED,
                      fontsize=10.5, loc="left", pad=9)
        eax.set_xlabel("")
        self.eax = eax

        # --- footnote ---------------------------------------------------------
        self.foot = self.fig_text(
            L, 0.038,
            "The GMMs above live in RGB, as in the paper and in cv2.grabCut. "
            "RGB ties brightness to hue, so a shadowed part of an object can sit "
            "a whole component away from its lit part;\nLab or HSV would separate "
            "those two clusters more cleanly. "
            "  ·   m  toggles the morphological open/close cleanup on the final "
            "mask.",
            fontsize=11, color=T.DIM, va="center", linespacing=1.7)

        self._refresh_keys()
        self._apply(None, 1.0)

    # ------------------------------------------------------------- appearance
    def _refresh_keys(self):
        k = {
            "idle": "drag a box on the photo   ·   ←  back to the maths   ·   q  quit",
            "drawing": "release to start",
            "running": "→  next iteration   ·   a  run to convergence   ·   "
                       "f  final   ·   r  reset",
            "converged": "f  final extraction   ·   →  another iteration   ·   r  reset",
            "final": "m  morphology on/off   ·   r  reset   ·   e  save frame   ·   q  quit",
        }[self.state]
        if self.runner:
            self.runner.hud_keys.set_text(k)

    def _veil_image(self, fg, strength=1.0, extract=False):
        """RGBA overlay: foreground untouched, background pushed to the page."""
        h, w = fg.shape
        out = np.zeros((h, w, 4))
        out[..., :3] = self._page if not extract else np.array(to_rgb(T.PANEL))
        a = np.where(fg, 0.0, 0.80 if not extract else 1.0)
        out[..., 3] = a * strength
        return out

    def _contour_of(self, mask):
        segs = []
        m = (mask.astype(np.uint8)) * 255
        cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < 40:
                continue
            p = c[:, 0, :].astype(float)
            segs.append(np.vstack([p, p[:1]]))
        return segs

    def _apply(self, rec, t):
        """Paint whatever the current state is, with transition progress t."""
        e = T.smooth(t)
        if self.session is None or rec is None:
            self.veil.set_data(np.zeros(self.img.shape[:2] + (4,)))
            self.contour.set_segments([])
            self.prompt.set_visible(self.state in ("idle", "drawing"))
            return
        self.prompt.set_visible(False)

        extract = (self.state == "final")
        fg = self.display_mask()
        new = self._veil_image(fg, 1.0, extract)
        if self._prev_veil is not None and self._prev_veil.shape == new.shape:
            cur = self._prev_veil + (new - self._prev_veil) * e
        else:
            cur = new * e
        self.veil.set_data(np.clip(cur, 0, 1))
        if e >= 1.0:
            self._prev_veil = new
        self.contour.set_segments(self._contour_of(fg))
        self.contour.set_alpha(0.0 if extract else 1.0)

    def display_mask(self):
        m = self.session.fg_mask
        if self.morph and self.state == "final":
            m = RI.clean_mask(m)
        return m

    def _update_panels(self, frac=1.0):
        s = self.session
        if s is None or not s.history:
            return
        rec = s.history[-1]
        i, j = RI.PROJ
        self.sc_f.set_offsets(rec["zf"][:, [i, j]] if len(rec["zf"]) else np.empty((0, 2)))
        self.sc_b.set_offsets(rec["zb"][:, [i, j]] if len(rec["zb"]) else np.empty((0, 2)))
        allz = np.vstack([z for z in (rec["zf"], rec["zb"]) if len(z)])
        for setlim, k in ((self.cax.set_xlim, i), (self.cax.set_ylim, j)):
            lo, hi = np.percentile(allz[:, k], [0.4, 99.6])
            pad = 0.17 * (hi - lo) + 0.02
            setlim(lo - pad, hi + pad)
        self.ell_f.update(*rec["fg_params"], alpha=1.0)
        self.ell_b.update(*rec["bg_params"], alpha=1.0)

        xs = [r["it"] for r in s.history]
        ys = [r["E"] for r in s.history]
        self.chart.n_max = max(6, len(xs))
        self.chart.set_data(xs, ys, frac)

        self.bignum.set_text(str(rec["it"]))
        self.stat1.set_text(f"{rec['changed']:,}")
        self.stat2.set_text(f"{rec['E']:,.0f}")
        for a in (self.stat1c, self.stat2c):
            a.set_alpha(1)
        self.stage.set_text({"running": "ITERATING",
                             "converged": "CONVERGED",
                             "final": "FOREGROUND EXTRACTED"}.get(self.state, "READY"))
        self.imgcap.set_text(
            f"{self.img.shape[1]} × {self.img.shape[0]}  ·  "
            f"{self.img.shape[0] * self.img.shape[1]:,} pixel nodes  ·  "
            f"box {self.rect[2]} × {self.rect[3]} px"
            + ("   ·   morphological cleanup ON" if (self.morph and self.state == "final")
               else ""))

    # ------------------------------------------------------------------ draw
    def draw(self, beat, t):
        self._apply(self.session.history[-1] if (self.session and self.session.history)
                    else None, t)
        if self.session and self.session.history:
            self._update_panels(frac=t)

    # ---------------------------------------------------------------- actions
    def _reset(self):
        self.session = None
        self.rect = None
        self.state = "idle"
        self.morph = False
        self.drag0 = None
        self._prev_veil = None
        self.band.set_visible(False)
        self.chart.set_data([], [])
        self.bignum.set_text("–")
        self.stat1.set_text("")
        self.stat2.set_text("")
        for a in (self.stat1c, self.stat2c):
            a.set_alpha(0)
        self.stage.set_text("READY")
        self.imgcap.set_text("")
        self.ell_f.set_alpha(0)
        self.ell_b.set_alpha(0)
        self.sc_f.set_offsets(np.empty((0, 2)))
        self.sc_b.set_offsets(np.empty((0, 2)))
        self._apply(None, 1.0)
        self._refresh_keys()

    def _step(self):
        if self.session is None:
            return
        self.session.step()
        self.state = "converged" if self.session.converged() else "running"
        self._refresh_keys()
        self.runner.replay()

    def _run_out(self, limit=8):
        while self.session and len(self.session.history) < limit \
                and not self.session.converged():
            self.session.step()
        self.state = "converged"
        self._refresh_keys()
        self.runner.replay()

    def _finalise(self):
        if self.session is None or not self.session.history:
            return
        self.state = "final"
        self._refresh_keys()
        self.runner.replay()

    # ----------------------------------------------------------------- events
    def on_press(self, e):
        if e.inaxes is not self.iax or self.state == "drawing":
            return False
        self.drag0 = (e.xdata, e.ydata)
        self.state = "drawing"
        self.band.set_visible(True)
        self.band.set_bounds(e.xdata, e.ydata, 0, 0)
        self.prompt.set_visible(False)
        self._refresh_keys()
        return True

    def on_motion(self, e):
        if self.state != "drawing" or self.drag0 is None or e.xdata is None:
            return False
        x0, y0 = self.drag0
        self.band.set_bounds(min(x0, e.xdata), min(y0, e.ydata),
                             abs(e.xdata - x0), abs(e.ydata - y0))
        return True

    def on_release(self, e):
        if self.state != "drawing" or self.drag0 is None:
            return False
        x0, y0 = self.drag0
        x1 = e.xdata if e.xdata is not None else x0
        y1 = e.ydata if e.ydata is not None else y0
        h, w = self.img.shape[:2]
        x, y = int(max(0, min(x0, x1))), int(max(0, min(y0, y1)))
        rw, rh = int(abs(x1 - x0)), int(abs(y1 - y0))
        rw, rh = min(rw, w - x - 1), min(rh, h - y - 1)
        self.drag0 = None
        if rw < 12 or rh < 12:                    # a stray click, not a box
            self.band.set_visible(False)
            self.state = "idle"
            self.prompt.set_visible(True)
            self._refresh_keys()
            return True
        self.rect = (x, y, rw, rh)
        self.session = RI.GrabCutSession(self.img, self.rect)
        self._prev_veil = None
        self.state = "running"
        self._step()
        return True

    def on_key(self, e):
        k = e.key
        if k == "r":
            self._reset()
            return True
        if k in ("right", " ", "enter"):
            if self.session is None:
                self.runner.set_note("draw a box first")
            else:
                self._step()
            return True
        if k == "a":
            self._run_out()
            return True
        if k == "f":
            self._finalise()
            return True
        if k == "m":
            if self.state != "final":
                self.runner.set_note("press f for the final extraction first")
            else:
                self.morph = not self.morph
                self._prev_veil = None
                self.runner.replay()
            return True
        if k in ("left", "backspace", "pageup"):
            self.band.set_visible(False)
            self.runner.goto_scene(self.runner.i - 1)
            return True
        return False
