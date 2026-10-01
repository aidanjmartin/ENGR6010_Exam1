"""Act 1: slope is not enough. Gradient descent vs Newton on a stretched bowl."""
from __future__ import annotations

import math

import numpy as np
import pygame

from acts.base import Act
from core import draw, text, theme
from core.anim import clamp, ease_in_out, ease_out, fade, seg
from core.mathtext import Eq, pulse
from core.widgets import Slider, legend
from sim import optim

PLOT = pygame.Rect(150, 150, 760, 760)
DOMAIN = 10.0
RIGHT_X = 1010


def to_px(p):
    s = PLOT.w / (2 * DOMAIN)
    return (PLOT.centerx + p[0] * s, PLOT.centery - p[1] * s)


def step_schedule(n, first=0.5, decay=0.9, floor=0.06):
    d = [max(floor, first * decay ** k) for k in range(n)]
    return np.concatenate([[0.0], np.cumsum(d)])


_contour_cache = {}


def contour_surface(gamma):
    """Filled, outlined contours of f for one gamma (cached; re-rendered while gamma animates)."""
    key = round(gamma, 3)
    surf = _contour_cache.get(key)
    if surf is not None:
        return surf
    if len(_contour_cache) > 4:
        _contour_cache.clear()
    surf = pygame.Surface(PLOT.size)
    surf.fill(theme.BG_PANEL)
    s = PLOT.w / (2 * DOMAIN)
    radii = [1.3 * k for k in range(1, 12)]
    sg = math.sqrt(gamma)
    cx, cy = PLOT.w / 2, PLOT.h / 2
    for i, r in enumerate(reversed(radii)):
        k = len(radii) - i
        a, b = r * s, r * s / sg
        shade = theme.mix(theme.BG_PANEL, theme.CURVATURE, 0.03 + 0.10 * (1 - k / len(radii)) ** 1.5)
        pygame.draw.ellipse(surf, shade, pygame.Rect(cx - a, cy - b, 2 * a, 2 * b))
    for k, r in enumerate(radii):
        a, b = r * s, r * s / sg
        pts = draw.ellipse_points((cx, cy), np.diag([a, b]), 160)
        draw.polyline(surf, theme.mix(theme.BG_PANEL, theme.CURVATURE, 0.42 - 0.02 * k), pts, 2, closed=True)
    _contour_cache[key] = surf
    return surf


def draw_plot_frame(c, gamma, lbl_x1, lbl_x2):
    c.blit(contour_surface(gamma), PLOT.topleft)
    old = c.get_clip()
    c.set_clip(PLOT)
    ox, oy = to_px((0, 0))
    pygame.draw.line(c, theme.GRID_MAJOR, (PLOT.left, oy), (PLOT.right, oy), 1)
    pygame.draw.line(c, theme.GRID_MAJOR, (ox, PLOT.top), (ox, PLOT.bottom), 1)
    draw.circle(c, theme.TEXT, (ox, oy), 7)
    draw.circle(c, theme.BG_PANEL, (ox, oy), 3)
    c.set_clip(old)
    pygame.draw.rect(c, theme.PANEL_EDGE, PLOT, 2, border_radius=4)
    lbl_x1.draw(c, (PLOT.right - 14, PLOT.bottom - 14), "bottomright", register=False)
    lbl_x2.draw(c, (PLOT.left + 14, PLOT.top + 10), "topleft", register=False)


class PathAnim:
    """Animated iterates with an eased move per step."""

    def __init__(self, path, first=0.5, decay=0.9, floor=0.06):
        self.path = np.asarray(path)
        self.times = step_schedule(len(self.path) - 1, first, decay, floor)

    @property
    def duration(self):
        return self.times[-1]

    def state(self, t):
        """(index of last completed step, position now)."""
        if t <= 0:
            return 0, self.path[0]
        if t >= self.duration:
            return len(self.path) - 1, self.path[-1]
        k = int(np.searchsorted(self.times, t, side="right") - 1)
        f = ease_in_out((t - self.times[k]) / (self.times[k + 1] - self.times[k]))
        return k, self.path[k] + f * (self.path[k + 1] - self.path[k])


class CurvatureAct(Act):
    number = 1
    title = "Slope is not enough"
    captions = [
        "On a round bowl, following the slope works.",
        "Stretch the bowl, and the slope points the wrong way.",
        "Newton uses curvature H.\nBut what if you cannot compute H?",
    ]
    settle = 12.0
    GAMMA_ANIM = (0.3, 2.2)     # start, duration of the 1 -> 25 stretch in beat 2
    RUN_DELAY = 0.5

    def __init__(self):
        super().__init__()
        self.gamma = 1.0
        self.run_t0 = 0.4
        self.animating_gamma = False
        self.dragging = False
        self.slider = Slider(RIGHT_X + 10, 846, 700, "drag to stretch the bowl", 1.0, 50.0, 25.0, "{:.0f}",
                             theme.CURVATURE, log=True, on_change=self._slide, on_release=self._release)
        self.eq_gd = Eq([r"x \leftarrow x \,-\,", (r"\alpha\,\nabla f", "g")], size=theme.MATH_BIG + 4,
                        colors={"g": theme.GRADIENT})
        self.eq_nt = Eq([r"x \leftarrow x \,-\,", (r"H^{-1}", "H"), r"\,", (r"\nabla f", "g")],
                        size=theme.MATH_BIG + 4, colors={"H": theme.CURVATURE, "g": theme.GRADIENT})
        self.eq_f = Eq([r"f(x) = \frac{1}{2}\left(x_1^2 + ", (r"\gamma", "gam"), r"\, x_2^2\right)"],
                       size=theme.MATH, colors={"gam": theme.CURVATURE})
        self.lbl_x1 = Eq([r"x_1"], size=theme.MATH_SMALL, color=theme.TEXT_DIM)
        self.lbl_x2 = Eq([r"x_2"], size=theme.MATH_SMALL, color=theme.TEXT_DIM)
        self._recompute(1.0)

    def equations(self):
        return [self.eq_gd, self.eq_nt, self.eq_f, self.lbl_x1, self.lbl_x2]

    # ---- state ------------------------------------------------------------------------
    def _recompute(self, gamma):
        self.gamma = gamma
        self.gd = PathAnim(optim.gradient_descent(gamma), 0.42, 0.9, 0.05)
        self.nt = PathAnim(optim.newton(gamma), 0.9)

    def on_beat(self):
        self.dragging = False
        if self.beat == 0:
            self._recompute(1.0)
            self.run_t0 = 0.5
        elif self.beat == 1:
            self._recompute(1.0)
            self.animating_gamma = True
            self.run_t0 = sum(self.GAMMA_ANIM) + self.RUN_DELAY
            self.slider.set(25.0)
        else:
            self.animating_gamma = False
            self._recompute(25.0)
            self.run_t0 = -100.0

    def update(self, dt):
        super().update(dt)
        if self.beat == 1 and self.animating_gamma:
            s0, d = self.GAMMA_ANIM
            f = seg(self.t, s0, d)
            g = math.exp(math.log(25.0) * f)
            if f >= 1.0:
                self.animating_gamma = False
                self._recompute(25.0)
            else:
                self.gamma = g

    def _slide(self, v):
        self.dragging = True
        self.animating_gamma = False
        self._recompute(v)

    def _release(self, v):
        self.dragging = False
        self._recompute(v)
        self.run_t0 = self.t + 0.15

    def handle(self, ev):
        if self.beat == 1 and self.slider.handle(ev):
            return True
        if ev.kind == "key" and ev.key == pygame.K_g:
            if self.beat != 1:
                self.enter(1)
            else:
                self.reset_beat()
            return True
        return False

    # ---- drawing ------------------------------------------------------------------------
    def _draw_plot(self, c):
        draw_plot_frame(c, self.gamma, self.lbl_x1, self.lbl_x2)

    def _draw_path(self, c, anim, t, color, dashed=False, marker="dot", width=4, show_grad=False):
        k, pos = anim.state(t)
        pts = [to_px(p) for p in anim.path[:k + 1]] + [to_px(pos)]
        old = c.get_clip()
        c.set_clip(PLOT.inflate(-4, -4))
        n = len(pts)
        # Older segments fade toward the background, a trail behind the moving
        # iterate; once the run is over the whole path comes back up.
        settled = clamp((t - anim.duration) / 0.8)
        for i in range(1, n):
            age = (n - 1 - i)
            a = 0.35 + 0.65 * math.exp(-age / 6.0)
            a = a + (0.85 - a) * settled if age > 0 else a
            col = theme.mix(theme.BG_PANEL, color, a)
            if dashed:
                draw.dashed_line(c, col, pts[i - 1], pts[i], width, 16, 10)
            else:
                draw.line(c, col, pts[i - 1], pts[i], width)
        for i, p in enumerate(pts[:-1]):
            if marker == "ring":
                draw.ring(c, color, p, 9, 3)
            else:
                draw.circle(c, theme.mix(theme.BG_PANEL, color, 0.8), p, 4.5)
        draw.glow(c, color, pts[-1], 30, 0.55)
        draw.circle(c, color, pts[-1], 9)
        if show_grad and 0 < t < anim.duration:
            g = optim.grad(np.asarray(pos), self.gamma)
            if np.linalg.norm(g) > 1e-6:
                d = -g / np.linalg.norm(g)
                p0 = np.array(to_px(pos))
                p1 = p0 + np.array([d[0], -d[1]]) * 95
                draw.dashed_line(c, theme.TEXT_FAINT, p0, to_px((0, 0)), 2, 8, 8)
                draw.arrow(c, theme.GRADIENT, p0, p1, 4, 16)
        c.set_clip(old)
        return k

    def draw(self, c):
        t = self.t
        self.draw_header(c)
        self._draw_plot(c)

        run_t = t - self.run_t0
        if self.dragging:
            run_t = 1e9
        k_gd = self._draw_path(c, self.gd, run_t, theme.GRADIENT, show_grad=self.beat == 1)
        k_nt = self._draw_path(c, self.nt, run_t, theme.CURVATURE, dashed=True, marker="ring", width=4)

        # ---- right column
        x = RIGHT_X
        la = fade(t, 0.1)
        self.eq_f.draw(c, (x, 190), "midleft", la, highlight={"gam": 0.9 * pulse(t)} if self.beat == 1 and self.animating_gamma else None)
        legend(c, [("gradient descent", theme.GRADIENT, "dot"),
                   ("Newton's method", theme.CURVATURE, "dasharrow"),
                   ("contours of f", theme.CURVATURE, "ellipse")], (x + 4, 280), la)

        if self.beat < 2:
            self._draw_counters(c, x, 450, k_gd, k_nt, run_t)
            ga = fade(t, 0.2)
            text.blit(c, f"γ = {self.gamma:.0f}" if self.gamma >= 1.5 else "γ = 1  (round bowl)",
                      theme.HEADING, theme.CURVATURE, (x, 700), "midleft", ga, name="gamma")
            text.blit(c, "condition number: how stretched the bowl is", theme.LABEL, theme.TEXT_DIM,
                      (x, 748), "midleft", ga, name="gamma-sub")
            if self.beat == 1:
                self.slider.set(self.gamma) if not self.dragging else None
                self.slider.draw(c, fade(t, 0.4))
        else:
            self._draw_rules(c, x)
        self.draw_caption(c)

    def _draw_counters(self, c, x, y, k_gd, k_nt, run_t):
        a = fade(self.t, 0.2)
        text.blit(c, "steps to reach the center", theme.LABEL, theme.TEXT_DIM, (x, y), "midleft", a, name="steps")
        rows = [("Gradient descent", theme.GRADIENT, k_gd, self.gd), ("Newton's method", theme.CURVATURE, k_nt, self.nt)]
        for i, (name, col, k, anim) in enumerate(rows):
            yy = y + 62 + i * 70
            text.blit(c, name, theme.BODY, col, (x, yy), "midleft", a, name="counter-name")
            done = run_t >= anim.duration
            n = k if run_t > 0 else 0
            s = f"{n}" + ("" if not done else "")
            text.blit(c, s, theme.HEADING + 8, col if done else theme.TEXT, (x + 700, yy), "midright", a,
                      name="counter")

    def _draw_rules(self, c, x):
        t = self.t
        a1 = fade(t, 0.3, 0.6)
        a2 = fade(t, 1.0, 0.6)
        dy1 = 16 * (1 - a1)
        dy2 = 16 * (1 - a2)
        text.blit(c, "Gradient descent", theme.BODY, theme.GRADIENT, (x, 480 + dy1), "midleft", a1, name="rule1")
        self.eq_gd.draw(c, (x, 550 + dy1), "midleft", a1)
        text.blit(c, "Newton's method", theme.BODY, theme.CURVATURE, (x, 680 + dy2), "midleft", a2, name="rule2")
        hl = seg(t, 1.9, 0.5) * pulse(t - 1.9)
        self.eq_nt.draw(c, (x, 750 + dy2), "midleft", a2, highlight={"H": hl})
        if hl > 0:
            r = self.eq_nt.key_rect("H", (x, 750 + dy2), "midleft")
            if r:
                text.blit(c, "curvature: needs every second derivative", theme.SMALL + 2, theme.CURVATURE,
                          (r.left, r.bottom + 22), "topleft", seg(t, 2.1, 0.5), name="H-note")
