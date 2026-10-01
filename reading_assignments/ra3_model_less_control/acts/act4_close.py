"""Act 4: three takeaways, revealed one per Space press."""
from __future__ import annotations

import math

import numpy as np
import pygame

from acts import robotdraw as rd
from acts.base import Act
from core import draw, text, theme
from core.anim import ease_out, fade, seg
from sim import optim
from sim import robot as rb

TAKEAWAYS = [
    (theme.GRADIENT, "Gradient steps only see slope.",
     "Curvature-aware steps handle stretched problems."),
    (theme.ESTIMATE, "Secant updates estimate a model from observed motion:",
     "the smallest correction that explains the newest measurement."),
    (theme.ESTIMATE, "An estimated Jacobian plus a small constrained solve at every step",
     "keeps a continuum robot stable when anatomy breaks the model."),
]
ROW_Y = [285, 505, 725]
ICON = 170


class CloseAct(Act):
    number = 4
    title = "Three things to keep"
    captions = [""]
    settle = 4.0

    def builds_in(self, beat):
        return 2

    def draw(self, c):
        self.draw_header(c)
        for i, (col, l1, l2) in enumerate(TAKEAWAYS):
            if i > self.build:
                break
            t = self.t if i == 0 else (self.build_t if i == self.build else 99.0)
            a = fade(t, 0.15 if i == 0 else 0.0, 0.55)
            dx = 30 * (1 - a)
            y = ROW_Y[i]
            box = pygame.Rect(150 + dx, y - ICON // 2, ICON, ICON)
            draw.panel(c, box, theme.dim(theme.BG_PANEL, a), theme.dim(theme.PANEL_EDGE, a), 14)
            old = c.get_clip()
            c.set_clip(box.inflate(-4, -4))
            [self._icon_curvature, self._icon_secant, self._icon_robot][i](c, box, a, t)
            c.set_clip(old)
            tx = box.right + 60
            text.blit(c, l1, theme.BODY + 2, theme.TEXT, (tx, y - 24), "midleft", a, name=f"take{i}a")
            text.blit(c, l2, theme.BODY + 2, theme.TEXT_DIM, (tx, y + 24), "midleft", a, name=f"take{i}b")
            pygame.draw.line(c, theme.dim(col, a), (tx - 28, y - 44), (tx - 28, y + 44), 4)
        fa = fade(self.t, 0.6, 0.8)
        text.blit(c, "Yip and Camarillo, “Model-Less Feedback Control of Continuum Manipulators in "
                     "Constrained Environments,” IEEE T-RO 30(4), 2014",
                  theme.SMALL, theme.TEXT_FAINT, (150, 952), "midleft", fa, name="ref1")
        text.blit(c, "Follow-up: Mo et al., Cyborg and Bionic Systems, 2022 — population-based secant control "
                     "for laser-assisted endoscopic surgery",
                  theme.SMALL, theme.TEXT_FAINT, (150, 992), "midleft", fa, name="ref2")

    # ---- small pictures that recall each act ----------------------------------------------
    def _icon_curvature(self, c, box, a, t):
        cx, cy = box.center
        for k in range(1, 6):
            rx, ry = 14 * k, 14 * k / 3.2
            pts = draw.ellipse_points((cx, cy), np.diag([rx, ry]), 64)
            draw.polyline(c, theme.mix(theme.BG_PANEL, theme.CURVATURE, 0.45 * a), pts, 2, closed=True)
        path = optim.gradient_descent(10.0, x0=np.array([-8.8, 2.4]))
        s = 7.5
        pts = [(cx + p[0] * s, cy - p[1] * s) for p in path]
        n = max(2, int(len(pts) * seg(t, 0.2, 1.2)))
        draw.polyline(c, theme.dim(theme.GRADIENT, a), pts[:n], 3)
        f = seg(t, 0.4, 0.8)
        p0 = np.array(pts[0])
        draw.dashed_line(c, theme.dim(theme.CURVATURE, a), p0, p0 + f * (np.array([cx, cy]) - p0), 3, 8, 6)

    def _icon_secant(self, c, box, a, t):
        x0, y0 = box.left + 20, box.bottom - 30
        pts = [(x0 + u * 130, y0 - (0.3 * u + 0.9 * u * u) * 110) for u in np.linspace(0, 1, 40)]
        draw.polyline(c, theme.dim(theme.TRUTH, a), pts, 3)
        ia, ib = 12, 34 - int(14 * seg(t, 0.3, 1.2))
        pa, pb = np.array(pts[ia]), np.array(pts[ib])
        d = (pb - pa) / np.linalg.norm(pb - pa)
        draw.line(c, theme.dim(theme.ESTIMATE, a), pa - d * 60, pb + d * 60, 3)
        for p in (pa, pb):
            draw.circle(c, theme.dim(theme.TRUTH, a), p, 6)

    def _icon_robot(self, c, box, a, t):
        view = rd.View(box, (2.0, 2.0), 1.6)
        rd.endoscope(c, view, 0.9, a)
        bend = 11.0 * seg(t, 0.2, 1.2, ease_out)
        y = np.array([45.0, rb.Y_SLACK + bend, rb.Y_SLACK])
        tip = rd.instrument(c, view, y, 0.9, 1.0, theme.ESTIMATE, a)
        J = rb.jacobian(y, 0.9, 1.0) * np.array([1.0, 0.28, 0.28])
        rd.jacobian_arrows(c, view, tip, J, theme.ESTIMATE_COLS, a, labels=False, length=38, width=3)
