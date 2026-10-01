"""Act 0: the title. A continuum robot draws itself and reaches for a target."""
from __future__ import annotations

import numpy as np

from acts import robotdraw as rd
from acts.base import Act
from core import draw, text, theme
from core.anim import ease_in_out, ease_out, ease_out_back, fade, seg
from sim import robot as rb


class TitleAct(Act):
    number = 0
    title = "When the Model Is Wrong"
    captions = [""]

    PHI = 0.42
    Y_FINAL = np.array([52.0, 17.5, rb.Y_SLACK])

    def __init__(self):
        super().__init__()
        import pygame
        self.view = rd.View(pygame.Rect(1080, 60, 760, 960), (10.0, 30.0), 4.6)
        self.target_mm = rb.tip(self.Y_FINAL, self.PHI, 1.0)
        self.trail = []

    def on_beat(self):
        self.trail = []

    def _state(self, t):
        phi = self.PHI * seg(t, 0.9, 1.2)
        ins = 52.0 * seg(t, 1.3, 1.1, ease_out)
        bend = ease_out_back(max(0.0, (t - 2.3) / 1.5), 1.2) if t > 2.3 else 0.0
        y = np.array([ins, rb.Y_SLACK + (self.Y_FINAL[1] - rb.Y_SLACK) * bend, rb.Y_SLACK])
        return y, phi

    def update(self, dt):
        super().update(dt)
        if 1.3 < self.t < 4.6:
            y, phi = self._state(self.t)
            self.trail.append(rb.tip(y, phi, 1.0))
            self.trail = self.trail[-70:]

    def _edge_fade(self):
        """Background-coloured vignette so the grid dissolves into the page."""
        if getattr(self, "_fade_surf", None) is None:
            import pygame
            w, h = self.view.rect.size
            yy, xx = np.mgrid[0:h, 0:w]
            dx = np.minimum(xx, w - 1 - xx) / 180.0
            dy = np.minimum(yy, h - 1 - yy) / 180.0
            a = 1 - np.clip(np.minimum(dx, dy), 0, 1) ** 1.5
            s = pygame.Surface((w, h), pygame.SRCALPHA)
            s.fill((*theme.BG, 0))
            pa = pygame.surfarray.pixels_alpha(s)
            pa[:] = (a.T * 255).astype(np.uint8)
            del pa
            self._fade_surf = s
        return self._fade_surf

    def draw(self, c):
        t = self.t
        v = self.view
        # faint workspace grid behind the robot, fading toward the edges
        rd.grid(c, v, 0.55 * fade(t, 0.0, 1.0))
        c.blit(self._edge_fade(), v.rect.topleft)
        y, phi = self._state(t)
        rd.endoscope(c, v, phi, 1.0, reveal=seg(t, 0.0, 1.1, ease_out))
        if t > 1.3:
            rd.trail(c, v, self.trail, theme.ESTIMATE, 0.7 * (1 - seg(t, 4.6, 1.0)))
            rd.instrument(c, v, y, phi, 1.0, theme.ESTIMATE, 1.0, glow=seg(t, 3.4, 0.8))
        ta = fade(t, 1.9, 0.6)
        if ta > 0:
            rd.target(c, v, self.target_mm, ta, t)

        x = 150
        text.blit(c, self.title, 64, theme.TEXT, (x, 400), "bottomleft", fade(t, 0.5, 0.9), name="title")
        w = 470 * seg(t, 0.9, 1.0)
        if w > 1:
            draw.line(c, theme.ESTIMATE, (x + 2, 440), (x + 2 + w, 440), 3)
        text.blit(c, "Optimization for continuum robots", theme.SUBTITLE + 4, theme.TEXT_DIM,
                  (x, 478), "topleft", fade(t, 1.3, 0.8), name="subtitle")
        text.blit(c, "with uncertain kinematics", theme.SUBTITLE + 4, theme.TEXT_DIM,
                  (x, 530), "topleft", fade(t, 1.4, 0.8), name="subtitle2")
        text.blit(c, "Built on Yip and Camarillo, IEEE Transactions on Robotics, 2014",
                  theme.SMALL + 2, theme.TEXT_FAINT, (x, 980), "bottomleft", fade(t, 2.4, 0.8), name="footer")
        text.blit(c, "ENGR 6010  ·  AI in Robotics", theme.SMALL + 2, theme.TEXT_FAINT,
                  (x, 1020), "bottomleft", fade(t, 2.6, 0.8), name="footer2")
