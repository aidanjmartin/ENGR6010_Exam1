"""Drawing the endoscope, the continuum instrument and its Jacobian arrows."""
from __future__ import annotations

import math

import numpy as np
import pygame

from core import draw, text, theme
from sim import robot as rb

ENDO_SHADES = [(36, 43, 60), (50, 59, 80), (64, 75, 100), (80, 93, 122), (98, 112, 142),
               (118, 132, 162), (136, 150, 178)]


class View:
    """Millimetre workspace (y up) -> virtual pixels inside a panel rectangle."""

    def __init__(self, rect, center_mm, scale):
        self.rect = pygame.Rect(rect)
        self.center_mm = np.asarray(center_mm, float)
        self.scale = float(scale)

    def px(self, p):
        p = np.asarray(p, float)
        c = self.rect.center
        return (c[0] + (p[..., 0] - self.center_mm[0]) * self.scale,
                c[1] - (p[..., 1] - self.center_mm[1]) * self.scale)

    def pts(self, P):
        x, y = self.px(np.asarray(P, float))
        return np.column_stack([x, y])

    def mm(self, pos):
        c = self.rect.center
        return np.array([(pos[0] - c[0]) / self.scale + self.center_mm[0],
                         -(pos[1] - c[1]) / self.scale + self.center_mm[1]])

    def vec(self, v):
        """A workspace vector (mm) in pixels, y flipped."""
        return np.array([v[0], -v[1]]) * self.scale

    @staticmethod
    def lerp(a, b, t):
        r = pygame.Rect(round(a.rect.x + (b.rect.x - a.rect.x) * t), round(a.rect.y + (b.rect.y - a.rect.y) * t),
                        round(a.rect.w + (b.rect.w - a.rect.w) * t), round(a.rect.h + (b.rect.h - a.rect.h) * t))
        return View(r, a.center_mm + (b.center_mm - a.center_mm) * t,
                    math.exp(math.log(a.scale) + (math.log(b.scale) - math.log(a.scale)) * t))


def grid(surf, view, alpha=1.0, step=10.0):
    r = view.rect
    old = surf.get_clip()
    surf.set_clip(r.clip(old) if old else r)
    lo = view.mm((r.left, r.bottom))
    hi = view.mm((r.right, r.top))
    x = math.floor(lo[0] / step) * step
    while x <= hi[0]:
        px = view.px(np.array([x, 0.0]))[0]
        major = abs(x / 50 - round(x / 50)) < 1e-6
        pygame.draw.line(surf, theme.dim(theme.GRID_MAJOR if major else theme.GRID, alpha), (px, r.top), (px, r.bottom), 1)
        x += step
    y = math.floor(lo[1] / step) * step
    while y <= hi[1]:
        py = view.px(np.array([0.0, y]))[1]
        major = abs(y / 50 - round(y / 50)) < 1e-6
        pygame.draw.line(surf, theme.dim(theme.GRID_MAJOR if major else theme.GRID, alpha), (r.left, py), (r.right, py), 1)
        y += step
    surf.set_clip(old)


def endoscope(surf, view, phi, alpha=1.0, reveal=1.0):
    bottom = view.mm((0, view.rect.bottom + 40))[1]
    P = rb.endoscope_centerline(phi, bottom)
    pts = view.pts(P)
    if reveal < 1.0:
        pts = _partial(pts, reveal)
    shades = [theme.dim(c, alpha) for c in ENDO_SHADES]
    draw.tube(surf, pts, rb.ENDO_RADIUS * view.scale, shades)
    if reveal >= 1.0:
        E, psi = rb.exit_pose(phi)
        n = np.array([-math.sin(psi), math.cos(psi)])
        a = view.px(E + n * rb.ENDO_RADIUS * 0.92)
        b = view.px(E - n * rb.ENDO_RADIUS * 0.92)
        draw.line(surf, theme.dim((170, 184, 210), alpha), a, b, 4)
        lens = E - n * rb.ENDO_RADIUS * 0.45
        draw.circle(surf, theme.dim((120, 190, 255), alpha * 0.9), view.px(lens), max(2.0, 0.9 * view.scale))


def _partial(pts, f):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    L = cum[-1] * f
    out = [pts[0]]
    for i in range(1, len(pts)):
        if cum[i] <= L:
            out.append(pts[i])
        else:
            t = (L - cum[i - 1]) / max(seg[i - 1], 1e-9)
            out.append(pts[i - 1] + t * (pts[i] - pts[i - 1]))
            break
    if len(out) < 2:
        out.append(out[0])
    return np.array(out)


def instrument(surf, view, y, phi, g, color, alpha=1.0, glow=0.0):
    P = rb.backbone(y, phi, g, n=40)
    pts = view.pts(P)
    s = view.scale
    col = theme.dim(color, alpha)
    dark = theme.dim(theme.mix(color, theme.BG, 0.55), alpha)
    if glow > 0:
        draw.glow(surf, color, pts[-1], 34, 0.35 * glow * alpha)
    draw.tapered_stroke(surf, dark, pts, 3.8 * s, 2.6 * s)
    draw.tapered_stroke(surf, col, pts, 2.6 * s, 1.7 * s)
    # vertebrae
    L = rb.L0 + y[0]
    k = max(2, int(L / 7.0))
    for i in range(1, k):
        j = int(round(i / k * (len(pts) - 1)))
        j = min(max(j, 1), len(pts) - 2)
        t = pts[j + 1] - pts[j - 1]
        t = t / max(np.linalg.norm(t), 1e-9)
        n = np.array([-t[1], t[0]])
        hw = (3.4 - 1.0 * j / len(pts)) * s / 2
        draw.line(surf, dark, pts[j] + n * hw, pts[j] - n * hw, 2)
    draw.circle(surf, col, pts[-1], max(4.0, 1.5 * s))
    draw.circle(surf, theme.dim(theme.BG, 1.0), pts[-1], max(1.5, 0.6 * s))
    return pts[-1]


def jacobian_arrows(surf, view, tip_px, J_w, colors, alpha=1.0, dashed=False, length=74.0,
                    labels=True, label_size=theme.SMALL, width=4, grow=None):
    """Arrows for the columns of J W (tip motion per normalised actuator step)."""
    ends = []
    for i in range(3):
        g = 1.0 if grow is None else grow[i]
        if g <= 0.01:
            ends.append(None)
            continue
        v = view.vec(J_w[:, i]) * length / view.scale * g
        p1 = np.asarray(tip_px) + v
        col = theme.dim(colors[i], alpha)
        draw.arrow(surf, col, tip_px, p1, width, 17, dashed=dashed)
        ends.append(p1)
        if labels and np.linalg.norm(v) > 20:
            u = v / np.linalg.norm(v)
            lp = p1 + u * 24
            text.blit(surf, theme.COL_SHORT[i], label_size, col, lp, "center", kind="scene",
                      name="col-label")
    return ends


def target(surf, view, p_mm, alpha=1.0, t=0.0):
    c = view.px(p_mm)
    r = 15 + 1.5 * math.sin(t * 3.0)
    draw.crosshair(surf, theme.dim(theme.TARGET, alpha), c, r, 2)


def trail(surf, view, P, color, alpha=1.0, width=3):
    if len(P) < 2:
        return
    pts = view.pts(np.asarray(P))
    n = len(pts)
    for i in range(1, n):
        a = alpha * (i / n) ** 1.6
        draw.line(surf, theme.dim(color, 0.8 * a), pts[i - 1], pts[i], width)
