"""Antialiased drawing primitives in virtual pixels.

Opacity on the dark background is done by blending colours toward BG (theme.dim),
which is exact on plain background and avoids per-primitive alpha surfaces.
"""
from __future__ import annotations

import math

import numpy as np
import pygame

from core import theme


def _pt(p):
    return (float(p[0]), float(p[1]))


def line(surf, color, p0, p1, width=2.0):
    if width <= 1.5:
        pygame.draw.aaline(surf, color, _pt(p0), _pt(p1))
    else:
        pygame.draw.aaline(surf, color, _pt(p0), _pt(p1), max(1, int(round(width))))


def polyline(surf, color, pts, width=2.0, closed=False):
    pts = [_pt(p) for p in pts]
    if len(pts) < 2:
        return
    if width <= 1.5:
        pygame.draw.aalines(surf, color, closed, pts)
        return
    w = max(1, int(round(width)))
    n = len(pts) if closed else len(pts) - 1
    for i in range(n):
        pygame.draw.aaline(surf, color, pts[i], pts[(i + 1) % len(pts)], w)
    if w >= 4:
        r = w / 2 - 0.5
        for p in (pts if closed else pts[1:-1]):
            pygame.draw.aacircle(surf, color, p, r)


def dashed_line(surf, color, p0, p1, width=2.0, dash=14.0, gap=9.0, phase=0.0):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.hypot(*d))
    if L < 1e-6:
        return
    u = d / L
    s = -phase % (dash + gap)
    s = s - (dash + gap) if s > 0 else s
    while s < L:
        a, b = max(s, 0.0), min(s + dash, L)
        if b > a:
            line(surf, color, p0 + u * a, p0 + u * b, width)
        s += dash + gap


def dashed_polyline(surf, color, pts, width=2.0, dash=14.0, gap=9.0, closed=False):
    pts = [np.asarray(p, float) for p in pts]
    if closed:
        pts = pts + [pts[0]]
    carry = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        dashed_line(surf, color, a, b, width, dash, gap, phase=carry)
        carry += float(np.hypot(*(b - a)))


def circle(surf, color, c, r):
    pygame.draw.aacircle(surf, color, _pt(c), max(0.5, r))


def ring(surf, color, c, r, width=2):
    pygame.draw.aacircle(surf, color, _pt(c), max(1.0, r), max(1, int(round(width))))


def polygon(surf, color, pts):
    pts = [_pt(p) for p in pts]
    pygame.draw.polygon(surf, color, pts)
    pygame.draw.aalines(surf, color, True, pts)


def arrow(surf, color, p0, p1, width=4.0, head=18.0, dashed=False, head_width=None):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.hypot(*d))
    if L < 1.0:
        return
    u = d / L
    nrm = np.array([-u[1], u[0]])
    head = min(head, 0.6 * L)
    hw = (head_width or head * 0.62)
    base = p1 - u * head
    if dashed:
        dashed_line(surf, color, p0, base + u * 2, width, dash=12, gap=8)
    else:
        line(surf, color, p0, base + u * 2, width)
    polygon(surf, color, [p1, base + nrm * hw, base - nrm * hw])


def tapered_stroke(surf, color, pts, w0, w1, edge=None):
    """Filled stroke whose width goes from w0 at the first point to w1 at the last."""
    P = np.asarray(pts, float)
    n = len(P)
    if n < 2:
        return
    T = np.gradient(P, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
    N = np.column_stack([-T[:, 1], T[:, 0]])
    w = np.linspace(w0, w1, n)[:, None] / 2
    left = P + N * w
    right = P - N * w
    poly = list(map(tuple, left)) + list(map(tuple, right[::-1]))
    pygame.draw.polygon(surf, color, poly)
    pygame.draw.aalines(surf, edge or color, True, poly)


def tube(surf, pts, radius, shades):
    """Soft-shaded tube: concentric strokes from dark rim to light core."""
    P = [_pt(p) for p in pts]
    k = len(shades)
    for i, col in enumerate(shades):
        r = radius * (1 - i / k)
        w = max(1, int(2 * r))
        for a, b in zip(P[:-1], P[1:]):
            pygame.draw.line(surf, col, a, b, w)
        for p in P:
            pygame.draw.circle(surf, col, (int(round(p[0])), int(round(p[1]))), max(1, int(r)))
    # Antialias the outer silhouette.
    Pn = np.asarray(P)
    T = np.gradient(Pn, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
    N = np.column_stack([-T[:, 1], T[:, 0]])
    pygame.draw.aalines(surf, shades[0], False, list(map(tuple, Pn + N * radius)))
    pygame.draw.aalines(surf, shades[0], False, list(map(tuple, Pn - N * radius)))


_glow_cache: dict = {}


def glow(surf, color, c, radius, strength=1.0):
    radius = max(2, int(radius))
    key = (tuple(color), radius)
    g = _glow_cache.get(key)
    if g is None:
        size = radius * 2
        yy, xx = np.mgrid[0:size, 0:size]
        d = np.hypot(xx - radius + 0.5, yy - radius + 0.5) / radius
        a = np.clip(1 - d, 0, 1) ** 2.2
        g = pygame.Surface((size, size), pygame.SRCALPHA)
        g.fill((*color, 0))
        pa = pygame.surfarray.pixels_alpha(g)
        pa[:] = (a.T * 255).astype(np.uint8)
        del pa
        _glow_cache[key] = g
    g.set_alpha(int(255 * max(0.0, min(1.0, strength))))
    surf.blit(g, (c[0] - radius, c[1] - radius))
    g.set_alpha(255)   # keep per-pixel alpha (None would drop it)


def crosshair(surf, color, c, r=16, width=2):
    x, y = c
    ring(surf, color, c, r, width)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        line(surf, color, (x + dx * r * 0.45, y + dy * r * 0.45), (x + dx * r * 1.55, y + dy * r * 1.55), width)


def warning_icon(surf, color, c, size=28):
    x, y = c
    h = size * 0.9
    pts = [(x, y - h / 2), (x + size / 2, y + h / 2), (x - size / 2, y + h / 2)]
    polygon(surf, color, pts)
    pygame.draw.line(surf, theme.BG, (x, y - h / 2 + size * 0.3), (x, y + h / 2 - size * 0.34), max(2, int(size / 9)))
    pygame.draw.circle(surf, theme.BG, (int(x), int(y + h / 2 - size * 0.18)), max(1, int(size / 14)))


def panel(surf, rect, fill=theme.BG_PANEL, edge=theme.PANEL_EDGE, radius=18, width=2):
    pygame.draw.rect(surf, fill, rect, border_radius=radius)
    if edge is not None:
        pygame.draw.rect(surf, edge, rect, width=width, border_radius=radius)


def ellipse_points(center, M, n=96):
    """Points c + M [cos t, sin t] for a 2x2 matrix M (virtual pixels)."""
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    circ = np.stack([np.cos(t), np.sin(t)])
    return (np.asarray(center, float)[:, None] + M @ circ).T


def brace_h(surf, color, x0, x1, y, depth=14, width=2, up=False):
    """Horizontal curly brace under (or over) a span."""
    s = -1 if up else 1
    xm = (x0 + x1) / 2
    pts = []
    for (a, b) in ((x0, xm), (xm, x1)):
        for k in range(17):
            t = k / 16
            x = a + (b - a) * t
            # quarter-circle shoulders, point at the middle
            if a == x0:
                yy = y + s * depth * (0.5 * (1 - math.cos(math.pi * min(t * 4, 1))) * 0.5
                                      + (0.5 if t > 0.9 else 0) * (t - 0.9) * 10)
            else:
                yy = y + s * depth * (0.5 * (1 - math.cos(math.pi * min((1 - t) * 4, 1))) * 0.5
                                      + (0.5 if t < 0.1 else 0) * (0.1 - t) * 10)
            pts.append((x, yy))
    polyline(surf, color, pts, width)
