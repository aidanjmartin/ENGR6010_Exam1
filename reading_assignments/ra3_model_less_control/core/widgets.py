"""Slider, legend, sparkline and bar widgets. All geometry in virtual pixels."""
from __future__ import annotations

import math

import pygame

from core import draw, text, theme
from core.text import REC


class Slider:
    def __init__(self, x, y, w, label, vmin, vmax, value, fmt="{:.2f}", color=theme.TEXT,
                 log=False, on_change=None, on_release=None):
        self.x, self.y, self.w = x, y, w
        self.label, self.fmt, self.color = label, fmt, color
        self.vmin, self.vmax, self.log = vmin, vmax, log
        self.value = value
        self.on_change, self.on_release = on_change, on_release
        self.dragging = False
        self.visible = True
        self.hover = False

    # value <-> fraction along the track
    def _frac(self, v):
        if self.log:
            return (math.log(v) - math.log(self.vmin)) / (math.log(self.vmax) - math.log(self.vmin))
        return (v - self.vmin) / (self.vmax - self.vmin)

    def _value(self, f):
        f = min(1.0, max(0.0, f))
        if self.log:
            return math.exp(math.log(self.vmin) + f * (math.log(self.vmax) - math.log(self.vmin)))
        return self.vmin + f * (self.vmax - self.vmin)

    def hit_rect(self):
        return pygame.Rect(self.x - 18, self.y - 22, self.w + 36, 44)

    def set(self, v, notify=False):
        self.value = min(self.vmax, max(self.vmin, v))
        if notify and self.on_change:
            self.on_change(self.value)

    def handle(self, ev):
        if not self.visible:
            return False
        if ev.kind == "down" and ev.pos and self.hit_rect().collidepoint(ev.pos):
            self.dragging = True
            self.set(self._value((ev.pos[0] - self.x) / self.w), notify=True)
            return True
        if ev.kind == "motion" and ev.pos:
            self.hover = self.hit_rect().collidepoint(ev.pos)
            if self.dragging:
                self.set(self._value((ev.pos[0] - self.x) / self.w), notify=True)
                return True
        if ev.kind == "up" and self.dragging:
            self.dragging = False
            if self.on_release:
                self.on_release(self.value)
            return True
        return False

    def draw(self, surf, alpha=1.0):
        if not self.visible or alpha <= 0.01:
            return
        c = theme.dim(self.color, alpha)
        track = theme.dim(theme.PANEL_EDGE, alpha * 1.4 if alpha < 0.7 else 1.0)
        f = self._frac(self.value)
        hx = self.x + f * self.w
        pygame.draw.line(surf, track, (self.x, self.y), (self.x + self.w, self.y), 6)
        pygame.draw.line(surf, theme.dim(self.color, 0.55 * alpha), (self.x, self.y), (hx, self.y), 6)
        r = 14 if (self.dragging or self.hover) else 12
        draw.circle(surf, c, (hx, self.y), r)
        draw.circle(surf, theme.dim(theme.BG, 1.0), (hx, self.y), r - 5)
        draw.circle(surf, c, (hx, self.y), r - 8)
        lr = text.blit(surf, self.label, theme.SMALL, theme.dim(theme.TEXT_DIM, alpha),
                       (self.x, self.y - 22), "bottomleft", name="slider-label")
        vr = text.blit(surf, self.fmt.format(self.value), theme.SMALL, c,
                       (self.x + self.w, self.y - 22), "bottomright", name="slider-value")
        REC.add(pygame.Rect(self.x - 16, lr.top, self.w + 32, self.y + 16 - lr.top), "ui", "slider")


def legend(surf, items, pos, alpha=1.0, size=theme.SMALL + 2, row=40, swatch=46):
    """items: (label, colour, style) with style in line | dash | dot | arrow | ellipse."""
    x, y = pos
    rects = []
    for i, (label, color, style) in enumerate(items):
        cy = y + i * row
        c = theme.dim(color, alpha)
        if style == "dash":
            draw.dashed_line(surf, c, (x, cy), (x + swatch, cy), 3, dash=10, gap=6)
        elif style == "dot":
            draw.circle(surf, c, (x + swatch / 2, cy), 7)
        elif style == "arrow":
            draw.arrow(surf, c, (x, cy), (x + swatch, cy), 3, 13)
        elif style == "dasharrow":
            draw.arrow(surf, c, (x, cy), (x + swatch, cy), 3, 13, dashed=True)
        elif style == "ellipse":
            pygame.draw.ellipse(surf, c, (x + 4, cy - 9, swatch - 8, 18), 3)
        elif style == "ring":
            draw.ring(surf, c, (x + swatch / 2, cy), 9, 3)
        else:
            draw.line(surf, c, (x, cy), (x + swatch, cy), 4)
        rects.append(text.blit(surf, label, size, theme.dim(theme.TEXT, alpha),
                               (x + swatch + 16, cy), "midleft", name="legend"))
    return rects


def sparkline(surf, values, rect, color, vmax, alpha=1.0, warn_above=None):
    rect = pygame.Rect(rect)
    pygame.draw.line(surf, theme.dim(theme.PANEL_EDGE, alpha), rect.bottomleft, rect.bottomright, 2)
    if len(values) < 2:
        return
    n = len(values)
    pts = []
    for i, v in enumerate(values):
        x = rect.right - (n - 1 - i) * rect.w / 359.0
        if x < rect.left:
            continue
        y = rect.bottom - min(v, vmax) / vmax * rect.h
        pts.append((x, y))
    if len(pts) >= 2:
        fill = [(pts[0][0], rect.bottom)] + pts + [(pts[-1][0], rect.bottom)]
        pygame.draw.polygon(surf, theme.dim(color, 0.16 * alpha), fill)
        draw.polyline(surf, theme.dim(color, alpha), pts, 3)
        draw.circle(surf, theme.dim(color, alpha), pts[-1], 5)


def hbar(surf, rect, frac, color, alpha=1.0):
    rect = pygame.Rect(rect)
    pygame.draw.rect(surf, theme.dim(theme.PANEL_EDGE, alpha), rect, border_radius=rect.h // 2)
    w = max(rect.h, int(rect.w * min(1.0, max(0.0, frac))))
    pygame.draw.rect(surf, theme.dim(color, alpha), (rect.x, rect.y, w, rect.h), border_radius=rect.h // 2)
