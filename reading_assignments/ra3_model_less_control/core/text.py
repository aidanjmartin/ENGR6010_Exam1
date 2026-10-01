"""UI font, cached text surfaces, and the layout recorder used by the layout test.

Text is always rendered at virtual resolution and cached; the present step scales
the finished canvas, never the text.
"""
from __future__ import annotations

import os
from collections import OrderedDict

import pygame

from core import theme

# (pygame family name, token that must appear in the resolved file name)
_FALLBACKS = [
    ("inter", "inter"),
    ("segoeui", "segoe"),
    ("helveticaneue", "helveticaneue"),
    ("arial", "arial"),
]
_font_path = None
_fonts: dict = {}
_cache: OrderedDict = OrderedDict()
_CACHE_MAX = 1500


def _resolve_font():
    global _font_path
    if _font_path is not None:
        return _font_path
    for name, token in _FALLBACKS:
        path = pygame.font.match_font(name)
        if path and token in os.path.basename(path).lower().replace(" ", ""):
            _font_path = path
            return path
    try:  # matplotlib always ships DejaVu Sans
        import matplotlib
        _font_path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
    except Exception:
        _font_path = ""
    return _font_path


def font(size: int, bold: bool = False) -> pygame.font.Font:
    key = (size, bold)
    f = _fonts.get(key)
    if f is None:
        path = _resolve_font()
        f = pygame.font.Font(path or None, size)
        f.bold = bold
        _fonts[key] = f
    return f


def render(s: str, size: int, color, bold: bool = False) -> pygame.Surface:
    key = (s, size, tuple(color), bold)
    surf = _cache.get(key)
    if surf is None:
        surf = font(size, bold).render(s, True, color)
        _cache[key] = surf
        if len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    else:
        _cache.move_to_end(key)
    return surf


def size_of(s: str, size: int, bold: bool = False):
    return render(s, size, (255, 255, 255), bold).get_size()


# ---- layout recorder -------------------------------------------------------------

class LayoutRecorder:
    """Collects the rectangles of registered text and UI elements during one draw."""

    def __init__(self):
        self.enabled = False
        self.items = []

    def start(self):
        self.enabled = True
        self.items = []

    def stop(self):
        self.enabled = False
        return self.items

    def add(self, rect, kind="text", name=""):
        if self.enabled:
            self.items.append((pygame.Rect(rect), kind, name))


REC = LayoutRecorder()


def anchored(rect: pygame.Rect, pos, anchor: str) -> pygame.Rect:
    setattr(rect, anchor, (round(pos[0]), round(pos[1])))
    return rect


def blit(target, s: str, size: int, color, pos, anchor="topleft", alpha=1.0,
         bold=False, kind="text", name=None) -> pygame.Rect:
    """Draw a string. Registers its rectangle with the layout recorder."""
    surf = render(s, size, color, bold)
    rect = anchored(surf.get_rect(), pos, anchor)
    if alpha <= 0.003:
        return rect
    if alpha < 0.997:
        surf.set_alpha(int(255 * alpha))
        target.blit(surf, rect)
        surf.set_alpha(255)   # keep per-pixel alpha (None would drop it)
    else:
        target.blit(surf, rect)
    REC.add(rect, kind, name or s)
    return rect


def blit_lines(target, lines, size, color, pos, anchor="midtop", alpha=1.0,
               line_gap=1.3, kind="text", name=None):
    """Several centred lines stacked downward from pos."""
    rects = []
    x, y = pos
    for i, line in enumerate(lines):
        r = blit(target, line, size, color, (x, y + i * size * line_gap), anchor, alpha,
                 kind=kind, name=name or line)
        rects.append(r)
    return rects


def caption(target, s: str, alpha=1.0, color=None):
    """The one- or two-line caption at the bottom of every beat."""
    lines = s.split("\n")
    size = theme.CAPTION
    gap = 1.28
    total = size * gap * (len(lines) - 1)
    y0 = theme.CAPTION_CY - total / 2
    for i, line in enumerate(lines):
        blit(target, line, size, color or theme.TEXT, (theme.VW / 2, y0 + i * size * gap),
             "center", alpha, name="caption")
