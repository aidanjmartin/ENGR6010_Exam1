"""TeX strings -> cached pygame surfaces with individually coloured terms.

matplotlib.mathtext has no \\color, so each equation is written as a list of
segments, some tagged with a key:

    Eq([r"x \\leftarrow x -", (r"\\alpha\\,\\nabla f", "grad")], colors={"grad": GRADIENT})

The whole string is rendered once, and once more per key with that key's
segments replaced by \\phantom{...}. Phantoms keep the layout identical, so the
difference of the two renders is exactly that key's ink. Invisible sentinel
delimiters at both ends pin every render to the same bounding box.
"""
from __future__ import annotations

import math

import numpy as np
import pygame

from core import theme
from core.text import REC

_parser = None
_SENT = r"\phantom{\frac{\frac{\frac{X}{X}}{X}}{\frac{X}{\frac{X}{X}}}}"
_L = r"\left|" + _SENT + r"\right.\;\;"
_R = r"\;\;\left." + _SENT + r"\right|"


def _setup():
    global _parser
    if _parser is None:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["mathtext.fontset"] = "cm"
        from matplotlib.mathtext import MathTextParser
        _parser = MathTextParser("agg")
    return _parser


def _raster(tex: str, size: float):
    from matplotlib.font_manager import FontProperties
    p = _setup()
    r = p.parse("$" + _L + tex + _R + "$", dpi=72, prop=FontProperties(size=size))
    img = np.asarray(r.image, dtype=np.float32) / 255.0
    baseline = int(round(r.height - r.depth))
    return img, baseline


def _colorize(alpha: np.ndarray, color) -> pygame.Surface:
    h, w = alpha.shape
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    surf.fill((*color, 0))
    a = pygame.surfarray.pixels_alpha(surf)
    a[:] = np.clip(alpha.T * 255.0, 0, 255).astype(np.uint8)
    del a
    return surf


class Eq:
    def __init__(self, segments, size=theme.MATH, color=theme.TEXT, colors=None):
        self.segments = [(s, None) if isinstance(s, str) else (s[0], s[1]) for s in segments]
        self.size = size
        self.color = color
        self.colors = dict(colors or {})
        self.keys = []
        for _, k in self.segments:
            if k is not None and k not in self.keys:
                self.keys.append(k)
        self._built = False

    def _tex(self, hide=()):
        out = []
        for s, k in self.segments:
            if k is not None and (k in hide or hide == "all"):
                out.append(r"\phantom{" + s + "}")
            else:
                out.append("{" + s + "}" if k is not None else s)
        return " ".join(out)

    def build(self):
        if self._built:
            return self
        full, baseline = _raster(self._tex(), self.size)
        base, _ = _raster(self._tex(hide="all"), self.size) if self.keys else (full, baseline)
        layers = {}
        for k in self.keys:
            without, _ = _raster(self._tex(hide=(k,)), self.size)
            layers[k] = np.clip(full - _match(without, full.shape), 0.0, 1.0)
        base = _match(base, full.shape)
        # Crop the sentinels: find the ink columns belonging to them at each edge.
        cols = full.max(axis=0) > 0.02
        x0 = _skip_run(cols, 0, 1)
        x1 = _skip_run(cols, len(cols) - 1, -1) + 1
        content = full[:, x0:x1]
        rows = np.where(content.max(axis=1) > 0.02)[0]
        pad = 6
        y0 = max(0, rows[0] - pad) if len(rows) else 0
        y1 = min(full.shape[0], rows[-1] + 1 + pad) if len(rows) else full.shape[0]
        cols_c = np.where(content.max(axis=0) > 0.02)[0]
        cx0 = x0 + (cols_c[0] if len(cols_c) else 0) - pad
        cx1 = x0 + (cols_c[-1] + 1 if len(cols_c) else content.shape[1]) + pad
        cx0, cx1 = max(0, cx0), min(full.shape[1], cx1)
        crop = (slice(y0, y1), slice(cx0, cx1))
        self.base = _colorize(base[crop], self.color)
        self.layers, self.boxes = {}, {}
        for k, a in layers.items():
            a = a[crop]
            self.layers[k] = _colorize(a, self.colors.get(k, self.color))
            ys, xs = np.where(a > 0.05)
            if len(xs):
                self.boxes[k] = pygame.Rect(int(xs.min()), int(ys.min()),
                                            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        self.w, self.h = cx1 - cx0, y1 - y0
        self.baseline = baseline - y0
        self._built = True
        return self

    # ---- geometry -------------------------------------------------------------------
    def rect(self, pos, anchor="midleft"):
        self.build()
        r = pygame.Rect(0, 0, self.w, self.h)
        if anchor == "baseline_left":
            r.topleft = (round(pos[0]), round(pos[1] - self.baseline))
        elif anchor == "baseline_center":
            r.topleft = (round(pos[0] - self.w / 2), round(pos[1] - self.baseline))
        else:
            setattr(r, anchor, (round(pos[0]), round(pos[1])))
        return r

    def _box(self, key):
        """Ink box of one key, or the union of several keys given as a tuple."""
        if isinstance(key, tuple):
            boxes = [self.boxes[k] for k in key if k in self.boxes]
            return boxes[0].unionall(boxes[1:]) if boxes else None
        return self.boxes.get(key)

    def key_rect(self, key, pos, anchor="midleft"):
        r = self.rect(pos, anchor)
        b = self._box(key)
        return b.move(r.topleft) if b else None

    # ---- drawing --------------------------------------------------------------------
    def draw(self, target, pos, anchor="midleft", alpha=1.0, key_alpha=None,
             highlight=None, colors=None, register=True, base_alpha=None):
        """key_alpha: per-key opacity; highlight: {key: strength 0..1} glow boxes."""
        self.build()
        r = self.rect(pos, anchor)
        if alpha <= 0.003:
            return r
        if highlight:
            for k, s in highlight.items():
                box = self._box(k)
                if s > 0.01 and box is not None:
                    k0 = k[0] if isinstance(k, tuple) else k
                    col = (colors or {}).get(k0) or self.colors.get(k0, self.color)
                    _glow_box(target, box.move(r.topleft), col, s * alpha)
        ba = alpha if base_alpha is None else base_alpha * alpha
        _blit_alpha(target, self.base, r.topleft, ba)
        for k, surf in self.layers.items():
            a = alpha * (key_alpha.get(k, 1.0) if key_alpha else 1.0)
            if colors and k in colors:
                surf = _recolor(surf, colors[k])
            _blit_alpha(target, surf, r.topleft, a)
        if register:
            REC.add(r, "text", "eq")
        return r


def _match(a, shape):
    if a.shape == shape:
        return a
    out = np.zeros(shape, dtype=a.dtype)
    h, w = min(shape[0], a.shape[0]), min(shape[1], a.shape[1])
    out[:h, :w] = a[:h, :w]
    return out


def _skip_run(cols, start, step):
    """From an edge, skip the sentinel's ink run and the gap after it."""
    i = start
    n = len(cols)
    while 0 <= i < n and not cols[i]:
        i += step
    while 0 <= i < n and cols[i]:
        i += step
    return min(max(i, 0), n - 1)


_recolor_cache = {}


def _recolor(surf, color):
    key = (id(surf), tuple(color))
    s = _recolor_cache.get(key)
    if s is None:
        s = surf.copy()
        s.fill((*color, 0), special_flags=pygame.BLEND_RGBA_MAX)
        s.fill((*color, 255), special_flags=pygame.BLEND_RGBA_MIN)
        _recolor_cache[key] = s
    return s


def _blit_alpha(target, surf, pos, a):
    if a <= 0.003:
        return
    if a < 0.997:
        surf.set_alpha(int(255 * a))
        target.blit(surf, pos)
        surf.set_alpha(255)   # keep per-pixel alpha (None would drop it)
    else:
        target.blit(surf, pos)


def _glow_box(target, rect, color, strength):
    r = rect.inflate(18, 12)
    s = pygame.Surface(r.size, pygame.SRCALPHA)
    pygame.draw.rect(s, (*color, int(46 * strength)), s.get_rect(), border_radius=10)
    pygame.draw.rect(s, (*color, int(170 * strength)), s.get_rect(), width=2, border_radius=10)
    target.blit(s, r)


def pulse(t, period=1.6):
    return 0.65 + 0.35 * math.sin(2 * math.pi * t / period)
