"""Layout rules shared by the layout test and the screenshot harness."""
from __future__ import annotations

from core import theme


def check_items(items):
    """items: (Rect, kind, name) recorded during one draw.

    Every registered element must sit inside the canvas with MARGIN to spare, and
    no two layout text blocks may overlap. 'scene' labels (drawn on moving objects)
    are held to the margin rule only.
    """
    problems = []
    m = theme.MARGIN
    for r, kind, name in items:
        if r.w == 0 or r.h == 0:
            continue
        if r.left < m or r.top < m or r.right > theme.VW - m or r.bottom > theme.VH - m:
            problems.append(f"margin: {name!r} {tuple(r)}")
    texts = [(r, n) for r, k, n in items if k == "text" and r.w > 0 and r.h > 0]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, na = texts[i]
            b, nb = texts[j]
            # glyph boxes include some leading; require a real overlap
            if a.inflate(-6, -6).colliderect(b.inflate(-6, -6)):
                problems.append(f"overlap: {na!r} {tuple(a)} x {nb!r} {tuple(b)}")
    return problems
