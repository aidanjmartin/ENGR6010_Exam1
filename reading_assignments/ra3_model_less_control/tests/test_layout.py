"""Every beat, rendered headless: registered text and UI stay 40 px inside the
canvas, layout text blocks never overlap, and the present step is exact."""
import pygame
import pytest

from core import theme
from core.canvas import compute_transform
from core.text import REC
from tests.layout_check import check_items
from tools.screenshot_all import run_beat


def all_beats():
    counts = [1, 3, 3, 5, 1]
    return [(a, b) for a, n in enumerate(counts) for b in range(n)]


def test_beat_counts(app):
    assert [a.beat_count for a in app.acts] == [1, 3, 3, 5, 1]


@pytest.mark.parametrize("act_i,beat", all_beats())
def test_layout(app, act_i, beat):
    run_beat(app, act_i, beat, draw_every=5)
    REC.start()
    app.draw()
    items = REC.stop()
    assert any(k == "text" for _, k, _ in items)
    assert check_items(items) == []


@pytest.mark.parametrize("size", [(1280, 720), (1600, 900), (1920, 1080), (2560, 1440),
                                  (1920, 1200), (1366, 768), (1000, 1000)])
def test_transform_is_uniform_and_centred(size):
    W, H = size
    scale, (ox, oy), (sw, sh) = compute_transform(W, H)
    assert abs(sw / theme.VW - sh / theme.VH) < 2 / theme.VH      # uniform scale
    assert sw <= W and sh <= H and (sw == W or sh == H)            # fills one axis
    assert abs(ox - (W - sw - ox)) <= 1 and abs(oy - (H - sh - oy)) <= 1


def test_help_overlay_fits(app):
    app.show_help = True
    app.help_t = 1.0
    REC.start()
    app.draw()
    items = [it for it in REC.stop() if it[2] == "help"]
    app.show_help = False
    app.help_t = 0.0
    for r, _, _ in items:
        assert r.left >= theme.MARGIN and r.bottom <= theme.VH - theme.MARGIN
