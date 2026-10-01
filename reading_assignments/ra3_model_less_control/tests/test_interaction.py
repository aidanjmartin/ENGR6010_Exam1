"""Drive every act with the presenter's keys and the mouse; nothing may raise,
and per-frame work must stay well inside a 60 fps budget."""
import time

import pygame
import pytest

from acts.base import Ev


def frames(app, n, dt=1 / 60):
    for _ in range(n):
        app.update(dt)
        app.draw()


def test_walk_every_beat_forward_and_back(app):
    app.goto(0)
    seen = set()
    for _ in range(40):
        seen.add((app.act_i, app.act.beat))
        app.handle_key(pygame.K_SPACE)
        frames(app, 3)
    assert len(seen) == 13
    for _ in range(40):
        app.handle_key(pygame.K_LEFT)
        frames(app, 3)
    assert (app.act_i, app.act.beat) == (0, 0)


def test_act_keys_and_mouse(app):
    keys = [pygame.K_g, pygame.K_t, pygame.K_b, pygame.K_j, pygame.K_n, pygame.K_r, pygame.K_h, pygame.K_i]
    for act_i in range(5):
        app.goto(act_i)
        for b in range(app.act.beat_count):
            app.act.enter(b)
            for k in keys:
                app.handle_key(k)
                frames(app, 2)
            for pos in [(500, 400), (1400, 400), (300, 910), (900, 846), None]:
                app.act.handle(Ev("down", pos, button=1))
                app.act.handle(Ev("motion", (pos[0] + 40, pos[1]) if pos else None))
                app.act.handle(Ev("up", pos, button=1))
                frames(app, 2)
    app.show_help = False
    app.show_indicator = True


def test_act3_click_sets_target_in_both_panels(app):
    app.goto(3, 2)
    frames(app, 90)
    act = app.act
    before = act.sim.target_local.copy()
    act.handle(Ev("down", (1400, 330), button=1))
    assert (act.sim.target_local != before).any()


def test_act1_slider_reruns(app):
    app.goto(1, 1)
    frames(app, 10)
    act = app.act
    s = act.slider
    x = s.x + 0.3 * s.w
    act.handle(Ev("down", (x, s.y), button=1))
    act.handle(Ev("motion", (x + 50, s.y)))
    act.handle(Ev("up", (x + 50, s.y), button=1))
    assert abs(act.gamma - s.value) < 1e-9
    assert 1.0 <= act.gamma <= 50.0


def test_frame_budget(app):
    from core.canvas import present_onto
    target = pygame.Surface((1600, 900))
    worst = {}
    for act_i, beat in [(1, 1), (2, 1), (3, 3), (3, 4)]:
        app.goto(act_i, beat)
        app.fade_t = 1.0
        frames(app, 120)
        t0 = time.perf_counter()
        n = 60
        for _ in range(n):
            app.update(1 / 60)
            app.draw()
            present_onto(target, app.canvas)
        worst[(act_i, beat)] = (time.perf_counter() - t0) / n
    assert max(worst.values()) < 1 / 60, worst


def test_faded_text_keeps_transparency(app):
    # Regression: resetting surface alpha with set_alpha(None) dropped per-pixel
    # alpha, so text drawn after a fade turned into solid rectangles.
    from core import text, theme
    from core.mathtext import Eq
    eq = Eq([r"x \leftarrow x"]).build()
    for draw in (lambda s, a: text.blit(s, "Hello", 40, theme.TEXT, (10, 10), alpha=a),
                 lambda s, a: eq.draw(s, (10, 10), "topleft", alpha=a, register=False)):
        for a in (0.5, 1.0, 1.0):
            s = pygame.Surface((400, 120))
            s.fill(theme.BG)
            draw(s, a)
            assert tuple(s.get_at((11, 11)))[:3] == theme.BG
