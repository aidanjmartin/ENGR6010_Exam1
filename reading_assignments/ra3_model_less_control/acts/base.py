"""Common interface for every act.

A beat is one idea. Inside a beat, an act may have presenter-driven builds (extra
Space presses that reveal the next row) and automatic animations driven by the
beat clock self.t. Nothing advances beats on a timer.
"""
from __future__ import annotations

from dataclasses import dataclass

from core import text, theme


@dataclass
class Ev:
    kind: str              # down | up | motion | key
    pos: tuple = None      # virtual pixels, or None when outside the canvas
    key: int = 0
    button: int = 0


class Act:
    number = 0
    title = ""
    captions: list = []    # one caption per beat
    settle = 6.0           # seconds of sim time the screenshot harness waits per beat

    def __init__(self):
        self.beat = 0
        self.t = 0.0
        self.build = 0
        self.build_t = 0.0

    # ---- structure ---------------------------------------------------------------
    @property
    def beat_count(self):
        return len(self.captions)

    def builds_in(self, beat):
        return 0

    def equations(self):
        """Every Eq this act uses, so the loader can pre-render them."""
        return []

    # ---- navigation ------------------------------------------------------------------
    def enter(self, beat=0, full=False):
        self.beat = max(0, min(beat, self.beat_count - 1))
        self._start_beat()
        if full:
            self.build = self.builds_in(self.beat)

    def _start_beat(self):
        self.t = 0.0
        self.build = 0
        self.build_t = 0.0
        self.on_beat()

    def next_beat(self):
        """'build' if a build was revealed, 'beat' if the beat advanced, None at the end."""
        if self.build < self.builds_in(self.beat):
            self.build += 1
            self.build_t = 0.0
            self.on_build()
            return "build"
        if self.beat < self.beat_count - 1:
            self.beat += 1
            self._start_beat()
            return "beat"
        return None

    def prev_beat(self):
        if self.beat > 0:
            self.beat -= 1
            self._start_beat()
            self.build = self.builds_in(self.beat)
            return "beat"
        return None

    def reset_beat(self):
        self._start_beat()

    # ---- hooks -----------------------------------------------------------------------
    def on_beat(self):
        pass

    def on_build(self):
        pass

    def update(self, dt):
        self.t += dt
        self.build_t += dt

    def draw(self, canvas):
        pass

    def handle(self, ev: Ev) -> bool:
        return False

    def reveal_all(self):
        self.build = self.builds_in(self.beat)

    def screenshot_script(self):
        """Scripted presenter actions for the screenshot harness: (time, action)."""
        return [(0.0, self.reveal_all)]

    # ---- shared drawing ----------------------------------------------------------------
    def draw_header(self, canvas, alpha=1.0):
        text.blit(canvas, self.title, theme.TITLE - 4, theme.TEXT, (theme.GUTTER, theme.HEADER_Y),
                  "topleft", alpha, name="title")

    def draw_caption(self, canvas, start=0.25):
        from core.anim import fade
        cap = self.captions[self.beat]
        if cap:
            text.caption(canvas, cap, fade(self.t, start, 0.6))
