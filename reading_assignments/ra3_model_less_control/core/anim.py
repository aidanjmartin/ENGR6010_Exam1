"""Easing and timing helpers. Beats are driven by a local clock t (seconds since the
beat started), and every animated quantity is a pure function of t. Replaying a beat
is just resetting t."""
from __future__ import annotations

import math


def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def lerp(a, b, t):
    return a + (b - a) * t


def ease_in_out(t):
    """Cubic ease-in-out on [0, 1]."""
    t = clamp(t)
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def ease_out(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3


def ease_in(t):
    t = clamp(t)
    return t * t * t


def ease_out_back(t, s=1.4):
    t = clamp(t) - 1
    return 1 + (s + 1) * t * t * t + s * t * t


def smoothstep(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def seg(t, start, dur, ease=ease_in_out):
    """Eased progress of an animation that starts at `start` and lasts `dur`."""
    if dur <= 0:
        return 1.0 if t >= start else 0.0
    return ease((t - start) / dur)


def fade(t, start, dur=0.45):
    return seg(t, start, dur, ease_out)


def approach(cur, target, rate, dt):
    """Frame-rate independent exponential approach."""
    return target + (cur - target) * math.exp(-rate * dt)


class Tween:
    """A value that eases from one number to another when set."""

    def __init__(self, value=0.0, dur=0.6, ease=ease_in_out):
        self.a = self.b = float(value)
        self.t = self.dur = dur
        self.ease = ease

    def set(self, value, dur=None, instant=False):
        self.a = self.value
        self.b = float(value)
        self.dur = self.dur if dur is None else dur
        self.t = self.dur if instant else 0.0

    def update(self, dt):
        self.t = min(self.t + dt, self.dur)

    @property
    def value(self):
        if self.dur <= 0:
            return self.b
        return lerp(self.a, self.b, self.ease(self.t / self.dur))

    @property
    def done(self):
        return self.t >= self.dur
