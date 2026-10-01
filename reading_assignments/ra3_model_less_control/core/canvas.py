"""One 1920 x 1080 virtual canvas, presented letterboxed at any window size.

Everything in the demo is drawn to the virtual canvas in virtual pixels. The
window size is used in exactly one place: compute_transform(), which the present
step and the mouse mapping both call. That makes windowed and full-screen layouts
identical up to a uniform scale by construction.
"""
from __future__ import annotations

import os
import sys

import pygame

from core.theme import BG, VH, VW


def enable_dpi_awareness():
    """Stop Windows display scaling from stretching or offsetting the window."""
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass


def compute_transform(win_w: int, win_h: int):
    """Uniform scale and centred offset that fit the virtual canvas in the window."""
    scale = min(win_w / VW, win_h / VH)
    sw, sh = max(1, round(VW * scale)), max(1, round(VH * scale))
    ox, oy = (win_w - sw) // 2, (win_h - sh) // 2
    return scale, (ox, oy), (sw, sh)


def present_onto(target: pygame.Surface, canvas: pygame.Surface, cache: dict | None = None):
    """Scale the virtual canvas into target, centred, with background letterbox bars.

    Used for the real window and, unchanged, by the screenshot harness.
    """
    W, H = target.get_size()
    scale, (ox, oy), (sw, sh) = compute_transform(W, H)
    target.fill(BG)
    if (sw, sh) == (VW, VH):
        target.blit(canvas, (ox, oy))
        return
    buf = None
    if cache is not None:
        buf = cache.get((sw, sh))
        if buf is None:
            cache.clear()
            buf = cache[(sw, sh)] = pygame.Surface((sw, sh), 0, canvas)
    if buf is not None:
        pygame.transform.smoothscale(canvas, (sw, sh), buf)
    else:
        buf = pygame.transform.smoothscale(canvas, (sw, sh))
    target.blit(buf, (ox, oy))


class Display:
    """Owns the OS window. Never caches the display surface across mode changes."""

    def __init__(self, fullscreen: bool = False, windowed_size=(1600, 900)):
        self.windowed_size = windowed_size
        self.fullscreen = fullscreen
        self._scaled_cache: dict = {}
        self._apply_mode()

    def _apply_mode(self):
        if self.fullscreen:
            pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)
        self._scaled_cache.clear()

    def toggle_fullscreen(self):
        if not self.fullscreen:
            surf = pygame.display.get_surface()
            if surf is not None:
                self.windowed_size = surf.get_size()
        self.fullscreen = not self.fullscreen
        self._apply_mode()

    def on_resize(self, size=None):
        # SDL has already resized the window surface; only the transform changes.
        if size and not self.fullscreen:
            self.windowed_size = size
        self._scaled_cache.clear()

    def window_size(self):
        return pygame.display.get_surface().get_size()

    def present(self, canvas: pygame.Surface):
        present_onto(pygame.display.get_surface(), canvas, self._scaled_cache)
        pygame.display.flip()

    def to_virtual(self, pos):
        """Window pixel -> virtual pixel, or None inside the letterbox bars."""
        W, H = self.window_size()
        scale, (ox, oy), (sw, sh) = compute_transform(W, H)
        x, y = pos
        if not (ox <= x < ox + sw and oy <= y < oy + sh):
            return None
        return ((x - ox) / scale, (y - oy) / scale)


def make_canvas() -> pygame.Surface:
    return pygame.Surface((VW, VH))


def headless():
    """Configure SDL for offscreen rendering (tests and the screenshot harness)."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
