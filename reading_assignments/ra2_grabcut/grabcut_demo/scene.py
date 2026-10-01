"""
Scene framework: the thing that makes this a live demo rather than a video.

A Scene owns one screen. It builds its artists once in `enter()` and then only
*mutates* them in `draw(beat, t)` - alpha, position, color - which is what keeps
60 fps possible with a few hundred edges on screen.

Each scene is divided into BEATS. The presenter drives beats with the arrow keys
or space; the runner animates `t` from 0 to 1 across a beat and then holds at 1
until the next keypress. Nothing is on a wall clock, so the demo can be paused
mid-sentence to answer a question and picked back up exactly where it was.
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from . import theme as T

FPS = 30


class Scene:
    """Base class. Subclasses override `title`, `n_beats`, `enter`, `draw`."""

    title = ""
    subtitle = ""
    n_beats = 1
    beat_seconds = 0.85     # transition length for a normal beat
    interactive = False     # True -> scene gets first refusal on every key

    def __init__(self):
        self.fig = None
        self.runner = None
        self.axes = []
        self.fig_artists = []

    # -- lifecycle ----------------------------------------------------------
    def enter(self, fig, runner):
        self.fig, self.runner = fig, runner
        self.axes = []
        self.fig_artists = []
        self.build()

    def build(self):
        """Create axes and artists. Called once."""

    def draw(self, beat, t):
        """Update artists for `beat` at progress `t` in [0, 1]."""

    def leave(self):
        for ax in self.axes:
            try:
                self.fig.delaxes(ax)
            except Exception:
                pass
        # figure-level artists are not owned by any axes, so they survive
        # delaxes and would bleed into the next scene
        for art in self.fig_artists:
            try:
                art.remove()
            except Exception:
                pass
        self.axes = []
        self.fig_artists = []

    # -- helpers ------------------------------------------------------------
    def add_axes(self, rect, **kw):
        ax = self.fig.add_axes(rect, **kw)
        self.axes.append(ax)
        return ax

    def fig_text(self, *a, **kw):
        """A figure-level text that is cleaned up when the scene exits."""
        t = self.fig.text(*a, **kw)
        self.fig_artists.append(t)
        return t

    def seconds(self, beat):
        return self.beat_seconds

    def export_beats(self):
        """Beats worth saving as report figures: {beat_index: 'filename'}."""
        return {}

    # -- optional event hooks; return True to swallow the event -------------
    def on_key(self, event):
        return False

    def on_press(self, event):
        return False

    def on_motion(self, event):
        return False

    def on_release(self, event):
        return False


class Runner:
    """Owns the figure, the scene list, the clock and the presenter HUD."""

    def __init__(self, scenes, out_dir, title="GrabCut"):
        self.scenes = scenes
        self.out_dir = out_dir
        self.i = 0
        self.beat = 0
        self.t = 1.0
        self.playing = False
        self._fade = None       # (direction, progress, pending_scene_index)
        self._saved = set()
        self._elapsed = 0.0

        T.apply_theme()
        self.fig = plt.figure(figsize=T.FIGSIZE, dpi=T.DPI)
        self.fig.canvas.manager.set_window_title(title)

        self._build_hud()
        self.fig.canvas.mpl_connect("key_press_event", self._key)
        self.fig.canvas.mpl_connect("button_press_event", self._press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._motion)
        self.fig.canvas.mpl_connect("button_release_event", self._release)

        self.scene = None
        self._enter(0)

        self.timer = self.fig.canvas.new_timer(interval=int(1000 / FPS))
        self.timer.add_callback(self._tick)

    # -- HUD ----------------------------------------------------------------
    def _build_hud(self):
        f = self.fig
        self.hud_title = f.text(0.042, 0.945, "", color=T.TEXT, fontsize=25,
                                va="center", ha="left", fontweight="medium")
        self.hud_sub = f.text(0.042, 0.898, "", color=T.MUTED, fontsize=13.5,
                              va="center", ha="left")
        self.hud_rule = f.add_artist(
            plt.Line2D([0.042, 0.958], [0.872, 0.872], color=T.PANEL_EDGE, lw=1.0))
        self.hud_step = f.text(0.958, 0.945, "", color=T.DIM, fontsize=11.5,
                               va="center", ha="right", family="monospace")
        self.hud_keys = f.text(0.958, 0.038, "", color=T.DIM, fontsize=10.5,
                               va="center", ha="right", family="DejaVu Sans")
        self.hud_note = f.text(0.042, 0.038, "", color=T.DIM, fontsize=10.5,
                               va="center", ha="left")
        # full-frame curtain used to cross-fade between scenes
        self.curtain = Rectangle((0, 0), 1, 1, transform=f.transFigure,
                                 facecolor=T.BG, edgecolor="none",
                                 zorder=1000, alpha=0.0, visible=False)
        f.add_artist(self.curtain)

    def set_note(self, text=""):
        self.hud_note.set_text(text)

    def _refresh_hud(self):
        s = self.scene
        self.hud_title.set_text(s.title)
        self.hud_sub.set_text(s.subtitle)
        self.hud_step.set_text(f"{self.i + 1:02d} / {len(self.scenes):02d}")
        if s.interactive:
            self.hud_keys.set_text("drag a box   ·   → step   ·   m morphology"
                                   "   ·   r reset   ·   q quit")
        else:
            self.hud_keys.set_text("→ / space  next   ·   ←  back"
                                   "   ·   e  save frame   ·   q  quit")
        self.hud_note.set_text("")

    # -- scene switching ----------------------------------------------------
    def _enter(self, index, at_end=False):
        """
        Enter a scene. `at_end` lands on its final beat, fully drawn - which is
        what stepping *backwards* into a scene should do; otherwise pressing
        left after overshooting throws away the whole previous screen.
        """
        if self.scene is not None:
            self.scene.leave()
        self.i = index
        self.scene = self.scenes[index]
        self.scene.enter(self.fig, self)
        self._refresh_hud()
        if at_end:
            self.beat = self.scene.n_beats - 1
            self.t, self._elapsed, self.playing = 1.0, 0.0, False
            for b in range(self.scene.n_beats):     # replay so accumulated
                self.scene.draw(b, 1.0)             # state is correct
        else:
            self.beat, self.t, self._elapsed = 0, 0.0, 0.0
            self.playing = True
            self.scene.draw(0, 0.0)

    def goto_scene(self, index, fade=True, at_end=False):
        index = max(0, min(index, len(self.scenes) - 1))
        if index == self.i:
            return
        if fade:
            self._fade = ["out", 0.0, index, at_end]
            self.curtain.set_visible(True)
        else:
            self._enter(index, at_end)
        self.fig.canvas.draw_idle()

    # -- beat stepping ------------------------------------------------------
    def next(self):
        s = self.scene
        if self.playing:                      # let an in-flight beat snap to end
            self.t, self.playing = 1.0, False
            s.draw(self.beat, 1.0)
            self.fig.canvas.draw_idle()
            return
        if self.beat + 1 < s.n_beats:
            self.beat += 1
            self.t, self._elapsed, self.playing = 0.0, 0.0, True
        else:
            self.goto_scene(self.i + 1)

    def prev(self):
        if self.beat > 0:
            self.beat -= 1
            self.t, self.playing = 1.0, False
            self.scene.draw(self.beat, 1.0)
            self.fig.canvas.draw_idle()
        elif self.i > 0:
            self._fade = ["out", 0.0, self.i - 1, True]
            self.curtain.set_visible(True)

    def replay(self):
        self.t, self._elapsed, self.playing = 0.0, 0.0, True

    # -- clock --------------------------------------------------------------
    def _tick(self):
        dt = 1.0 / FPS
        dirty = False

        if self._fade is not None:
            d, p, target, at_end = self._fade
            p += dt / 0.22
            if d == "out":
                if p >= 1.0:
                    self.curtain.set_alpha(1.0)
                    self._enter(target, at_end)
                    self._fade = ["in", 0.0, target, at_end]
                else:
                    self.curtain.set_alpha(T.smooth(p))
                    self._fade[1] = p
            else:
                if p >= 1.0:
                    self.curtain.set_alpha(0.0)
                    self.curtain.set_visible(False)
                    self._fade = None
                else:
                    self.curtain.set_alpha(1.0 - T.smooth(p))
                    self._fade[1] = p
            dirty = True

        if self.playing:
            self._elapsed += dt
            dur = max(self.scene.seconds(self.beat), 1e-6)
            self.t = min(self._elapsed / dur, 1.0)
            self.scene.draw(self.beat, self.t)
            if self.t >= 1.0:
                self.playing = False
                self._autosave()
            dirty = True

        if dirty:
            self.fig.canvas.draw_idle()

    def _autosave(self):
        wants = self.scene.export_beats()
        name = wants.get(self.beat)
        if name and name not in self._saved:
            self._saved.add(name)
            self.save_frame(name)

    def save_frame(self, name=None):
        os.makedirs(self.out_dir, exist_ok=True)
        if name is None:
            name = f"{type(self.scene).__name__.lower()}_b{self.beat}"
        path = os.path.join(self.out_dir, f"{name}.png")
        keys, note = self.hud_keys.get_text(), self.hud_note.get_text()
        self.hud_keys.set_text("")
        self.hud_note.set_text("")
        self.fig.savefig(path, facecolor=T.BG)
        self.hud_keys.set_text(keys)
        self.hud_note.set_text(note)
        print(f"  saved  {os.path.relpath(path)}")
        return path

    # -- events -------------------------------------------------------------
    def _key(self, event):
        k = event.key
        if k in ("q", "escape"):
            plt.close(self.fig)
            return
        if self.scene.on_key(event):
            self.fig.canvas.draw_idle()
            return
        if k in ("right", " ", "enter", "pagedown", "down"):
            self.next()
        elif k in ("left", "backspace", "pageup", "up"):
            self.prev()
        elif k == "e":
            self.save_frame()
            self.set_note("frame saved to out/")
        elif k == "R":
            self.replay()
        self.fig.canvas.draw_idle()

    def _press(self, e):
        if self.scene.on_press(e):
            self.fig.canvas.draw_idle()

    def _motion(self, e):
        if self.scene.on_motion(e):
            self.fig.canvas.draw_idle()

    def _release(self, e):
        if self.scene.on_release(e):
            self.fig.canvas.draw_idle()

    # -- run ----------------------------------------------------------------
    def run(self):
        self.timer.start()
        plt.show()

    def render_all(self, every_beat=False):
        """Headless pass: walk every scene/beat, save the ones marked for export."""
        for idx in range(len(self.scenes)):
            self._enter(idx)
            for b in range(self.scene.n_beats):
                self.beat = b
                self.scene.draw(b, 1.0)
                wants = self.scene.export_beats()
                if b in wants:
                    self.save_frame(wants[b])
                elif every_beat:
                    self.save_frame(f"{idx:02d}_{type(self.scene).__name__}_b{b}")
