"""When the Model Is Wrong: an interactive demo for ENGR 6010.

    python main.py                  windowed, from the title
    python main.py --fullscreen     full screen at desktop resolution
    python main.py --act 3          jump to an act
    python main.py --screenshots out/   render every beat headless and verify layout
"""
from __future__ import annotations

import argparse
import sys
import time

from core import canvas as cv


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--fullscreen", action="store_true", help="start full screen")
    g.add_argument("--windowed", action="store_true", help="start in a 1600 x 900 window (default)")
    p.add_argument("--act", type=int, default=0, help="start at act N (0-4)")
    p.add_argument("--screenshots", metavar="OUTDIR", help="render every beat headless, then exit")
    return p.parse_args(argv)


# Keys that navigate or toggle globally; everything else goes to the act.
HELP_ROWS = [
    ("Space  /  →", "next beat or build"),
    ("←", "previous beat"),
    ("1 – 4", "jump to act"),
    ("R", "replay current beat"),
    ("F", "toggle full screen"),
    ("H", "this help"),
    ("I", "hide / show beat indicator"),
    ("G", "Act 1: stretch the bowl (γ 1 → 25)"),
    ("click", "Act 3: set a target in both panels"),
    ("T", "Act 3: square trajectory on / off"),
    ("B", "Act 3: bend / straighten the endoscope"),
    ("J", "Act 3: dashed true-Jacobian arrows"),
    ("N", "Act 3: sensor noise on / off"),
    ("Esc", "quit"),
]


class App:
    FADE = 0.45

    def __init__(self, headless=False):
        import pygame
        self.pg = pygame
        from acts.act0_title import TitleAct
        from acts.act1_curvature import CurvatureAct
        from acts.act2_secant import SecantAct
        from acts.act3_robot import RobotAct
        from acts.act4_close import CloseAct
        self.acts = [TitleAct(), CurvatureAct(), SecantAct(), RobotAct(), CloseAct()]
        self.canvas = cv.make_canvas()
        self.act_i = 0
        self.show_help = False
        self.show_indicator = True
        self.help_t = 0.0
        self.snapshot = None
        self.fade_t = 1.0
        self.slide = 0
        self.headless = headless

    @property
    def act(self):
        return self.acts[self.act_i]

    # ---- loading ------------------------------------------------------------------------
    def prerender(self, progress=None):
        eqs = [e for a in self.acts for e in a.equations()]
        for i, e in enumerate(eqs):
            e.build()
            if progress:
                progress((i + 1) / max(1, len(eqs)))

    # ---- navigation ---------------------------------------------------------------------
    def _begin_transition(self, slide=0):
        self.snapshot = self.canvas.copy()
        self.fade_t = 0.0
        self.slide = slide

    def goto(self, act_i, beat=0, full=False):
        slide = 0 if act_i == self.act_i else (1 if act_i > self.act_i else -1)
        self._begin_transition(slide)
        self.act_i = act_i
        self.act.enter(beat, full=full)

    def next(self):
        r = self.act.next_beat()
        if r == "beat":
            self._begin_transition()
        elif r is None and self.act_i < len(self.acts) - 1:
            self.goto(self.act_i + 1, 0)

    def prev(self):
        r = self.act.prev_beat()
        if r == "beat":
            self._begin_transition()
        elif r is None and self.act_i > 0:
            a = self.acts[self.act_i - 1]
            self.goto(self.act_i - 1, a.beat_count - 1, full=True)

    # ---- per frame ------------------------------------------------------------------------
    def update(self, dt):
        self.act.update(dt)
        self.fade_t = min(1.0, self.fade_t + dt / self.FADE)
        self.help_t = min(1.0, self.help_t + dt / 0.25) if self.show_help else max(0.0, self.help_t - dt / 0.25)

    def draw(self):
        from core import theme
        from core.anim import ease_in_out
        c = self.canvas
        c.fill(theme.BG)
        self.act.draw(c)
        if self.snapshot is not None and self.fade_t < 1.0:
            p = ease_in_out(self.fade_t)
            if self.slide:
                fresh = c.copy()
                c.fill(theme.BG)
                off = int(70 * (1 - p)) * self.slide
                fresh.set_alpha(int(255 * p))
                c.blit(fresh, (off, 0))
                self.snapshot.set_alpha(int(255 * (1 - p)))
                c.blit(self.snapshot, (-int(70 * p) * self.slide, 0))
            else:
                self.snapshot.set_alpha(int(255 * (1 - p)))
                c.blit(self.snapshot, (0, 0))
            self.snapshot.set_alpha(None)
        elif self.fade_t >= 1.0:
            self.snapshot = None
        if self.show_indicator:
            self.draw_indicator(c)
        if self.help_t > 0:
            self.draw_help(c, self.help_t)

    def draw_indicator(self, c):
        from core import text, theme
        a = self.act
        s = f"Act {a.number} · {a.beat + 1}/{a.beat_count}"
        text.blit(c, s, theme.SMALL, theme.TEXT_FAINT, (theme.VW - theme.MARGIN - 8, theme.MARGIN + 8),
                  "topright", name="indicator", kind="ui")

    def draw_help(self, c, a):
        from core import draw, text, theme
        pg = self.pg
        veil = pg.Surface((theme.VW, theme.VH), pg.SRCALPHA)
        veil.fill((*theme.BG, int(200 * a)))
        c.blit(veil, (0, 0))
        w, h = 1000, 110 + 50 * len(HELP_ROWS)
        r = pg.Rect(0, 0, w, h)
        r.center = (theme.VW // 2, theme.VH // 2)
        draw.panel(c, r, theme.dim(theme.BG_PANEL, a), theme.dim(theme.PANEL_EDGE, a))
        text.blit(c, "Controls", theme.HEADING, theme.dim(theme.TEXT, a), (r.x + 60, r.y + 36), name="help")
        for i, (k, d) in enumerate(HELP_ROWS):
            y = r.y + 120 + i * 50
            text.blit(c, k, theme.LABEL, theme.dim(theme.ESTIMATE, a), (r.x + 60, y), name="help")
            text.blit(c, d, theme.LABEL, theme.dim(theme.TEXT_DIM, a), (r.x + 330, y), name="help")

    # ---- input ---------------------------------------------------------------------------------
    def handle_key(self, key):
        pg = self.pg
        from acts.base import Ev
        if key == pg.K_ESCAPE:
            if self.show_help:
                self.show_help = False
                return True
            return False
        if key in (pg.K_SPACE, pg.K_RIGHT, pg.K_PAGEDOWN):
            self.next()
        elif key in (pg.K_LEFT, pg.K_PAGEUP):
            self.prev()
        elif key in (pg.K_1, pg.K_2, pg.K_3, pg.K_4):
            self.goto(key - pg.K_0, 0)
        elif key == pg.K_0:
            self.goto(0, 0)
        elif key == pg.K_r:
            self._begin_transition()
            self.act.reset_beat()
        elif key == pg.K_h:
            self.show_help = not self.show_help
        elif key == pg.K_i:
            self.show_indicator = not self.show_indicator
        else:
            self.act.handle(Ev("key", key=key))
        return True


def run(args):
    cv.enable_dpi_awareness()
    import pygame
    pygame.init()
    pygame.display.set_caption("When the Model Is Wrong")
    display = cv.Display(fullscreen=args.fullscreen)
    pygame.key.set_repeat(0)

    from core import text, theme
    from acts.base import Ev

    # Loading screen while the maths is pre-rendered.
    loading = cv.make_canvas()

    def progress(f):
        loading.fill(theme.BG)
        text.blit(loading, "When the Model Is Wrong", theme.HEADING, theme.TEXT_DIM,
                  (theme.VW / 2, theme.VH / 2 - 50), "center")
        bar = pygame.Rect(0, 0, 520, 8)
        bar.center = (theme.VW // 2, theme.VH // 2 + 30)
        pygame.draw.rect(loading, theme.PANEL_EDGE, bar, border_radius=4)
        pygame.draw.rect(loading, theme.ESTIMATE, (bar.x, bar.y, int(bar.w * f), bar.h), border_radius=4)
        display.present(loading)
        pygame.event.pump()

    progress(0.0)
    app = App()
    app.prerender(progress)
    app.goto(max(0, min(4, args.act)), 0)
    app.snapshot = None

    clock = pygame.time.Clock()
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_f:
                    display.toggle_fullscreen()
                elif not app.handle_key(e.key):
                    running = False
            elif e.type in (pygame.VIDEORESIZE, pygame.WINDOWSIZECHANGED):
                display.on_resize(getattr(e, "size", None) if e.type == pygame.VIDEORESIZE else None)
            elif e.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
                pos = display.to_virtual(e.pos)
                kind = {pygame.MOUSEBUTTONDOWN: "down", pygame.MOUSEBUTTONUP: "up",
                        pygame.MOUSEMOTION: "motion"}[e.type]
                if kind == "down" and pos is None:
                    continue   # clicks in the letterbox bars are ignored
                if kind == "down" and getattr(e, "button", 1) not in (1, 3):
                    continue
                app.act.handle(Ev(kind, pos, button=getattr(e, "button", 0)))
        app.update(dt)
        app.draw()
        display.present(app.canvas)
    pygame.quit()


def main(argv=None):
    args = parse_args(argv)
    if args.screenshots:
        from tools.screenshot_all import run_harness
        ok = run_harness(args.screenshots)
        sys.exit(0 if ok else 1)
    run(args)


if __name__ == "__main__":
    main()
