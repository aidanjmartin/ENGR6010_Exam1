"""Verification harness: render every beat headless and prove the layout is
identical at every window size.

For each beat it saves
  <out>/<act>_<beat>_virtual.png            the raw 1920 x 1080 canvas
  <out>/sizes/<act>_<beat>_<W>x<H>.png      the real present step into a W x H surface
then crops each presented frame to its content area, scales it back to
1920 x 1080 and checks the mean absolute pixel difference against the canvas
(see frame_diff). A control run with the content shifted by 6 px shows the
metric would catch a real misalignment.
It also checks the letterbox bars are plain background and runs the layout
checks (margins, text overlap) on every beat.
"""
from __future__ import annotations

import json
import os
import sys
import time

SIZES = [(1280, 720), (1600, 900), (1920, 1080), (2560, 1440), (1920, 1200), (1366, 768)]
MAX_MEAN_DIFF = 1.5     # 8-bit levels, after the box filter


def run_beat(app, ai, beat, dt=1 / 60, draw_every=1):
    """Enter a beat, run its screenshot script, and let animations finish.

    Frames are drawn as the live app draws them, so state that leaks from one
    frame to the next (surface alpha, caches) shows up in the screenshots too."""
    app.act_i = ai
    act = app.act
    act.enter(beat)
    app.snapshot = None
    app.fade_t = 1.0
    script = sorted(act.screenshot_script(), key=lambda e: e[0])
    t = 0.0
    i = 0
    n = 0
    while t < act.settle:
        while i < len(script) and script[i][0] <= t:
            script[i][1]()
            i += 1
        app.update(dt)
        if draw_every and n % draw_every == 0:
            app.draw()
        t += dt
        n += 1
    while i < len(script):
        script[i][1]()
        i += 1


def _box(a, f=4):
    w, h, _ = a.shape
    return a[:w // f * f, :h // f * f].reshape(w // f, f, h // f, f, 3).mean(axis=(1, 3))


def _back_to_virtual(presented, size):
    """Crop the content area of a presented frame and scale it back to 1920 x 1080."""
    import pygame
    from core.canvas import compute_transform
    from core.theme import VH, VW
    _, (ox, oy), (sw, sh) = compute_transform(*size)
    content = presented.subsurface(pygame.Rect(ox, oy, sw, sh)).copy()
    # Nearest-neighbour on the way back adds no blur or brightness bias of its own.
    return pygame.transform.scale(content, (VW, VH))


def frame_diff(canvas_arr, back):
    """Mean absolute difference after a 4 x 4 box filter (which suppresses the
    sub-pixel resampling blur) and removal of the global brightness offset that
    pygame's smoothscale introduces. Units: 8-bit levels."""
    import numpy as np
    import pygame
    a = _box(canvas_arr)
    b = _box(pygame.surfarray.array3d(back).astype(np.float64))
    d = b - a
    d -= d.mean(axis=(0, 1))
    return float(np.abs(d).mean())


def compare(canvas, presented, size):
    import numpy as np
    import pygame
    from core.canvas import compute_transform, present_onto
    from core.theme import BG
    W, H = size
    _, (ox, oy), (sw, sh) = compute_transform(W, H)
    a = pygame.surfarray.array3d(canvas).astype(np.float64)
    diff = frame_diff(a, _back_to_virtual(presented, size))
    # Control: the same pipeline with the content drawn 6 virtual px off. A layout
    # shift that small must score far above the threshold.
    shifted = pygame.Surface(canvas.get_size())
    shifted.fill(BG)
    shifted.blit(canvas, (6, 0))
    ctrl_target = pygame.Surface(size)
    present_onto(ctrl_target, shifted)
    control = frame_diff(a, _back_to_virtual(ctrl_target, size))
    full = pygame.surfarray.array3d(presented)
    bars_ok = True
    if ox > 0:
        bars_ok &= bool(np.all(full[:ox] == BG)) and bool(np.all(full[ox + sw:] == BG))
    if oy > 0:
        bars_ok &= bool(np.all(full[:, :oy] == BG)) and bool(np.all(full[:, oy + sh:] == BG))
    return diff, control, bars_ok


def run_harness(outdir: str) -> bool:
    from core.canvas import headless
    headless()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import pygame
    pygame.init()
    pygame.display.set_mode((64, 64))
    from core.canvas import present_onto
    from core.text import REC
    from main import App
    from tests.layout_check import check_items

    os.makedirs(os.path.join(outdir, "sizes"), exist_ok=True)
    t0 = time.time()
    app = App(headless=True)
    app.prerender()
    report, ok = [], True
    for ai, act in enumerate(app.acts):
        for b in range(act.beat_count):
            run_beat(app, ai, b)
            REC.start()
            app.draw()
            items = REC.stop()
            name = f"act{ai}_beat{b + 1}"
            pygame.image.save(app.canvas, os.path.join(outdir, f"{name}_virtual.png"))
            problems = check_items(items)
            row = {"beat": name, "layout_problems": problems, "sizes": {}}
            if problems:
                ok = False
            for size in SIZES:
                target = pygame.Surface(size)
                present_onto(target, app.canvas)
                pygame.image.save(target, os.path.join(outdir, "sizes", f"{name}_{size[0]}x{size[1]}.png"))
                diff, control, bars = compare(app.canvas, target, size)
                row["sizes"][f"{size[0]}x{size[1]}"] = {"mean_abs_diff": round(diff, 3),
                                                       "shifted_6px_control": round(control, 3),
                                                       "bars_clean": bars}
                if diff > MAX_MEAN_DIFF or not bars or control < 1.8 * MAX_MEAN_DIFF:
                    ok = False
            report.append(row)
            worst = max(v["mean_abs_diff"] for v in row["sizes"].values())
            ctrl = min(v["shifted_6px_control"] for v in row["sizes"].values())
            print(f"{name:14s} worst diff {worst:5.2f} (6 px shift would score >= {ctrl:5.2f})  "
                  f"layout {'OK' if not problems else problems}")
    with open(os.path.join(outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"{'PASS' if ok else 'FAIL'}  ({time.time() - t0:.1f}s, {len(report)} beats)  report: {outdir}/report.json")
    pygame.quit()
    return ok


if __name__ == "__main__":
    run_harness(sys.argv[1] if len(sys.argv) > 1 else "out")
