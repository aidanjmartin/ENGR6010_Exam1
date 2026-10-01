"""
Headless figure export.

The live tool needs a mouse, so `--export` drives it programmatically with a
default box and saves the frames that are worth reusing as report figures.
"""

import os

import numpy as np

from . import realimg as RI
from . import theme as T


def export_gifs(runner, out_dir, fps=12, dpi=60):
    """
    Animate the two screens whose *motion* is the point - the min-cut sweep and
    the alternating loop - so the report can show them moving. Rendered at 60
    dpi (960x540) to keep the files small enough to drop into a document.
    """
    from matplotlib.animation import PillowWriter

    from .scenes_a import CutScene, LoopScene

    for cls, name, per_beat in ((CutScene, "07b_mincut_sweep", 11),
                                (LoopScene, "08c_loop", 9)):
        scene = next((s for s in runner.scenes if isinstance(s, cls)), None)
        if scene is None:
            continue
        runner._enter(runner.scenes.index(scene))
        runner.hud_keys.set_text("")
        path = os.path.join(out_dir, f"{name}.gif")
        writer = PillowWriter(fps=fps)
        writer.setup(runner.fig, path, dpi=dpi)
        for b in range(scene.n_beats):
            for k in range(per_beat):
                scene.draw(b, (k + 1) / per_beat)
                writer.grab_frame(facecolor=T.BG)
            for _ in range(3):                      # a beat holds before the next
                writer.grab_frame(facecolor=T.BG)
        writer.finish()
        size = os.path.getsize(path) / 1e6
        print(f"  saved  {os.path.relpath(path)}  ({size:.1f} MB)")


def export_live_figures(runner, args, out_dir):
    """Run Part B to convergence with a default box and save its key frames."""
    from .part_b import LiveScene

    live = next((s for s in runner.scenes if isinstance(s, LiveScene)), None)
    if live is None:
        return
    idx = runner.scenes.index(live)
    runner._enter(idx)

    img = live.img
    live.rect = RI.default_rect(img)
    live.session = RI.GrabCutSession(img, live.rect)
    live._prev_veil = None

    live.state = "running"
    for _ in range(6):
        live.session.step()
        if live.session.converged():
            break
    live.state = "converged"
    live.draw(0, 1.0)
    runner.save_frame("10_live_converged")

    live.state = "final"
    live.morph = False
    live._prev_veil = None
    live.draw(0, 1.0)
    runner.save_frame("11_final_extraction")

    live.morph = True
    live._prev_veil = None
    live.draw(0, 1.0)
    runner.save_frame("12_final_extraction_morphology")

    _export_mask_compare(live, out_dir)


def _export_mask_compare(live, out_dir):
    """A side-by-side of the raw mask, the cleaned mask, and what moved."""
    import matplotlib.pyplot as plt

    raw = live.session.fg_mask
    clean = RI.clean_mask(raw)
    diff = raw ^ clean

    # crop to the subject: at full frame the speckle is a few pixels across and
    # invisible in a figure this size
    ys, xs = np.nonzero(raw | clean)
    pad = 18
    y0, y1 = max(0, ys.min() - pad), min(raw.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(raw.shape[1], xs.max() + pad)
    crop = lambda a: a[y0:y1, x0:x1]
    img = crop(live.img)
    ch, cw = img.shape[:2]

    fig = plt.figure(figsize=T.FIGSIZE, dpi=T.DPI)
    fig.text(0.042, 0.935, "Morphological cleanup of the final mask",
             fontsize=25, color=T.TEXT, va="center")
    fig.text(0.042, 0.885,
             "An opening then a closing with a 5-pixel elliptical element - not "
             "part of GrabCut, but it removes the speckle the cut leaves behind. "
             "Toggle it live with  m .",
             fontsize=13.5, color=T.MUTED, va="center")

    # size the panels so the crop fills the available height exactly
    top, bottom = 0.795, 0.075
    panel_h = top - bottom
    panel_w = panel_h * (cw / ch) * (T.FIGSIZE[1] / T.FIGSIZE[0])
    gap = 0.045
    total = 3 * panel_w + 2 * gap
    x_start = 0.5 - total / 2

    page = np.array([0.055, 0.066, 0.090])
    for i, (m, name) in enumerate(
            ((raw, "RAW MIN-CUT MASK"),
             (clean, "AFTER OPEN + CLOSE"),
             (diff, f"PIXELS CHANGED  ({int(diff.sum()):,})"))):
        ax = fig.add_axes([x_start + i * (panel_w + gap), bottom, panel_w, panel_h])
        T.blank(ax)
        mc = crop(m)
        if name.startswith("PIXELS"):
            shown = img / 255.0 * 0.22 + page * 0.55
            shown[mc] = np.array([1.0, 0.784, 0.341])
            ax.imshow(np.clip(shown, 0, 1))
        else:
            shown = img / 255.0
            shown = np.where(mc[..., None], shown, shown * 0.10 + page * 0.65)
            ax.imshow(np.clip(shown, 0, 1))
        ax.set_title(name, color=T.MUTED, fontsize=11.5, loc="left", pad=9)

    path = os.path.join(out_dir, "13_morphology_compare.png")
    fig.savefig(path, facecolor=T.BG)
    plt.close(fig)
    print(f"  saved  {os.path.relpath(path)}")
