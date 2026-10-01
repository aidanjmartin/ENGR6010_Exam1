"""
GrabCut on images with a known foreground mask.

Readings: Rother, Kolmogorov and Blake, "GrabCut," ACM TOG, 2004 (paper
reference [5]); Peng, Zhang and Zhang, "A survey of graph theoretical
approaches to image segmentation," Pattern Recognition, 2013 (reference [6]).

GrabCut models foreground and background color each with a Gaussian mixture
model (GMM), then finds the labeling that minimizes a data term from those
GMMs plus a smoothness term, with one minimum graph cut. It alternates the two
steps: refit the GMMs to the current labels, then cut again. The stepping comes
unchanged from the RA2 demo (`GrabCutSession` in
`reading_assignments/ra2_grabcut/grabcut_demo/realimg.py`), which calls
cv2.grabCut one iteration at a time; here it runs without the interactive
layer.

Data. An irregular foreground blob on a background. Each pixel's color is a
class mean plus spatially correlated Gaussian noise with the same spread in
both classes. The separation d is the distance between the two class means in
units of that spread. Along the line joining the means, two such Gaussians
share an overlap coefficient of 2 * Phi(-d / 2): 1 when the color
distributions coincide, 0 when they are disjoint.

Experiment 1. From a bounding box around the blob, run 10 iterations and
record IoU against the true mask after each one, for two separations: d = 2.5
(overlap 0.21) and d = 1.5 (overlap 0.45).

Experiment 2. Sweep d from 0 to 5, five random seeds each, and plot final IoU
against the overlap coefficient to show where the color model stops
separating the classes.

Finding. On this data GrabCut settles within one or two iterations, and the
outcome is all or nothing. Either the first cut separates the blob almost
perfectly, or it labels every pixel background (IoU 0) and stays there. The
foreground GMM starts from every pixel in the box, background included, while
the background GMM starts from clean outside pixels. When the color
distributions overlap enough, the background GMM explains the whole box better,
and once no pixel is labeled foreground, refitting cannot bring any back.

Run:  python grabcut.py
Out:  results/grabcut_iterations.png, results/grabcut_overlap_sweep.png
"""

import math

import cv2
import numpy as np

import common as C

C.use_reading_assignment(C.RA2_DIR)

from grabcut_demo.realimg import GrabCutSession  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

H, W = 180, 240
SIGMA = 22.0                    # color noise per channel, on a 0-255 scale
DIRECTION = np.array([1.0, -1.0, 0.4])
DIRECTION /= np.linalg.norm(DIRECTION)
BOX_PAD = 12
N_ITERS = 10
CASES = {"success": 2.5, "collapse": 1.5}
SWEEP_SEPARATIONS = [0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0, 5.0]
SWEEP_SEEDS = 5
SWEEP_ITERS = 8


# --- data ------------------------------------------------------------------
def true_mask():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    m = np.zeros((H, W), bool)
    for cx, cy, ax, ay in ((120, 92, 52, 38), (152, 66, 32, 28), (92, 112, 30, 26)):
        m |= ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 <= 1.0
    return m


def overlap_coefficient(d):
    """Shared area of two unit-variance Gaussians whose means sit d apart."""
    return 2.0 * 0.5 * (1.0 + math.erf((-d / 2.0) / math.sqrt(2.0)))


def make_image(mask, separation, rng):
    center = np.array([128.0, 128.0, 128.0])
    offset = 0.5 * separation * SIGMA * DIRECTION
    noise = rng.standard_normal((H, W, 3)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 1.0)
    noise /= noise.std()
    img = np.where(mask[..., None], center + offset, center - offset)
    img = img + SIGMA * noise
    return np.clip(img, 0, 255).astype(np.uint8)


def bounding_box(mask, pad=BOX_PAD):
    ys, xs = np.nonzero(mask)
    x0, y0 = max(xs.min() - pad, 0), max(ys.min() - pad, 0)
    x1, y1 = min(xs.max() + pad, W - 1), min(ys.max() + pad, H - 1)
    return (int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1))


def box_mask(rect):
    x, y, w, h = rect
    m = np.zeros((H, W), bool)
    m[y:y + h, x:x + w] = True
    return m


def run_grabcut(img, rect, truth, n_iters):
    """Step RA2's GrabCutSession and record IoU and energy after each pass."""
    session = GrabCutSession(img, rect)
    history = []
    for _ in range(n_iters):
        rec = session.step()
        history.append({"it": rec["it"], "iou": C.iou(rec["alpha"], truth),
                        "E": rec["E"], "changed": rec["changed"],
                        "alpha": rec["alpha"]})
    return history


# --- figures ---------------------------------------------------------------
TP, FN, FP = "#c9c8c3", C.BLUE, C.ORANGE


def error_map(pred, truth):
    rgb = np.ones(truth.shape + (3,))
    for m, color in ((pred & truth, TP), (truth & ~pred, FN), (pred & ~truth, FP)):
        rgb[m] = plt.matplotlib.colors.to_rgb(color)
    return rgb


def framed(ax):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(C.AXIS)


def plot_iterations(cases, rect, truth, box_iou, name):
    """One row per case: input, mask after iteration 1, final mask, IoU curve."""
    fig = plt.figure(figsize=(15, 8.2))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.35], wspace=0.22,
                          hspace=0.3)
    colors = {"success": C.BLUE, "collapse": C.ORANGE}
    for row, (label, (d, img, history)) in enumerate(cases.items()):
        ov = overlap_coefficient(d)
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(img)
        x, y, w, h = rect
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), w, h, fill=False, ec="white",
                               lw=2, ls="--"))
        C.image_axes(ax, f"Input, overlap {ov:.2f} (d = {d:g})")
        for k, rec in enumerate((history[0], history[-1]), start=1):
            ax = fig.add_subplot(gs[row, k])
            ax.imshow(error_map(rec["alpha"], truth), interpolation="nearest")
            C.image_axes(ax, f"After iteration {rec['it']}   IoU {rec['iou']:.3f}")
            framed(ax)

    ax = fig.add_subplot(gs[:, 3])
    for label, (d, _, history) in cases.items():
        its = [r["it"] for r in history]
        ious = [r["iou"] for r in history]
        ax.plot(its, ious, "-o", color=colors[label], lw=2, ms=7,
                label=f"overlap {overlap_coefficient(d):.2f}")
        ax.annotate(f"{ious[-1]:.3f}", (its[-1], ious[-1]),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=10, color=C.INK)
    ax.axhline(box_iou, color=C.MUTED, lw=1.2, ls="--")
    ax.text(1, box_iou + 0.015, f"box alone, IoU {box_iou:.2f}", fontsize=10,
            color=C.MUTED)
    ax.set_xticks(its)
    ax.set_xlabel("iteration")
    ax.set_ylabel("IoU with true mask")
    ax.set_ylim(-0.04, 1.1)
    ax.set_title("IoU after each iteration")
    ax.legend(loc="center right")
    C.style_axes(ax)

    handles = [Patch(color=TP, label="foreground, found"),
               Patch(color=FN, label="foreground, missed"),
               Patch(color=FP, label="background, wrongly included")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10.5,
               bbox_to_anchor=(0.38, 0.0))
    fig.suptitle("GrabCut settles in the first iteration: it either separates "
                 "the blob or labels everything background")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.09)
    C.save_figure(fig, name)


def plot_sweep(overlaps, scores, box_iou, examples, name):
    fig = plt.figure(figsize=(12, 7.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.6], hspace=0.32, wspace=0.08)
    for k, (img, alpha, ov, iou) in enumerate(examples):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(img)
        if alpha.any():
            ax.contour(alpha, levels=[0.5], colors="white", linewidths=1.6)
        C.image_axes(ax, f"overlap {ov:.2f}   IoU {iou:.2f}")

    ax = fig.add_subplot(gs[1, :])
    mean = [float(np.mean(s)) for s in scores]
    ax.plot(overlaps, mean, "-", color=C.BLUE, lw=2, label="mean final IoU")
    for ov, sc in zip(overlaps, scores):
        ax.scatter([ov] * len(sc), sc, s=46, color=C.BLUE, alpha=0.45,
                   edgecolor="white", linewidth=1.0, zorder=3)
    ax.scatter([], [], s=46, color=C.BLUE, alpha=0.45,
               label=f"one run ({SWEEP_SEEDS} seeds per overlap)")
    ax.axhline(box_iou, color=C.MUTED, lw=1.2, ls="--")
    ax.text(0.01, box_iou + 0.015, f"box alone, IoU {box_iou:.2f}", fontsize=10,
            color=C.MUTED)
    ax.set_xlabel("overlap coefficient of the foreground and background "
                  "color distributions")
    ax.set_ylabel(f"IoU after {SWEEP_ITERS} iterations")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.08)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.08))
    C.style_axes(ax, grid_axis="both")

    ok = [ov for ov, sc in zip(overlaps, scores) if min(sc) >= 0.9]
    bad = [ov for ov, sc in zip(overlaps, scores) if max(sc) <= 0.1]
    if ok and bad:
        title = (f"GrabCut always succeeds up to color overlap {max(ok):.2f} "
                 f"and always collapses from {min(bad):.2f}")
    else:
        title = "Final GrabCut IoU against color overlap"
    fig.suptitle(title)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.09)
    C.save_figure(fig, name)


# --- main ------------------------------------------------------------------
def main():
    truth = true_mask()
    rect = bounding_box(truth)
    box_iou = C.iou(box_mask(rect), truth)
    print(f"Bounding box {rect} (x, y, w, h); the box alone scores IoU "
          f"{box_iou:.3f}. Energy is RA2's re-evaluation of the GrabCut energy "
          "from the current mask, since OpenCV does not report it. At iteration "
          "1, pixels changed counts the pixels labeled foreground.")

    cases = {}
    for label, d in CASES.items():
        rng = np.random.default_rng(C.SEED)
        img = make_image(truth, d, rng)
        history = run_grabcut(img, rect, truth, N_ITERS)
        cases[label] = (d, img, history)
        C.print_heading(f"IoU per iteration, {label} case (d = {d:g}, "
                        f"overlap {overlap_coefficient(d):.2f})")
        C.print_table(["iteration", "IoU", "pixels changed", "energy"],
                      [[r["it"], f"{r['iou']:.4f}", r["changed"], f"{r['E']:.0f}"]
                       for r in history])
    plot_iterations(cases, rect, truth, box_iou, "grabcut_iterations.png")

    rows, overlaps, scores, finals = [], [], [], {}
    for d in SWEEP_SEPARATIONS:
        sc = []
        for s in range(SWEEP_SEEDS):
            rng = np.random.default_rng(C.SEED + 1000 * s + int(100 * d))
            sweep_img = make_image(truth, d, rng)
            hist = run_grabcut(sweep_img, rect, truth, SWEEP_ITERS)
            sc.append(hist[-1]["iou"])
            if s == 0:
                finals[d] = (sweep_img, hist[-1]["alpha"])
        ov = overlap_coefficient(d)
        overlaps.append(ov)
        scores.append(sc)
        rows.append([f"{d:g}", f"{ov:.3f}", f"{np.mean(sc):.3f}",
                     f"{sum(v >= 0.9 for v in sc)} of {SWEEP_SEEDS}",
                     f"{sum(v <= 0.1 for v in sc)} of {SWEEP_SEEDS}"])

    C.print_heading(f"Final IoU after {SWEEP_ITERS} iterations vs color overlap")
    C.print_table(["separation d", "overlap", "mean IoU", "runs with IoU >= 0.9",
                   "runs with IoU <= 0.1"], rows)
    print("No run ends between IoU 0.1 and 0.9: each one either separates the "
          "blob or collapses to an empty mask."
          if all(v <= 0.1 or v >= 0.9 for sc in scores for v in sc)
          else "Some runs end between IoU 0.1 and 0.9.")

    examples = []
    for d in (4.0, 2.0, 1.0):
        img_d, alpha = finals[d]
        examples.append((img_d, alpha, overlap_coefficient(d), C.iou(alpha, truth)))
    plot_sweep(overlaps, scores, box_iou, examples, "grabcut_overlap_sweep.png")


if __name__ == "__main__":
    main()
