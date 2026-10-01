"""
Demo 3 - block-matching optical flow.

A numpy port of the search loop in the University of Southampton optical-flow
teaching applet (`opticalFlow.java`): for every sample point, try every integer
displacement in a square search window, score each by the sum of squared
differences over a square patch, and keep the best.

The applet exposes two sliders - patch size and search range - and this program
exists to put numbers on what those sliders cost. The search is written as the
explicit nested loop the applet uses rather than vectorized, because the cost is
the thing being measured:

    operations per sample point = (2*max_disp + 1)^2  *  (2*window + 1)^2
                                   displacements tried    pixels per SSD

Both factors are quadratic, so both sliders are quadratic. The program closes by
showing the method's other hard limit: the search enumerates integers, so a
displacement of (2.5, 1.5) is not in the hypothesis space at all. That is what
Demo 4 fixes.

Run:  python demo3_block_matching_flow.py   (~1 minute; the slowness is the point)
Out:  out/block_matching_flow.png, out/block_matching_sweep.png
"""

import os
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# --- configuration ---------------------------------------------------------
SIZE = (240, 320)             # (rows, cols)
SEED = 20260908
WINDOW = 7                    # patch half-width -> (2*7+1)^2 = 225 px per SSD
MAX_DISP = 5                  # search half-range -> (2*5+1)^2 = 121 displacements
STEP = 8                      # sample-grid spacing
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Categorical palette (validated light-mode).
C_ERR = "#2a78d6"
C_TIME = "#eb6834"
C_INK = "#0b0b0b"
C_MUTED = "#52514e"
C_SURFACE = "#fcfcfb"


# --- 1. synthetic frame pair (imported by Demo 4) --------------------------
def make_frame_pair(size=SIZE, shift=(3, 2), seed=SEED):
    """A textured frame and a rigidly translated copy of it.

    Frame 1 is white noise passed through a Gaussian blur, which gives it
    gradients everywhere at a usable spatial scale - block matching needs
    texture to lock onto, and Demo 4's gradient method needs Ix, Iy to be
    meaningful. Frame 2 is frame 1 displaced by `shift` = (dx, dy), so the true
    flow is (dx, dy) at every pixel.

    An integer shift uses np.roll (exact, no resampling). A non-integer shift
    goes through cv2.warpAffine with bilinear interpolation.

    Returns (f1, f2, true_shift), frames float32 grayscale in [0, 1].
    """
    h, w = size
    dx, dy = float(shift[0]), float(shift[1])

    rng = np.random.default_rng(seed)
    f1 = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 2.0)
    f1 = (f1 - f1.min()) / (f1.max() - f1.min())     # normalize to [0, 1]

    if float(dx).is_integer() and float(dy).is_integer():
        # np.roll(f, (dy, dx), (0, 1)) moves content down by dy and right by dx.
        f2 = np.roll(f1, (int(dy), int(dx)), axis=(0, 1))
    else:
        M = np.float32([[1, 0, dx], [0, 1, dy]])     # dst(x) = src(x - d)
        f2 = cv2.warpAffine(f1, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)

    return f1.astype(np.float32), f2.astype(np.float32), (dx, dy)


# --- 2. the applet's search ------------------------------------------------
def block_match_flow(f1, f2, window=WINDOW, max_disp=MAX_DISP, step=STEP,
                     explicit=True):
    """Exhaustive integer-displacement block matching.

    For each sample point p on a grid of spacing `step`, score every integer
    offset (dx, dy) in [-max_disp, max_disp]^2 by

        SSD(dx, dy) = sum over the patch of ( f1(p + q) - f2(p + q + d) )^2

    and keep the lowest-scoring offset. Ties go to the first offset found, which
    scans in raster order from (-max_disp, -max_disp).

    `explicit=True` runs the fully unrolled loop, including the per-pixel
    accumulation - this is the applet's structure and the configuration whose
    runtime is reported. `explicit=False` computes each patch SSD as one numpy
    array operation; it visits exactly the same displacements and returns the
    same answer, and is used only where an unrolled run would take minutes (the
    parameter sweeps).

    Returns (points, flow, n_ssd): points as (N, 2) int (x, y), flow as (N, 2)
    float (u, v), and the number of SSD evaluations performed.
    """
    h, w = f1.shape
    margin = window + max_disp                   # keep every patch access in bounds
    ys = list(range(margin, h - margin, step))
    xs = list(range(margin, w - margin, step))

    points = np.empty((len(ys) * len(xs), 2), np.int32)
    flow = np.zeros((len(ys) * len(xs), 2), np.float32)
    n_ssd = 0

    if explicit:
        # Plain Python lists: scalar indexing into a numpy array is several times
        # slower per access and would swamp the measurement.
        A = f1.tolist()
        B = f2.tolist()

    k = 0
    for y in ys:
        for x in xs:
            best_ssd = float("inf")
            best_u = best_v = 0
            for dy in range(-max_disp, max_disp + 1):
                for dx in range(-max_disp, max_disp + 1):
                    if explicit:
                        ssd = 0.0
                        for j in range(-window, window + 1):
                            row1 = A[y + j]
                            row2 = B[y + j + dy]
                            for i in range(-window, window + 1):
                                d = row1[x + i] - row2[x + i + dx]
                                ssd += d * d
                    else:
                        p1 = f1[y - window:y + window + 1, x - window:x + window + 1]
                        p2 = f2[y + dy - window:y + dy + window + 1,
                                x + dx - window:x + dx + window + 1]
                        diff = p1 - p2
                        ssd = float(np.dot(diff.ravel(), diff.ravel()))
                    n_ssd += 1
                    if ssd < best_ssd:
                        best_ssd = ssd
                        best_u, best_v = dx, dy
            points[k] = (x, y)
            flow[k] = (best_u, best_v)
            k += 1

    return points, flow, n_ssd


# --- 3. scoring ------------------------------------------------------------
def score_flow(flow, true_shift):
    """Mean absolute error per component, and the exactly-recovered fraction."""
    truth = np.asarray(true_shift, np.float32)
    err = np.abs(flow - truth)
    exact = np.all(np.isclose(flow, truth), axis=1).mean()
    return {
        "mae": float(err.mean()),
        "mae_u": float(err[:, 0].mean()),
        "mae_v": float(err[:, 1].mean()),
        "exact_fraction": float(exact),
        "mean_u": float(flow[:, 0].mean()),
        "mean_v": float(flow[:, 1].mean()),
    }


def run_timed(f1, f2, **kw):
    t0 = time.perf_counter()
    points, flow, n_ssd = block_match_flow(f1, f2, **kw)
    return points, flow, n_ssd, time.perf_counter() - t0


# --- 4. parameter sweep ----------------------------------------------------
def sweep(f1, f2, true_shift, param, values, fixed):
    """Vary one slider, hold the other, record error and runtime."""
    rows = []
    for v in values:
        kw = dict(fixed)
        kw[param] = v
        # The sweep must run the unrolled loop. The numpy-patch path spends most
        # of its time in per-call overhead rather than in the patch, so its
        # runtime is nearly flat in `window` - it would hide the very quadratic
        # this sweep exists to show.
        _, flow, n_ssd, dt = run_timed(f1, f2, explicit=True, **kw)
        s = score_flow(flow, true_shift)
        ops = (2 * kw["max_disp"] + 1) ** 2 * (2 * kw["window"] + 1) ** 2
        rows.append({param: v, "mae": s["mae"], "time": dt,
                     "n_ssd": n_ssd, "ops_per_point": ops})
    return rows


def print_sweep(rows, param, label):
    print(f"\n  varying {label}")
    print("  " + "-" * 74)
    print(f"  {label:>10s}{'ops/point':>14s}{'SSD evals':>12s}"
          f"{'MAE px':>10s}{'time s':>10s}{'time ratio':>13s}")
    print("  " + "-" * 74)
    base = rows[0]["time"]
    for r in rows:
        print(f"  {r[param]:>10d}{r['ops_per_point']:>14,d}{r['n_ssd']:>12,d}"
              f"{r['mae']:>10.3f}{r['time']:>10.2f}{r['time'] / base:>12.1f}x")
    print("  " + "-" * 74)


# --- 5. figures ------------------------------------------------------------
def _style(ax):
    ax.set_facecolor(C_SURFACE)
    ax.grid(True, color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d5d4d0")
    ax.tick_params(colors=C_MUTED, length=0)


def plot_flow_field(f1, points, flow, true_shift, stats, dt, path):
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(C_SURFACE)
    ax.imshow(f1, cmap="gray", interpolation="nearest")
    # Every sample point is solved for, but drawing all of them at a legible
    # arrow length covers the frame; show every second one in each direction.
    ux, uy = np.unique(points[:, 0]), np.unique(points[:, 1])
    keep = (np.isin(points[:, 0], ux[::2]) & np.isin(points[:, 1], uy[::2]))
    ax.quiver(points[keep, 0], points[keep, 1], flow[keep, 0], flow[keep, 1],
              color=C_ERR, angles="xy", scale_units="xy", scale=0.22,
              width=0.004, headwidth=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Recovered flow, window={WINDOW}, max_disp={MAX_DISP}, step={STEP}",
                 fontsize=13, color=C_INK, pad=12)
    txt = (f"true shift      (u, v) = ({true_shift[0]:.2f}, {true_shift[1]:.2f})\n"
           f"mean recovered  (u, v) = ({stats['mean_u']:.2f}, {stats['mean_v']:.2f})\n"
           f"MAE = {stats['mae']:.3f} px      exact = {100 * stats['exact_fraction']:.1f}%\n"
           f"{len(points)} sample points in {dt:.2f} s")
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, va="bottom", ha="left",
            fontsize=10, family="monospace", color=C_INK,
            bbox=dict(boxstyle="round,pad=0.5", fc=C_SURFACE, ec="#d5d4d0", lw=0.8))
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    print(f"\nfigure written: {path}")


def plot_sweeps(win_rows, disp_rows, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.patch.set_facecolor(C_SURFACE)

    panels = [
        (axes[0, 0], win_rows, "window", "MAE (px)", "mae", C_ERR,
         "Error vs patch size"),
        (axes[0, 1], win_rows, "window", "runtime (s)", "time", C_TIME,
         "Runtime vs patch size"),
        (axes[1, 0], disp_rows, "max_disp", "MAE (px)", "mae", C_ERR,
         "Error vs search range"),
        (axes[1, 1], disp_rows, "max_disp", "runtime (s)", "time", C_TIME,
         "Runtime vs search range"),
    ]
    for ax, rows, param, ylab, key, color, title in panels:
        xs = [r[param] for r in rows]
        ys = [r[key] for r in rows]
        ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=8, zorder=3)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9, color=C_INK)
        ax.set_xlabel(param, fontsize=10, color=C_MUTED)
        ax.set_ylabel(ylab, fontsize=10, color=C_MUTED)
        ax.set_title(title, fontsize=12, color=C_INK)
        ax.set_xticks(xs)
        # Error can be identically zero; pin the floor so the axis does not
        # invent negative error values.
        ax.set_ylim(0, max(ys) * 1.35 if max(ys) > 0 else 1.0)
        _style(ax)

    axes[0, 1].text(0.03, 0.93,
                    "cost ~ (2w+1)^2 : quadratic in patch size",
                    transform=axes[0, 1].transAxes, fontsize=9.5, color=C_MUTED,
                    va="top")
    axes[1, 1].text(0.03, 0.93,
                    "cost ~ (2d+1)^2 : quadratic in search range",
                    transform=axes[1, 1].transAxes, fontsize=9.5, color=C_MUTED,
                    va="top")

    fig.suptitle("The applet's two sliders are both quadratic in cost",
                 fontsize=14, color=C_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    print(f"figure written: {path}")


# --- main ------------------------------------------------------------------
def main():
    print("=" * 78)
    print("DEMO 3 - block-matching optical flow".center(78))
    print("=" * 78)
    print(f"\nframes {SIZE[1]}x{SIZE[0]} px, seed {SEED}")

    # --- headline run: the fully unrolled search ----------------------------
    f1, f2, true_shift = make_frame_pair(shift=(3, 2))
    print(f"true displacement (u, v) = ({true_shift[0]:.1f}, {true_shift[1]:.1f}) px\n")
    print(f"[1] Exhaustive search, window={WINDOW}, max_disp={MAX_DISP}, step={STEP}")
    print("-" * 78)
    print("    running the unrolled loop (this is the applet's actual cost) ...")
    points, flow, n_ssd, dt = run_timed(f1, f2, explicit=True)
    stats = score_flow(flow, true_shift)

    patch_px = (2 * WINDOW + 1) ** 2
    disps = (2 * MAX_DISP + 1) ** 2
    theory_per_point = disps * patch_px
    measured_pixel_ops = n_ssd * patch_px

    print(f"    sample points            {len(points):>14,d}")
    print(f"    displacements per point  {disps:>14,d}   (2*{MAX_DISP}+1)^2")
    print(f"    pixels per SSD           {patch_px:>14,d}   (2*{WINDOW}+1)^2")
    print(f"    SSD evaluations          {n_ssd:>14,d}   = points * displacements")
    print(f"    theoretical ops/point    {theory_per_point:>14,d}   "
          f"= (2*{MAX_DISP}+1)^2 * (2*{WINDOW}+1)^2")
    print(f"    measured ops/point       {measured_pixel_ops // len(points):>14,d}   "
          f"= SSD evals * pixels / points")
    match = measured_pixel_ops // len(points) == theory_per_point
    print(f"    match                    {str(match).upper():>14s}")
    print(f"    total pixel operations   {measured_pixel_ops:>14,d}")
    print()
    print(f"    mean absolute error      {stats['mae']:>14.4f} px  "
          f"(u {stats['mae_u']:.4f}, v {stats['mae_v']:.4f})")
    print(f"    recovered exactly        {100 * stats['exact_fraction']:>13.1f}%")
    print(f"    wall clock               {dt:>14.2f} s")
    print(f"    throughput               {n_ssd / dt:>14,.0f} SSD/s")

    # Confirm the fast path is the same algorithm before using it in the sweeps.
    _, flow_fast, n_ssd_fast, dt_fast = run_timed(f1, f2, explicit=False)
    identical = np.array_equal(flow, flow_fast) and n_ssd == n_ssd_fast
    print(f"\n    numpy-patch path agrees with the unrolled loop: "
          f"{str(identical).upper()} ({dt / dt_fast:.1f}x faster, "
          f"same {n_ssd_fast:,d} displacements visited)")
    assert identical, "fast path diverged from the reference loop"

    # --- 4. the two sliders -------------------------------------------------
    print("\n[2] Parameter sweep - the applet's two sliders")
    print("-" * 78)
    win_rows = sweep(f1, f2, true_shift, "window", [3, 5, 7, 9, 11],
                     {"max_disp": MAX_DISP, "step": STEP})
    print_sweep(win_rows, "window", "window")
    disp_rows = sweep(f1, f2, true_shift, "max_disp", [2, 3, 5, 8, 11],
                      {"window": WINDOW, "step": STEP})
    print_sweep(disp_rows, "max_disp", "max_disp")

    w_ops = win_rows[-1]["ops_per_point"] / win_rows[0]["ops_per_point"]
    d_ops = disp_rows[-1]["ops_per_point"] / disp_rows[0]["ops_per_point"]
    w_time = win_rows[-1]["time"] / win_rows[0]["time"]
    d_time = disp_rows[-1]["time"] / disp_rows[0]["time"]
    print(f"\n  Both sliders are quadratic, and the clock agrees with the count.")
    print(f"  Patch size 3 -> 11 multiplies the work per point by {w_ops:.1f}x and the")
    print(f"  measured runtime by {w_time:.1f}x. Search range 2 -> 11 multiplies the work")
    print(f"  per point by {d_ops:.1f}x and the runtime by {d_time:.1f}x.")
    print()
    print(f"  Only one of those purchases changes the answer. max_disp=2 cannot")
    print(f"  represent a 3 px horizontal motion, so the search saturates at its")
    print(f"  own boundary and reports MAE {disp_rows[0]['mae']:.3f}; every larger range")
    print(f"  recovers the shift exactly and the extra cost buys nothing. Patch")
    print(f"  size buys nothing at all on this texture. The trap is that the")
    print(f"  saturated result at max_disp=2 looks like a perfectly ordinary flow")
    print(f"  field - the method cannot report that the true motion was outside")
    print(f"  the range it searched.")

    plot_sweeps(win_rows, disp_rows,
                os.path.join(OUT_DIR, "block_matching_sweep.png"))
    plot_flow_field(f1, points, flow, true_shift, stats, dt,
                    os.path.join(OUT_DIR, "block_matching_flow.png"))

    # --- 6. the sub-pixel wall ---------------------------------------------
    print("\n[3] Sub-pixel displacement - the method's hard limit")
    print("-" * 78)
    sub_shift = (2.5, 1.5)
    g1, g2, sub_true = make_frame_pair(shift=sub_shift)
    _, sub_flow, _, sub_dt = run_timed(g1, g2, explicit=True)
    sub_stats = score_flow(sub_flow, sub_true)

    print(f"{'':4s}{'':<26s}{'true':>12s}{'mean recovered':>18s}{'MAE px':>10s}")
    print("-" * 78)
    for label, truth, st in [("integer shift", true_shift, stats),
                             ("sub-pixel shift", sub_true, sub_stats)]:
        t_str = "({:.1f}, {:.1f})".format(truth[0], truth[1])
        r_str = "({:.2f}, {:.2f})".format(st["mean_u"], st["mean_v"])
        print(f"    {label:<26s}{t_str:>12s}{r_str:>18s}{st['mae']:>10.3f}")
    print("-" * 78)
    print(f"    sub-pixel run: {100 * sub_stats['exact_fraction']:.1f}% exact, "
          f"{sub_dt:.2f} s")
    print()
    print("    The search enumerates integer offsets, so (2.5, 1.5) is not in the")
    print("    hypothesis space - no amount of extra search range or patch size")
    print("    can reach it. Each component is forced to the nearer integer, and")
    print("    the best attainable MAE is 0.5 px per component by construction.")
    print("    Refining the answer requires a method that solves for a continuous")
    print("    displacement instead of enumerating discrete ones. That is")
    print("    Lucas-Kanade, in Demo 4.")

    if matplotlib.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
