"""
Demo 4 - Lucas-Kanade gradient-based optical flow.

Implements Lucas-Kanade directly from the normal equations in Szeliski,
"Computer Vision: Algorithms and Applications", Section 9.1.3, and puts it head
to head with Demo 3's exhaustive block-matching search on identical data (the
frame generator is imported from `demo3_block_matching_flow`).

The two methods sit at opposite ends of the same problem. Block matching
enumerates a finite set of integer displacements and picks the best one: robust
to any displacement inside its search range, blind to everything outside it, and
incapable of expressing a fraction of a pixel. Lucas-Kanade instead assumes the
image is locally linear, writes down one 2x2 least-squares system per point, and
solves it: continuous-valued, thousands of times cheaper, and valid only while
the first-order Taylor expansion holds - which is to say, only for small motion.

The program shows all of this, then repairs the small-motion restriction with the
coarse-to-fine pyramid of Szeliski Section 9.1.1.

Run:  python demo4_lucas_kanade.py    (run demo 3 first - this imports from it)
Out:  out/lk_comparison.png, out/lk_displacement_sweep.png
"""

import os
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from demo3_block_matching_flow import block_match_flow, make_frame_pair

# --- configuration ---------------------------------------------------------
SEED = 20260908
WINDOW = 7
STEP = 8
N_ITERS = 5
# 4 levels, not 3: the coarsest scale has to bring the motion inside the linear
# regime, and 15 px of displacement is still ~3.8 px after only two halvings -
# too large for the Gaussian-blurred texture, whose features are ~2 px wide at
# that level. Three levels visibly fails on the (12, 9) case; four succeeds.
LEVELS = 4
MIN_EIG = 1e-4
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Categorical palette (validated light-mode; worst adjacent CVD dE 9.2).
C_BM = "#2a78d6"
C_LK = "#eb6834"
C_LKI = "#1baf7a"
C_LKP = "#4a3aa7"
C_INK = "#0b0b0b"
C_MUTED = "#52514e"
C_SURFACE = "#fcfcfb"


# --- gradients -------------------------------------------------------------
def image_gradients(f):
    """Spatial gradients, scaled to true per-pixel derivatives.

    cv2.Sobel with ksize=3 applies [[-1,0,1],[-2,0,2],[-1,0,1]], which is 8x the
    derivative. A and b scale differently under that factor (A quadratically, b
    linearly), so the solved displacement would come out 8x too small if the
    scale were left in.
    """
    ix = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0)
    iy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0)
    return ix, iy


def _grid(shape, margin, step):
    h, w = shape
    ys = range(margin, h - margin, step)
    xs = range(margin, w - margin, step)
    return [(x, y) for y in ys for x in xs]


# --- 1. single-pass Lucas-Kanade -------------------------------------------
def lucas_kanade_flow(f1, f2, window=WINDOW, step=STEP, min_eig=MIN_EIG):
    """One least-squares solve per sample point (Szeliski Eq. 9.35).

    Over a square window, accumulate the five sums and solve

        A * du = b,     A = [[ sum Ix^2 ,  sum Ix*Iy ],   b = -[[ sum Ix*It ],
                             [ sum Ix*Iy,  sum Iy^2  ]]         [ sum Iy*It ]]

    for du = (u, v). A is the windowed second-moment (structure) matrix. Where
    the window has gradient in only one direction - an edge, or a flat region -
    A is singular or near-singular and only the normal component of the flow is
    observable. That is the aperture problem, and it shows up here as a small
    eigenvalue of A; those points are skipped rather than returned as noise.

    Returns (points, flow, info); skipped points carry NaN flow.
    """
    ix, iy = image_gradients(f1)
    it = f2 - f1                       # temporal difference, forward in time

    pts = _grid(f1.shape, window + 1, step)
    points = np.array(pts, np.int32)
    flow = np.full((len(pts), 2), np.nan, np.float32)

    n_skipped = 0
    min_eigs = []
    for k, (x, y) in enumerate(pts):
        sl = (slice(y - window, y + window + 1), slice(x - window, x + window + 1))
        gx, gy, gt = ix[sl], iy[sl], it[sl]

        a11 = float(np.sum(gx * gx))
        a12 = float(np.sum(gx * gy))
        a22 = float(np.sum(gy * gy))
        A = np.array([[a11, a12], [a12, a22]], np.float64)
        b = -np.array([float(np.sum(gx * gt)), float(np.sum(gy * gt))], np.float64)

        # Smaller eigenvalue of a symmetric 2x2 - the aperture-problem test.
        tr, det = a11 + a22, a11 * a22 - a12 * a12
        disc = max(tr * tr / 4.0 - det, 0.0)
        lam_min = tr / 2.0 - np.sqrt(disc)
        min_eigs.append(lam_min)
        if lam_min < min_eig:
            n_skipped += 1
            continue

        flow[k] = np.linalg.solve(A, b)

    info = {
        "n_points": len(pts),
        "n_skipped": n_skipped,
        "min_eig_min": float(np.min(min_eigs)),
        "min_eig_median": float(np.median(min_eigs)),
        "ops_per_point": 5 * (2 * window + 1) ** 2,
    }
    return points, flow, info


# --- bilinear patch sampling (for the warping steps) -----------------------
def _patch_bilinear(img, cx, cy, window):
    """Sample a (2*window+1)^2 patch centred at the real-valued (cx, cy)."""
    x0, y0 = int(np.floor(cx)), int(np.floor(cy))
    a, b = cx - x0, cy - y0
    h, w = img.shape
    if x0 - window < 0 or y0 - window < 0 or x0 + window + 2 > w or y0 + window + 2 > h:
        return None
    ys, xs = slice(y0 - window, y0 + window + 1), slice(x0 - window, x0 + window + 1)
    ys1, xs1 = slice(y0 - window + 1, y0 + window + 2), slice(x0 - window + 1, x0 + window + 2)
    return ((1 - b) * ((1 - a) * img[ys, xs] + a * img[ys, xs1])
            + b * ((1 - a) * img[ys1, xs] + a * img[ys1, xs1]))


def _lk_point(f1, f2, ix, iy, x, y, window, d_init=(0.0, 0.0),
              n_iters=N_ITERS, min_eig=MIN_EIG, trace=None):
    """Newton-Raphson Lucas-Kanade at one point.

    Each iteration warps f2 back toward f1 by the current estimate, re-linearizes
    around that warped position, and solves the same 2x2 system for a correction.
    The gradients of f1 - and therefore A - are fixed across iterations; only b
    changes, which is what makes the loop cheap.
    """
    u, v = float(d_init[0]), float(d_init[1])
    h, w = f1.shape
    if x - window < 0 or y - window < 0 or x + window + 1 > w or y + window + 1 > h:
        return None

    sl = (slice(y - window, y + window + 1), slice(x - window, x + window + 1))
    p1, gx, gy = f1[sl], ix[sl], iy[sl]

    a11 = float(np.sum(gx * gx))
    a12 = float(np.sum(gx * gy))
    a22 = float(np.sum(gy * gy))
    tr, det = a11 + a22, a11 * a22 - a12 * a12
    lam_min = tr / 2.0 - np.sqrt(max(tr * tr / 4.0 - det, 0.0))
    if lam_min < min_eig:
        return None
    A = np.array([[a11, a12], [a12, a22]], np.float64)

    for _ in range(n_iters):
        p2 = _patch_bilinear(f2, x + u, y + v, window)
        if p2 is None:
            break                                   # warped out of the frame
        gt = p2 - p1
        b = -np.array([float(np.sum(gx * gt)), float(np.sum(gy * gt))], np.float64)
        du = np.linalg.solve(A, b)
        u += float(du[0])
        v += float(du[1])
        if trace is not None:
            trace.append((u, v, float(np.hypot(*du))))
        if np.hypot(*du) < 1e-4:
            break
    return u, v


# --- 2. iterative Lucas-Kanade ---------------------------------------------
def lucas_kanade_iterative(f1, f2, window=WINDOW, step=STEP, n_iters=N_ITERS,
                           trace_point=None):
    """Single-scale Lucas-Kanade with the warp-and-resolve loop."""
    ix, iy = image_gradients(f1)
    pts = _grid(f1.shape, window + 2, step)
    points = np.array(pts, np.int32)
    flow = np.full((len(pts), 2), np.nan, np.float32)

    n_skipped = 0
    traces = {}
    for k, (x, y) in enumerate(pts):
        trace = [] if (trace_point is not None and k == trace_point) else None
        d = _lk_point(f1, f2, ix, iy, x, y, window, (0.0, 0.0), n_iters, trace=trace)
        if trace is not None:
            traces[(x, y)] = trace
        if d is None:
            n_skipped += 1
            continue
        flow[k] = d

    info = {
        "n_points": len(pts),
        "n_skipped": n_skipped,
        "traces": traces,
        "ops_per_point": 5 * (2 * window + 1) ** 2 * n_iters,
    }
    return points, flow, info


# --- 4. pyramidal (coarse-to-fine) Lucas-Kanade ----------------------------
def lucas_kanade_pyramidal(f1, f2, window=WINDOW, step=STEP, levels=LEVELS,
                           n_iters=N_ITERS):
    """Coarse-to-fine Lucas-Kanade (Szeliski Section 9.1.1).

    A displacement that violates the Taylor expansion at full resolution becomes
    a small displacement after enough downsampling: halving the image halves the
    motion. Solve at the coarsest level, double the estimate, use it as the
    starting warp one level down, and repeat. Each level only ever has to find a
    correction of a pixel or two, which is exactly where the linearization is
    valid.
    """
    pyr1, pyr2 = [f1], [f2]
    for _ in range(levels - 1):
        pyr1.append(cv2.pyrDown(pyr1[-1]))
        pyr2.append(cv2.pyrDown(pyr2[-1]))
    grads = [image_gradients(g) for g in pyr1]

    # Margin large enough that a full-resolution point is still interior at the
    # coarsest level, where one pixel covers 2^(levels-1) original pixels.
    margin = (window + 2) * (2 ** (levels - 1))
    pts = _grid(f1.shape, margin, step)
    points = np.array(pts, np.int32)
    flow = np.full((len(pts), 2), np.nan, np.float32)

    n_skipped = 0
    for k, (x, y) in enumerate(pts):
        d = (0.0, 0.0)
        ok = True
        for lvl in range(levels - 1, -1, -1):
            s = 2 ** lvl
            gx, gy = grads[lvl]
            res = _lk_point(pyr1[lvl], pyr2[lvl], gx, gy,
                            int(round(x / s)), int(round(y / s)),
                            window, d, n_iters)
            if res is None:
                ok = False
                break
            d = res
            if lvl > 0:
                d = (d[0] * 2.0, d[1] * 2.0)     # upscale into the next level
        if not ok:
            n_skipped += 1
            continue
        flow[k] = d

    info = {
        "n_points": len(pts),
        "n_skipped": n_skipped,
        "ops_per_point": 5 * (2 * window + 1) ** 2 * n_iters * levels,
    }
    return points, flow, info


# --- scoring ---------------------------------------------------------------
def score(flow, true_shift):
    """Mean absolute error over the points a method actually returned."""
    truth = np.asarray(true_shift, np.float64)
    valid = ~np.isnan(flow).any(axis=1)
    if not valid.any():
        return {"mae": float("nan"), "mean_u": float("nan"),
                "mean_v": float("nan"), "coverage": 0.0}
    f = flow[valid].astype(np.float64)
    return {
        "mae": float(np.abs(f - truth).mean()),
        "mean_u": float(f[:, 0].mean()),
        "mean_v": float(f[:, 1].mean()),
        "coverage": float(valid.mean()),
    }


def timed(fn, *args, **kw):
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    return out, time.perf_counter() - t0


def run_all(f1, f2, true_shift, max_disp, bm_explicit=True, bm_step=STEP):
    """Run all four methods on one frame pair and return their results."""
    results = {}

    (bp, bf, bn), bt = timed(block_match_flow, f1, f2, window=WINDOW,
                             max_disp=max_disp, step=bm_step,
                             explicit=bm_explicit)
    results["block matching"] = {
        "points": bp, "flow": bf.astype(np.float64), "time": bt,
        "ops_per_point": (2 * max_disp + 1) ** 2 * (2 * WINDOW + 1) ** 2,
        "note": f"exhaustive, max_disp={max_disp}",
        "n_points": len(bp), "n_skipped": 0, "color": C_BM,
        **score(bf.astype(np.float64), true_shift),
    }

    for name, fn, color, note in [
        ("LK single-pass", lucas_kanade_flow, C_LK, "1 solve/point"),
        ("LK iterative", lucas_kanade_iterative, C_LKI, f"{N_ITERS} warp+solve"),
        ("LK pyramidal", lucas_kanade_pyramidal, C_LKP,
         f"{LEVELS} levels x {N_ITERS} iters"),
    ]:
        (p, fl, info), dt = timed(fn, f1, f2, window=WINDOW, step=STEP)
        results[name] = {
            "points": p, "flow": fl.astype(np.float64), "time": dt,
            "ops_per_point": info["ops_per_point"], "note": note,
            "n_points": info["n_points"], "n_skipped": info["n_skipped"],
            "color": color, **score(fl.astype(np.float64), true_shift),
        }
    return results


# --- printing --------------------------------------------------------------
def print_table(results, true_shift, title):
    print(f"\n  {title}  -  true (u, v) = ({true_shift[0]:.2f}, {true_shift[1]:.2f})")
    print("  " + "-" * 96)
    print(f"  {'method':<18s}{'mean recovered':>18s}{'MAE px':>10s}"
          f"{'time s':>9s}{'ops/point':>12s}{'pts':>6s}{'skip':>6s}   {'cost note':<20s}")
    print("  " + "-" * 96)
    for name, r in results.items():
        rec = "({:.3f}, {:.3f})".format(r["mean_u"], r["mean_v"])
        print(f"  {name:<18s}{rec:>18s}{r['mae']:>10.4f}"
              f"{r['time']:>9.2f}{r['ops_per_point']:>12,d}"
              f"{r['n_points']:>6d}{r['n_skipped']:>6d}   {r['note']:<20s}")
    print("  " + "-" * 96)


# --- 5. figures ------------------------------------------------------------
def plot_quivers(f1, results, true_shift, path):
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.9))
    fig.patch.set_facecolor(C_SURFACE)

    for ax, (name, r) in zip(axes, results.items()):
        ax.imshow(f1, cmap="gray", interpolation="nearest")
        p, fl = r["points"], r["flow"]
        valid = ~np.isnan(fl).any(axis=1)
        p, fl = p[valid], fl[valid]
        ux, uy = np.unique(p[:, 0]), np.unique(p[:, 1])
        keep = np.isin(p[:, 0], ux[::3]) & np.isin(p[:, 1], uy[::3])
        ax.quiver(p[keep, 0], p[keep, 1], fl[keep, 0], fl[keep, 1],
                  color=r["color"], angles="xy", scale_units="xy",
                  scale=0.30, width=0.006, headwidth=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name}\nMAE {r['mae']:.3f} px   {r['time']:.2f} s",
                     fontsize=12, color=C_INK, pad=8)

    fig.suptitle(f"Four flow estimators on one frame pair, "
                 f"true shift ({true_shift[0]:.1f}, {true_shift[1]:.1f}) px",
                 fontsize=14, color=C_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    print(f"\nfigure written: {path}")


def plot_displacement_sweep(mags, curves, colors, path):
    fig, ax = plt.subplots(figsize=(10, 6.4))
    fig.patch.set_facecolor(C_SURFACE)
    ax.set_facecolor(C_SURFACE)

    # Line style is a second channel alongside hue: block matching and pyramidal
    # LK are both identically zero here, so colour alone would hide one of them.
    styles = {"block matching": ("-", 3.2, 8),
              "LK single-pass": ("-", 2.0, 7),
              "LK iterative": ("-", 2.0, 7),
              "LK pyramidal": ("--", 2.0, 6)}
    for name, ys in curves.items():
        ls, lw, ms = styles.get(name, ("-", 2.0, 7))
        ax.plot(mags, ys, linestyle=ls, marker="o", color=colors[name],
                linewidth=lw, markersize=ms, label=name, zorder=3)

    # Direct labels only where the curves are actually apart.
    for name in ("LK single-pass", "LK iterative"):
        y = curves[name][-1]
        dy = 10 if name == "LK iterative" else -12
        ax.annotate(name, (mags[-1], y), textcoords="offset points",
                    xytext=(9, dy), va="center", fontsize=9.5,
                    color=C_MUTED, annotation_clip=False)
    ax.annotate("block matching and LK pyramidal\ncoincide at 0.000 px throughout",
                (mags[len(mags) // 2], 0), textcoords="offset points",
                xytext=(0, 26), ha="center", fontsize=9.5, color=C_MUTED,
                arrowprops=dict(arrowstyle="-", color="#c9c8c3", lw=1))

    ax.set_xlabel("true displacement magnitude (px)", fontsize=11, color=C_MUTED)
    ax.set_ylabel("mean absolute error (px)", fontsize=11, color=C_MUTED)
    ax.set_title("Where each method breaks down", fontsize=13, color=C_INK, pad=12)
    ax.set_xticks(mags)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_ylim(0, max(max(v) for v in curves.values()) * 2.0)
    ax.grid(True, color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d5d4d0")
    ax.tick_params(colors=C_MUTED, length=0)
    ax.legend(frameon=False, fontsize=10, loc="upper left", labelcolor=C_MUTED)
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    print(f"figure written: {path}")


# --- main ------------------------------------------------------------------
def main():
    print("=" * 100)
    print("DEMO 4 - Lucas-Kanade gradient-based optical flow".center(100))
    print("=" * 100)
    print(f"\nframes 320x240 px, seed {SEED}, window={WINDOW}, step={STEP}, "
          f"iters={N_ITERS}, levels={LEVELS}")

    # --- 1. the aperture-problem guard --------------------------------------
    f1, f2, shift_int = make_frame_pair(shift=(3, 2))
    _, _, info = lucas_kanade_flow(f1, f2)
    print("\n[1] Conditioning of the 2x2 system A (the aperture-problem guard)")
    print("-" * 100)
    print(f"    sample points                {info['n_points']:>10d}")
    print(f"    skipped, lambda_min < {MIN_EIG:g}   {info['n_skipped']:>10d}")
    print(f"    smallest lambda_min seen     {info['min_eig_min']:>10.4f}")
    print(f"    median lambda_min            {info['min_eig_median']:>10.4f}")
    print("    The blurred-noise texture has gradient in every direction at every")
    print("    window, so A is well conditioned throughout and nothing is skipped.")
    print("    The guard is not decoration: on a single straight edge, or on a flat")
    print("    region, lambda_min collapses and only the flow component normal to")
    print("    the edge is observable - the aperture problem. Points like that must")
    print("    be reported as unsolvable, not returned as confident numbers.")

    # --- 2. watching the Newton-Raphson loop converge -----------------------
    print(f"\n[2] Iterative refinement at one sample point, true (u, v) = (3.0, 2.0)")
    print("-" * 100)
    _, _, it_info = lucas_kanade_iterative(f1, f2, trace_point=0)
    (px, py), trace = next(iter(it_info["traces"].items()))
    print(f"    point ({px}, {py})")
    print(f"    {'iter':>6s}{'u':>12s}{'v':>12s}{'|du|':>14s}")
    print("    " + "-" * 44)
    print(f"    {0:>6d}{0.0:>12.5f}{0.0:>12.5f}{'-':>14s}")
    for i, (u, v, mag) in enumerate(trace, 1):
        print(f"    {i:>6d}{u:>12.5f}{v:>12.5f}{mag:>14.2e}")
    print("    " + "-" * 44)
    print("    The first solve already lands close; the later iterations remove the")
    print("    residual left by linearizing about the wrong point. Convergence is")
    print("    quadratic once the estimate is inside the linear regime, which is")
    print("    what makes this a Newton-Raphson method rather than a search.")

    # --- 3. head to head, small motion --------------------------------------
    print("\n[3] Head to head: block matching vs Lucas-Kanade")
    print("-" * 100)
    res_int = run_all(f1, f2, shift_int, max_disp=5)
    print_table(res_int, shift_int, "integer shift")

    g1, g2, shift_sub = make_frame_pair(shift=(2.5, 1.5))
    res_sub = run_all(g1, g2, shift_sub, max_disp=5)
    print_table(res_sub, shift_sub, "sub-pixel shift")

    bm = res_sub["block matching"]
    lk = res_sub["LK iterative"]
    print(f"\n  On the sub-pixel shift block matching reports "
          f"({bm['mean_u']:.2f}, {bm['mean_v']:.2f}) for a true (2.50, 1.50): it can only")
    print(f"  name integers, so its MAE floor is 0.5 px per component no matter how")
    print(f"  long it searches. Iterative Lucas-Kanade returns "
          f"({lk['mean_u']:.3f}, {lk['mean_v']:.3f}), MAE {lk['mae']:.4f} px -")
    print(f"  a displacement block matching cannot express at all. It also does this")
    print(f"  for {bm['ops_per_point'] / lk['ops_per_point']:.0f}x fewer operations "
          f"per point ({bm['ops_per_point']:,d} vs {lk['ops_per_point']:,d}).")

    # --- 4. large displacement, and the pyramid that fixes it ---------------
    print("\n[4] Large displacement: where the Taylor expansion fails")
    print("-" * 100)
    h1, h2, shift_big = make_frame_pair(shift=(12, 9))
    res_big = run_all(h1, h2, shift_big, max_disp=13)
    print_table(res_big, shift_big, "large shift")

    lk1 = res_big["LK single-pass"]
    lkp = res_big["LK pyramidal"]
    bmb = res_big["block matching"]
    print(f"\n  Single-pass Lucas-Kanade collapses: it reports "
          f"({lk1['mean_u']:.2f}, {lk1['mean_v']:.2f}) against a true (12, 9),")
    print(f"  MAE {lk1['mae']:.2f} px. Nothing is wrong with the solve - the premise is")
    print(f"  wrong. I(x+u) ~ I(x) + u*Ix holds over a pixel or two, and 15 px of")
    print(f"  motion on a texture this fine leaves f2 uncorrelated with f1 inside the")
    print(f"  window, so It carries no usable information about the displacement.")
    print(f"  Block matching is untroubled ({bmb['mae']:.3f} px) because it never")
    print(f"  linearizes anything - but it needed max_disp=13, which cost it")
    print(f"  {bmb['ops_per_point']:,d} operations per point and {bmb['time']:.1f} s.")
    print()
    mag = float(np.hypot(*shift_big))
    print(f"  The pyramid repairs Lucas-Kanade without giving up its cost advantage:")
    print(f"  {LEVELS} levels bring {mag:.0f} px of motion down to "
          f"{mag / 2 ** (LEVELS - 1):.1f} px at the coarsest")
    print(f"  scale, where the linearization is valid again. Result: "
          f"({lkp['mean_u']:.2f}, {lkp['mean_v']:.2f}),")
    print(f"  MAE {lkp['mae']:.3f} px in {lkp['time']:.2f} s - "
          f"{bmb['time'] / lkp['time']:.0f}x faster than the search that matched it.")
    print(f"  It pays for the margin: a point must stay interior at every level, so")
    print(f"  the usable grid shrinks to {lkp['n_points']} points against "
          f"{bmb['n_points']} for block matching.")

    plot_quivers(h1, res_big, shift_big, os.path.join(OUT_DIR, "lk_comparison.png"))

    # --- 5. where each method breaks down -----------------------------------
    print("\n[5] Sweeping the true displacement from 1 to 15 px (horizontal)")
    print("-" * 100)
    mags = list(range(1, 16))
    names = ["block matching", "LK single-pass", "LK iterative", "LK pyramidal"]
    curves = {n: [] for n in names}
    colors = {}
    print(f"  {'shift px':>9s}" + "".join(f"{n:>18s}" for n in names))
    print("  " + "-" * 81)
    for m in mags:
        s1, s2, truth = make_frame_pair(shift=(m, 0))
        # Block matching uses the numpy-patch path here only: 15 shifts at
        # max_disp=16 unrolled is several minutes, and this panel plots error,
        # not time. The displacements visited - and so the answer - are identical.
        r = run_all(s1, s2, truth, max_disp=16, bm_explicit=False, bm_step=16)
        for n in names:
            curves[n].append(r[n]["mae"])
            colors[n] = r[n]["color"]
        print(f"  {m:>9d}" + "".join(f"{r[n]['mae']:>18.3f}" for n in names))
    print("  " + "-" * 81)

    lk_break = next((m for m, e in zip(mags, curves["LK single-pass"]) if e > 1.0), None)
    lkp_worst = max(curves["LK pyramidal"])
    iter_break = next((m for m, e in zip(mags, curves["LK iterative"]) if e > 1.0), None)
    worst = ("exact to 3 decimals" if lkp_worst < 5e-4
             else f"under {lkp_worst:.3f} px")
    print(f"\n  Single-pass Lucas-Kanade passes 1 px of error at a true displacement of")
    print(f"  {lk_break} px and is useless beyond it. Iterating buys a little more room -")
    print(f"  it holds to {iter_break - 1} px - because each warp re-linearizes closer to the")
    print(f"  answer, but it cannot help once the first It carries no signal at all.")
    print(f"  Pyramidal Lucas-Kanade is {worst} across the whole range, at a fraction")
    print(f"  of the search cost. Block matching also holds throughout, but only")
    print(f"  because max_disp=16 was chosen in advance to contain every shift in")
    print(f"  the sweep - the one piece of information you do not have in practice.")

    plot_displacement_sweep(mags, curves, colors,
                            os.path.join(OUT_DIR, "lk_displacement_sweep.png"))

    if matplotlib.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
