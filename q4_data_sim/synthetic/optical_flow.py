"""
Optical flow on frame pairs with known motion.

Reading: Szeliski, "Computer Vision: Algorithms and Applications," 2nd ed.,
2022, Chapter 9 (paper reference [2]); Homework 1.

The sparse methods come unchanged from the RA1 demos in
`reading_assignments/ra1_optical_flow/`, imported rather than copied:

    block matching            exhaustive integer search (demo 3)
    Lucas-Kanade single-pass  one 2x2 least-squares solve per point (demo 4)
    Lucas-Kanade iterative    warp and re-solve, 5 iterations (demo 4)
    Lucas-Kanade pyramidal    coarse-to-fine over 4 levels (demo 4)

OpenCV's Farneback method adds a dense field: one flow vector per pixel from a
polynomial expansion of each neighborhood.

Experiment 1. The RA1 frame generator shifts a blurred-noise texture by a known
(dx, dy). Each method runs on three shifts: (3, 2), an integer motion;
(2.5, 1.5), a sub-pixel motion; and (12, 9), a large motion. The score is the
endpoint error (EPE): the mean distance in pixels between the estimated and
the true flow vector, over the points a method returns.

Experiment 2. A dense field needs motion that varies across the image, so the
second frame here is the first one rotated by 4 degrees and scaled by 1.03
about the center. The figure encodes flow in HSV: hue gives the direction,
saturation and value give the magnitude. This is the color space of reference
[1] applied to motion, and it inherits the same weakness: where the flow is
near zero, its direction (hue) is undefined.

Experiment 3. The aperture check. On a single straight edge, the structure
tensor (the 2x2 matrix A that Lucas-Kanade inverts) has one eigenvalue near
zero, so only the flow component normal to the edge can be measured.

Run:  python optical_flow.py
Out:  results/optical_flow_epe.png, results/optical_flow_dense.png
"""

import cv2
import numpy as np

import common as C

C.use_reading_assignment(C.RA1_DIR)

from demo3_block_matching_flow import block_match_flow, make_frame_pair  # noqa: E402
from demo4_lucas_kanade import (MIN_EIG, image_gradients,  # noqa: E402
                                lucas_kanade_flow, lucas_kanade_iterative,
                                lucas_kanade_pyramidal)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

SHIFTS = [(3, 2), (2.5, 1.5), (12, 9)]
MAX_DISP = 13                   # block-matching search range; covers (12, 9)
DENSE_MARGIN = 20               # border excluded from Farneback scoring
ROTATION_DEG = 4.0
SCALE = 1.03
METHODS = ["block matching", "LK single-pass", "LK iterative", "LK pyramidal",
           "Farneback (dense)"]


# --- methods ---------------------------------------------------------------
def farneback(f1, f2, poly_n=7, poly_sigma=1.5):
    """OpenCV Farneback with one of its two documented polynomial settings.

    The default here is poly_n=7, poly_sigma=1.5. The other documented pair,
    poly_n=5, poly_sigma=1.1, fails on the (12, 9) shift; main() reports both.
    """
    a = np.clip(f1 * 255, 0, 255).astype(np.uint8)
    b = np.clip(f2 * 255, 0, 255).astype(np.uint8)
    return cv2.calcOpticalFlowFarneback(a, b, None, pyr_scale=0.5, levels=4,
                                        winsize=15, iterations=3, poly_n=poly_n,
                                        poly_sigma=poly_sigma, flags=0)


def endpoint_error(flow, truth):
    """Mean endpoint error over valid (non-NaN) vectors, and their fraction."""
    flow = np.asarray(flow, np.float64).reshape(-1, 2)
    truth = np.broadcast_to(np.asarray(truth, np.float64), flow.shape)
    valid = ~np.isnan(flow).any(axis=1)
    epe = np.hypot(*(flow[valid] - truth[valid]).T)
    return float(epe.mean()), float(valid.mean())


def run_methods(f1, f2, shift):
    out = {}
    _, fl, _ = block_match_flow(f1, f2, max_disp=MAX_DISP, explicit=False)
    out["block matching"] = endpoint_error(fl, shift)
    for name, fn in (("LK single-pass", lucas_kanade_flow),
                     ("LK iterative", lucas_kanade_iterative),
                     ("LK pyramidal", lucas_kanade_pyramidal)):
        _, fl, _ = fn(f1, f2)
        out[name] = endpoint_error(fl, shift)
    m = DENSE_MARGIN
    dense = farneback(f1, f2)[m:-m, m:-m]
    out["Farneback (dense)"] = endpoint_error(dense, shift)
    return out


# --- dense experiment ------------------------------------------------------
def rotate_scale_pair(f1):
    """Second frame and the exact flow field for a rotation plus scaling."""
    h, w = f1.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ROTATION_DEG, SCALE)
    f2 = cv2.warpAffine(f1, M, (w, h), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    u = M[0, 0] * xx + M[0, 1] * yy + M[0, 2] - xx
    v = M[1, 0] * xx + M[1, 1] * yy + M[1, 2] - yy
    return f2, np.dstack([u, v])


def flow_to_hsv_rgb(flow, max_mag):
    """Hue = direction; saturation and value = magnitude / max_mag."""
    u, v = flow[..., 0], flow[..., 1]
    hsv = np.zeros(flow.shape[:2] + (3,), np.float32)
    hsv[..., 0] = np.rad2deg(np.arctan2(v, u)) % 360
    mag = np.clip(np.hypot(u, v) / max_mag, 0, 1)
    hsv[..., 1] = mag
    hsv[..., 2] = mag
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def color_wheel(size=201):
    r = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(r, r)
    rgb = flow_to_hsv_rgb(np.dstack([xx, yy]), 1.0)
    rgb[np.hypot(xx, yy) > 1] = 1.0
    return rgb


# --- aperture check --------------------------------------------------------
def structure_tensor(img, x, y, window=7):
    ix, iy = image_gradients(img.astype(np.float32))
    sl = (slice(y - window, y + window + 1), slice(x - window, x + window + 1))
    gx, gy = ix[sl].astype(np.float64), iy[sl].astype(np.float64)
    A = np.array([[np.sum(gx * gx), np.sum(gx * gy)],
                  [np.sum(gx * gy), np.sum(gy * gy)]])
    return A, gx, gy


def edge_image(shift=(0.0, 0.0), size=64, angle_deg=30.0):
    """A soft straight edge through the center, displaced by `shift`."""
    t = np.deg2rad(angle_deg)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    d = (xx - shift[0] - size / 2) * np.cos(t) + (yy - shift[1] - size / 2) * np.sin(t)
    return 0.5 + 0.4 * np.tanh(d / 3.0), np.array([np.cos(t), np.sin(t)])


def aperture_check(rng):
    rows = []
    texture, _, _ = make_frame_pair(size=(64, 64), shift=(0, 0), seed=C.SEED)
    edge, normal = edge_image()
    flat = np.full((64, 64), 0.5)
    for name, img in (("textured patch", texture), ("straight edge", edge),
                      ("flat patch", flat)):
        A, _, _ = structure_tensor(img, 32, 32)
        lam = np.linalg.eigvalsh(A)
        ratio = lam[0] / lam[1] if lam[1] > 0 else float("nan")
        rows.append([name, f"{lam[1]:.3g}", f"{lam[0]:.3g}",
                     "undefined" if np.isnan(ratio) else f"{ratio:.1e}",
                     "skip" if lam[0] < MIN_EIG else "solve"])

    # What an edge reveals: solve A d = b with the pseudo-inverse, which
    # returns the minimum-norm answer when A is singular.
    true = np.array([0.6, 0.4])
    e2, _ = edge_image(shift=true)
    A, gx, gy = structure_tensor(edge, 32, 32)
    gt = (e2 - edge)[25:40, 25:40]
    b = -np.array([np.sum(gx * gt), np.sum(gy * gt)])
    # Treat eigenvalues below 0.1% of the largest as zero.
    est = np.linalg.pinv(A, rcond=1e-3) @ b
    normal_flow = (true @ normal) * normal
    return rows, true, normal_flow, est


# --- figures ---------------------------------------------------------------
def plot_epe(table, name):
    data = np.array([[table[s][m][0] for s in SHIFTS] for m in METHODS])
    fig, ax = plt.subplots(figsize=(9, 5.4))
    norm = LogNorm(vmin=0.01, vmax=max(20.0, data.max()))
    ax.imshow(data, cmap="Blues", norm=norm, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            dark = norm(val) > 0.6
            ax.text(j, i, f"{val:.3f} px", ha="center", va="center",
                    fontsize=12, color="white" if dark else C.INK)
    ax.set_xticks(range(len(SHIFTS)))
    ax.set_xticklabels([f"shift ({a:g}, {b:g})" for a, b in SHIFTS])
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(METHODS)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(SHIFTS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(METHODS)), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", length=0)

    good = [m for m in METHODS if max(table[s][m][0] for s in SHIFTS) < 0.5]
    title = (" and ".join(good) + " stay under 0.5 px on all three shifts"
             if good else "No method stays under 0.5 px on all three shifts")
    fig.suptitle(f"Endpoint error: {title}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    C.save_figure(fig, name)


def plot_dense(f1, truth, est, epe_map, mean_epe, name):
    max_mag = float(np.hypot(*truth.reshape(-1, 2).T).max())
    fig = plt.figure(figsize=(17, 4.8))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1.12, 0.7], wspace=0.18)

    panels = (
        (f1, "gray", "Frame 1"),
        (flow_to_hsv_rgb(truth, max_mag), None, "True flow (rotate 4 deg, scale 1.03)"),
        (flow_to_hsv_rgb(est, max_mag), None, "Farneback estimate"),
    )
    for k, (img, cmap, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(img, cmap=cmap, interpolation="nearest")
        C.image_axes(ax, title)

    ax = fig.add_subplot(gs[0, 3])
    im = ax.imshow(epe_map, cmap="Blues", vmin=0, vmax=max(1.0, np.percentile(epe_map, 99)))
    C.image_axes(ax, f"Endpoint error (interior mean {mean_epe:.2f} px)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("px")
    cb.outline.set_visible(False)

    ax = fig.add_subplot(gs[0, 4])
    # Row 0 of the wheel is y = -1, which image coordinates draw at the top.
    ax.imshow(color_wheel(), extent=(-1, 1, 1, -1), origin="upper")
    C.image_axes(ax, "Key: hue = direction,\nsaturation and value\n= magnitude")
    ax.text(1.12, 0, "+x", va="center", fontsize=10, color=C.MUTED)
    ax.text(0, 1.12, "+y (down)", ha="center", va="top", fontsize=10,
            color=C.MUTED, transform=ax.transData)
    ax.set_xlim(-1.3, 1.45)
    ax.set_ylim(1.3, -1.3)

    fig.suptitle("Farneback recovers a rotation-plus-zoom field to "
                 f"{mean_epe:.2f} px mean endpoint error")
    fig.text(0.5, 0.88, "Error concentrates at the border, where content enters "
             "the frame. At the center the motion is near zero, so its direction "
             "(hue) is undefined.", ha="center", fontsize=10.5, color=C.MUTED)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.03)
    C.save_figure(fig, name)


# --- main ------------------------------------------------------------------
def main():
    rng = np.random.default_rng(C.SEED)

    table = {}
    for shift in SHIFTS:
        f1, f2, truth = make_frame_pair(shift=shift, seed=C.SEED)
        table[shift] = run_methods(f1, f2, truth)

    C.print_heading("Endpoint error (px) on three known translations")
    print(f"Block matching searches +/-{MAX_DISP} px; Farneback is scored on "
          f"every pixel more than {DENSE_MARGIN} px from the border.\n")
    C.print_table(["method"] + [f"({a:g}, {b:g})" for a, b in SHIFTS],
                  [[m] + [f"{table[s][m][0]:.3f}" for s in SHIFTS]
                   for m in METHODS])
    coverage = min(table[s][m][1] for s in SHIFTS for m in METHODS)
    print(f"Lowest fraction of points with an answer, over all methods and "
          f"shifts: {100 * coverage:.0f}%.")
    C.print_heading("Farneback sensitivity to its polynomial neighborhood")
    sens = []
    for n, sigma in ((5, 1.1), (7, 1.5)):
        row = [f"poly_n={n}, poly_sigma={sigma}"]
        for shift in SHIFTS:
            f1, f2, truth = make_frame_pair(shift=shift, seed=C.SEED)
            m = DENSE_MARGIN
            dense = farneback(f1, f2, n, sigma)[m:-m, m:-m]
            row.append(f"{endpoint_error(dense, truth)[0]:.3f}")
        sens.append(row)
    C.print_table(["setting"] + [f"({a:g}, {b:g})" for a, b in SHIFTS], sens)
    print("Both settings appear in the OpenCV documentation. The 5-pixel "
          "polynomial fit fails on the large shift, so the other results use "
          "poly_n=7.")

    plot_epe(table, "optical_flow_epe.png")

    f1, _, _ = make_frame_pair(shift=(0, 0), seed=C.SEED)
    f2, truth = rotate_scale_pair(f1)
    est = farneback(f1, f2)
    m = DENSE_MARGIN
    epe_map = np.hypot(*(est - truth).transpose(2, 0, 1))
    mean_epe = float(epe_map[m:-m, m:-m].mean())
    C.print_heading("Dense flow: rotation by 4 degrees and scaling by 1.03")
    max_mag = float(np.hypot(*truth.reshape(-1, 2).T).max())
    C.print_table(["quantity", "value"],
                  [["largest true motion", f"{max_mag:.2f} px"],
                   ["Farneback mean EPE (interior)", f"{mean_epe:.3f} px"],
                   ["Farneback 95th-percentile EPE",
                    f"{np.percentile(epe_map[m:-m, m:-m], 95):.3f} px"]])
    plot_dense(f1, truth, est, epe_map, mean_epe, "optical_flow_dense.png")

    rows, true, normal_flow, est_edge = aperture_check(rng)
    C.print_heading("Aperture check: structure tensor eigenvalues (15 x 15 window)")
    C.print_table(["patch", "larger eigenvalue", "smaller eigenvalue", "ratio",
                   f"RA1 guard (min {MIN_EIG:g})"], rows)
    C.print_table(["flow vector", "u", "v"],
                  [["true motion", f"{true[0]:.3f}", f"{true[1]:.3f}"],
                   ["component normal to the edge", f"{normal_flow[0]:.3f}",
                    f"{normal_flow[1]:.3f}"],
                   ["Lucas-Kanade, pseudo-inverse", f"{est_edge[0]:.3f}",
                    f"{est_edge[1]:.3f}"]])
    print("On the edge, Lucas-Kanade can only return the normal component: "
          "motion along the edge leaves the image unchanged.")


if __name__ == "__main__":
    main()
