"""
Demo 2 - blob-centroid keypoint localization.

Reproduces the second stage of the surgical-instrument keypoint tracker of
Ghanekar et al., "Automatic Surgical Instrument Keypoint Tracking" (ISBI 2025):
a network emits a per-pixel multi-class segmentation map, and keypoint
coordinates are recovered by taking the centroid of each output blob. The
segmentation is learned; the localization stage is pure geometry, and that is
what this program isolates.

Isolating it exposes the two ways the stage fails, which are exactly the two
factors of Demo 1:

  - a class with no surviving blob produces a NULL RETURN -> detection rate falls,
    accuracy given detection is untouched
  - a class whose blob fragments still produces a coordinate, but the centroid of
    the largest surviving piece sits off the true keypoint -> detection rate stays
    at 100%, accuracy falls

Run:  python demo2_blob_centroid.py
Out:  out/blob_centroid.png
"""

import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# --- configuration ---------------------------------------------------------
SIZE = (480, 640)             # (rows, cols)
RADIUS = 5                    # keypoint disc radius, px
TAU = 15                      # correctness radius, px
SEED = 20260908
N_CLASSES = 4
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

CLASS_NAMES = {
    1: "instr A tip",
    2: "instr A jaw base",
    3: "instr B tip",
    4: "instr B jaw base",
}

# Categorical palette (validated light-mode; worst adjacent CVD dE 9.2).
CLASS_COLORS = ["#fcfcfb", "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
C_GT = "#008300"              # ground truth marker
C_PRED = "#e34948"            # prediction marker
C_INK = "#0b0b0b"
C_MUTED = "#52514e"

# Display crop: the keypoints occupy a small part of the frame, so the figure
# shows this window rather than all 640x480. Spurious blobs are placed inside it
# so the "noise" panel actually shows what it is describing.
VIEW = (130, 520, 60, 330)    # x0, x1, y0, y1

MODES = ["clean", "noise", "split", "missing"]
SPLIT_CLASS = 1               # instrument A tip fragments
MISSING_CLASS = 3             # instrument B tip vanishes


# --- 1. synthetic segmentation map -----------------------------------------
def make_synthetic_mask(size=SIZE, seed=SEED):
    """Two instruments, each contributing a tip and a jaw-base keypoint.

    Returns a uint8 label image (0 = background, 1..4 = keypoint classes) and the
    ground-truth positions as {class_id: (x, y)}.
    """
    del seed  # geometry is fixed; the seed matters only for the degradations
    h, w = size
    mask = np.zeros((h, w), np.uint8)

    gt = {
        1: (196, 168),        # instrument A, tip
        2: (262, 236),        # instrument A, jaw base
        3: (452, 152),        # instrument B, tip
        4: (392, 224),        # instrument B, jaw base
    }
    for cid, (x, y) in gt.items():
        cv2.circle(mask, (x, y), RADIUS, int(cid), -1)
    return mask, gt


def _shaft_direction(cid, gt):
    """Unit vector from a tip toward its own jaw base (the instrument axis)."""
    partner = {1: 2, 2: 1, 3: 4, 4: 3}[cid]
    v = np.array(gt[partner], float) - np.array(gt[cid], float)
    return v / np.linalg.norm(v)


# --- 2. degradations -------------------------------------------------------
def degrade_mask(mask, mode, seed=SEED, gt=None):
    """Apply one realistic segmentation failure to a clean mask."""
    out = mask.copy()
    rng = np.random.default_rng(seed)

    if mode == "clean":
        return out

    if mode == "noise":
        # Small spurious blobs of random keypoint classes elsewhere in the frame -
        # the speckle a per-pixel classifier leaves on shaft glare and tissue.
        x0, x1, y0, y1 = VIEW
        for _ in range(14):
            cid = int(rng.integers(1, N_CLASSES + 1))
            x = int(rng.integers(x0 + 10, x1 - 10))
            y = int(rng.integers(y0 + 10, y1 - 10))
            r = int(rng.integers(1, 4))
            if out[max(0, y - 12):y + 12, max(0, x - 12):x + 12].any():
                continue                       # don't touch a real keypoint
            cv2.circle(out, (x, y), r, cid, -1)
        return out

    if mode == "split":
        # A mask that leaked down the instrument shaft and then broke apart.
        # Build the leaked region as two lobes joined by a 3px-wide neck, then
        # erode: the neck disappears and the class fragments into two components,
        # the LARGER of which sits off the true keypoint.
        cid = SPLIT_CLASS
        cx, cy = gt[cid]
        d = _shaft_direction(cid, gt)
        far = (int(round(cx + 30 * d[0])), int(round(cy + 30 * d[1])))

        layer = np.zeros_like(out)
        cv2.circle(layer, (cx, cy), RADIUS - 1, 1, -1)   # lobe at the true keypoint
        cv2.circle(layer, far, RADIUS + 1, 1, -1)        # larger lobe down the shaft
        cv2.line(layer, (cx, cy), far, 1, 3)             # thin neck
        layer = cv2.erode(layer, np.ones((3, 3), np.uint8), iterations=1)

        out[out == cid] = 0
        out[layer > 0] = cid
        return out

    if mode == "missing":
        # The network found no evidence for this class at all: a null return.
        out[out == MISSING_CLASS] = 0
        return out

    raise ValueError(f"unknown mode: {mode}")


# --- 3. localization: centroid of the output segmentation blobs ------------
def localize(mask, n_classes=N_CLASSES):
    """Recover one coordinate per keypoint class from the segmentation map.

    This is the stage the paper calls centroid estimation of the output
    segmentation blobs: threshold the map for a class, find its connected
    components, and take the centroid of the largest one. A class with no
    component yields None - a null return, which is a detection failure rather
    than a localization error.
    """
    out = {}
    for cid in range(1, n_classes + 1):
        binary = (mask == cid).astype(np.uint8)
        n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        if n_labels <= 1:                      # label 0 is background
            out[cid] = None
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = 1 + int(np.argmax(areas))    # largest blob wins, per the paper
        cx, cy = centroids[largest]
        out[cid] = (float(cx), float(cy))
    return out


# --- 4. scoring ------------------------------------------------------------
def score(pred, gt, tau=TAU):
    """Per-keypoint error plus the PCK factorization from Demo 1."""
    rows = []
    n_det = n_cor = 0
    for cid in sorted(gt):
        p = pred.get(cid)
        if p is None:
            rows.append((cid, None, None, False))
            continue
        n_det += 1
        err = float(np.hypot(p[0] - gt[cid][0], p[1] - gt[cid][1]))
        ok = err < tau
        n_cor += int(ok)
        rows.append((cid, p, err, ok))

    n_gt = len(gt)
    return {
        "rows": rows,
        "n_gt": n_gt,
        "n_det": n_det,
        "n_cor": n_cor,
        "detection_rate": n_det / n_gt,
        "accuracy_given_detection": (n_cor / n_det) if n_det else 0.0,
        "pck": n_cor / n_gt,
    }


# --- 5. figure -------------------------------------------------------------
def plot_modes(masks, results, gt, path):
    cmap = ListedColormap(CLASS_COLORS)
    x0, x1, y0, y1 = VIEW
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.4))
    fig.patch.set_facecolor("#fcfcfb")

    for col, mode in enumerate(MODES):
        mask, res = masks[mode], results[mode]

        top = axes[0, col]
        top.imshow(mask, cmap=cmap, vmin=0, vmax=N_CLASSES, interpolation="nearest")
        top.set_title(f"{mode}\nPCK = {100 * res['pck']:.0f}%",
                      fontsize=12, color=C_INK, pad=8)
        top.set_xticks([])
        top.set_yticks([])

        bot = axes[1, col]
        bot.imshow(mask, cmap=cmap, vmin=0, vmax=N_CLASSES, interpolation="nearest")
        bot.set_xticks([])
        bot.set_yticks([])
        for cid, p, err, ok in res["rows"]:
            gx, gy = gt[cid]
            bot.scatter([gx], [gy], s=180, facecolors="none",
                        edgecolors=C_GT, linewidths=1.8, zorder=4)
            if p is None:
                bot.annotate("NULL", (gx + 12, gy - 12), color=C_PRED,
                             fontsize=9, fontweight="bold", zorder=5)
                continue
            bot.scatter([p[0]], [p[1]], s=90, marker="x",
                        color=C_PRED, linewidths=2.0, zorder=5)
            bot.annotate(f"{err:.1f} px", (p[0] + 12, p[1] + 16),
                         color=C_INK, fontsize=9, zorder=5,
                         bbox=dict(boxstyle="round,pad=0.18", fc="#fcfcfb",
                                   ec="#d5d4d0", lw=0.6))

        bot.set_xlabel(f"detect {100 * res['detection_rate']:.0f}%   "
                       f"acc|det {100 * res['accuracy_given_detection']:.0f}%",
                       fontsize=10, color=C_MUTED)

    # imshow resets the data limits, so crop last.
    for ax in axes.ravel():
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)          # image convention: y increases downward

    axes[0, 0].set_ylabel("segmentation map", fontsize=11, color=C_MUTED)
    axes[1, 0].set_ylabel("o truth   x centroid", fontsize=11, color=C_MUTED)
    fig.suptitle("Centroid of the largest blob, under four segmentation outcomes",
                 fontsize=14, color=C_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    print(f"figure written: {path}")


# --- main ------------------------------------------------------------------
def main():
    print("=" * 84)
    print("DEMO 2 - blob-centroid keypoint localization".center(84))
    print("=" * 84)
    print(f"\nframe {SIZE[1]}x{SIZE[0]} px, disc radius {RADIUS} px, "
          f"tau = {TAU} px, seed {SEED}")

    clean, gt = make_synthetic_mask()
    print("\nground truth")
    print("-" * 84)
    for cid, (x, y) in gt.items():
        print(f"  class {cid}  {CLASS_NAMES[cid]:<20s} ({x:>4d}, {y:>4d})")

    masks, results = {}, {}
    for mode in MODES:
        masks[mode] = degrade_mask(clean, mode, seed=SEED, gt=gt)
        results[mode] = score(localize(masks[mode]), gt)

    print("\nlocalization results")
    print("-" * 84)
    print(f"{'mode':<10s}{'keypoint':<22s}{'predicted (x, y)':>22s}"
          f"{'error px':>12s}{'< tau':>9s}")
    print("-" * 84)
    for mode in MODES:
        for i, (cid, p, err, ok) in enumerate(results[mode]["rows"]):
            label = mode if i == 0 else ""
            if p is None:
                print(f"{label:<10s}{CLASS_NAMES[cid]:<22s}{'NULL RETURN':>22s}"
                      f"{'-':>12s}{'no':>9s}")
            else:
                pos = f"({p[0]:7.2f}, {p[1]:7.2f})"
                print(f"{label:<10s}{CLASS_NAMES[cid]:<22s}{pos:>22s}"
                      f"{err:>12.2f}{('yes' if ok else 'NO'):>9s}")
        r = results[mode]
        print(f"{'':<10s}{'':<22s}{'':>22s}"
              f"{'PCK':>12s}{100 * r['pck']:>8.0f}%")
        print("-" * 84)

    print("\nPCK factorization per mode")
    print("-" * 84)
    print(f"{'mode':<12s}{'N_gt':>6s}{'N_det':>7s}{'N_cor':>7s}"
          f"{'detect':>10s}{'acc|det':>10s}{'PCK':>10s}")
    print("-" * 84)
    for mode in MODES:
        r = results[mode]
        print(f"{mode:<12s}{r['n_gt']:>6d}{r['n_det']:>7d}{r['n_cor']:>7d}"
              f"{100 * r['detection_rate']:>9.1f}%{100 * r['accuracy_given_detection']:>9.1f}%"
              f"{100 * r['pck']:>9.1f}%")
    print("-" * 84)

    # --- 6. tie the failures back to Demo 1 --------------------------------
    miss, split, noise = results["missing"], results["split"], results["noise"]
    print("\nwhat the two failure modes do to the two factors")
    print("-" * 84)
    print(f"  missing : detection {100 * miss['detection_rate']:.0f}% "
          f"(down from 100%), accuracy given detection "
          f"{100 * miss['accuracy_given_detection']:.0f}% (unchanged).")
    print("            Deleting a class costs detection only. The three keypoints")
    print("            still returned are localized exactly as well as before -")
    print("            the detector simply declined to answer for one of them.")
    print(f"  split   : detection {100 * split['detection_rate']:.0f}% (unchanged), "
          f"accuracy given detection {100 * split['accuracy_given_detection']:.0f}%.")
    print("            The fragmented class still returns a coordinate, so nothing")
    print("            is missing - but the largest surviving piece sits down the")
    print("            shaft, and the centroid follows it past tau. This is a")
    print("            confident wrong answer, which is the more dangerous of the two.")
    print(f"  noise   : detection {100 * noise['detection_rate']:.0f}%, "
          f"accuracy given detection {100 * noise['accuracy_given_detection']:.0f}%.")
    print("            Small spurious blobs cost nothing here: largest-component")
    print("            selection discards them, which is why the paper's rule is")
    print("            'largest blob' and not 'centroid of all pixels of the class'.")
    print()
    print("Both 'missing' and 'split' land at PCK = 75%. As in Demo 1, the single")
    print("number does not say which one happened.")

    plot_modes(masks, results, gt, os.path.join(OUT_DIR, "blob_centroid.png"))
    if matplotlib.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
