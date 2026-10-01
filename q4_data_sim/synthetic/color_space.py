"""
Color-space segmentation under changing illumination.

Reading: Ibraheem et al., "Understanding color models: A review," ARPN J. Sci.
Tech., 2012 (paper reference [1]).

The review argues that RGB entangles brightness with color, while HSV and
CIELAB separate them into one intensity channel (V, L*) and two color channels.
This script tests that claim on an image with exact ground truth.

Experiment 1. An orange disc sits on a mildly textured blue-gray background
with red and olive clutter: colors whose RGB values fall between those of the
lit and the shadowed disc, but at a different hue. The scene sits under a
left-to-right illumination gradient. A second copy adds a shadow band
across part of the disc. Each color space gets a box threshold (a lower and an
upper bound per channel), tuned on the unshadowed image against the true mask
and then applied unchanged to the shadowed image:

    RGB    thresholds R, G and B (no channel isolates brightness)
    HSV    thresholds H and S, ignores V
    CIELAB thresholds a* and b*, ignores L*

The score is intersection over union (IoU) with the true disc mask. When
several thresholds fit the unshadowed image equally well, tuning keeps the
widest one, so each color space gets its most tolerant accurate threshold.

Experiment 2. Hue is the angle of a color around the gray axis, so it is
undefined for a gray pixel. A ramp from gray to orange at fixed value, with a
small amount of sensor noise, shows the hue standard deviation growing without
bound as saturation goes to zero.

Run:  python color_space.py
Out:  results/color_space_segmentation.png, results/color_space_hue_instability.png
"""

import cv2
import numpy as np

import common as C
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

H, W = 240, 320
DISC_CENTER = (170, 120)        # (x, y)
DISC_RADIUS = 72
ORANGE_RGB = (0.90, 0.47, 0.12)
BACKGROUND_RGB = (0.36, 0.42, 0.50)
# Clutter ellipses: (center x, center y, half-width, half-height, RGB).
CLUTTER = (
    (40, 55, 30, 38, (0.78, 0.14, 0.10)),      # red
    (290, 200, 26, 30, (0.40, 0.46, 0.07)),    # olive
    (48, 195, 34, 30, (0.36, 0.42, 0.06)),     # olive
    (288, 42, 24, 30, (0.70, 0.12, 0.10)),     # red
)
ILLUMINATION = (0.60, 1.05)     # gain at the left and right edges
SHADOW_GAIN = 0.40              # light that reaches the shadow band
NOISE_SIGMA = 0.01              # sensor noise, on a [0, 1] scale

SPACES = {
    "RGB": {"channels": (0, 1, 2), "names": "R, G, B"},
    "HSV": {"channels": (0, 1), "names": "H, S"},
    "CIELAB": {"channels": (1, 2), "names": "a*, b*"},
}
PERCENTILES = (0.5, 1, 2, 5, 10)
MARGINS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
IOU_TOLERANCE = 0.005           # accept any box this close to the best IoU


# --- synthetic scene -------------------------------------------------------
def smooth_noise(rng, sigma):
    """Zero-mean, unit-peak band-limited noise: a mild surface texture."""
    n = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32),
                         (0, 0), sigma)
    return n / np.abs(n).max()


def make_scene(rng):
    """Return (unshadowed, shadowed, true_mask, shadow_gain_map)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    mask = (xx - DISC_CENTER[0]) ** 2 + (yy - DISC_CENTER[1]) ** 2 <= DISC_RADIUS ** 2

    texture = 1.0 + 0.08 * smooth_noise(rng, 3.0)
    disc_texture = 1.0 + 0.04 * smooth_noise(rng, 2.0)
    reflectance = np.where(mask[..., None],
                           np.array(ORANGE_RGB) * disc_texture[..., None],
                           np.array(BACKGROUND_RGB) * texture[..., None])
    for cx, cy, ax, ay, rgb in CLUTTER:
        blob = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 <= 1.0
        reflectance[blob] = np.array(rgb) * texture[blob][:, None]

    lo, hi = ILLUMINATION
    gradient = lo + (hi - lo) * xx / (W - 1)

    # A diagonal band with soft edges, crossing the lower-left part of the disc.
    d = (xx * np.cos(0.6) + yy * np.sin(0.6)) - 210.0
    band = (np.abs(d) < 28).astype(np.float32)
    band = cv2.GaussianBlur(band, (0, 0), 4.0)
    shadow = 1.0 - (1.0 - SHADOW_GAIN) * band

    def expose(light):
        img = reflectance * light[..., None]
        img = img + rng.normal(0, NOISE_SIGMA, img.shape)
        return np.clip(img, 0, 1).astype(np.float32)

    return expose(gradient), expose(gradient * shadow), mask, shadow


def convert(img_rgb, space):
    if space == "RGB":
        return img_rgb
    if space == "HSV":
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)      # H in degrees
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)          # L* 0-100, a*, b*


# --- threshold tuning ------------------------------------------------------
def apply_box(img, channels, lo, hi):
    keep = np.ones(img.shape[:2], bool)
    for c, a, b in zip(channels, lo, hi):
        keep &= (img[..., c] >= a) & (img[..., c] <= b)
    return keep


def tune_box(img, mask, channels):
    """Tune a box threshold on one image against its true mask.

    Every percentile and margin combination that reaches within IOU_TOLERANCE
    of the best IoU counts as accurate. Several do, because the unshadowed disc
    separates cleanly, so the tie goes to the widest box: the most tolerant
    threshold the unshadowed image can justify. Width is the product of each
    bound's span relative to that channel's span over the whole image.
    """
    vals = img[mask][:, channels]
    full = img.reshape(-1, img.shape[-1])[:, channels]
    span = full.max(axis=0) - full.min(axis=0)
    candidates = []
    for p in PERCENTILES:
        lo0 = np.percentile(vals, p, axis=0)
        hi0 = np.percentile(vals, 100 - p, axis=0)
        for m in MARGINS:
            pad = m * (hi0 - lo0)
            lo, hi = lo0 - pad, hi0 + pad
            score = C.iou(apply_box(img, channels, lo, hi), mask)
            width = float(np.prod((hi - lo) / span))
            candidates.append((score, width, lo, hi, p, m))
    best = max(c[0] for c in candidates)
    accurate = [c for c in candidates if c[0] >= best - IOU_TOLERANCE]
    score, _, lo, hi, p, m = max(accurate, key=lambda c: c[1])
    return score, lo, hi, p, m


# --- hue instability -------------------------------------------------------
def hue_spread(rng, saturations, value=0.6, hue=30.0, n=4000):
    """Circular standard deviation of measured hue at each nominal saturation."""
    out = []
    for s in saturations:
        hsv = np.tile(np.array([hue, s, value], np.float32), (n, 1, 1))
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        rgb = np.clip(rgb + rng.normal(0, NOISE_SIGMA, rgb.shape), 0, 1)
        h = np.deg2rad(cv2.cvtColor(rgb.astype(np.float32),
                                    cv2.COLOR_RGB2HSV)[..., 0].ravel())
        r = np.abs(np.mean(np.exp(1j * h)))
        out.append(np.rad2deg(np.sqrt(-2.0 * np.log(max(r, 1e-12)))))
    return np.array(out)


# --- figures ---------------------------------------------------------------
TP, FN, FP = "#c9c8c3", C.BLUE, C.ORANGE


def error_map(pred, truth):
    rgb = np.ones(truth.shape + (3,))
    for m, color in ((pred & truth, TP), (truth & ~pred, FN), (pred & ~truth, FP)):
        rgb[m] = plt.matplotlib.colors.to_rgb(color)
    return rgb


def plot_segmentation(images, masks, truth, scores, path_name):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.4))
    rows = ("Unshadowed", "Shadowed")
    for r, cond in enumerate(rows):
        ax = axes[r, 0]
        ax.imshow(images[cond])
        C.image_axes(ax, f"{cond} input")
        for c, space in enumerate(SPACES, start=1):
            ax = axes[r, c]
            ax.imshow(error_map(masks[cond][space], truth), interpolation="nearest")
            C.image_axes(ax, f"{space}   IoU {scores[cond][space]:.3f}")
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_color(C.AXIS)
    handles = [Patch(color=TP, label="disc, found"),
               Patch(color=FN, label="disc, missed"),
               Patch(color=FP, label="background, wrongly included")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11)

    sh = scores["Shadowed"]
    fig.suptitle(f"Under a shadow, HSV keeps IoU {sh['HSV']:.2f}; RGB falls to "
                 f"{sh['RGB']:.2f} and CIELAB to {sh['CIELAB']:.2f}")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97), h_pad=2.5)
    C.save_figure(fig, path_name)


def plot_hue(saturations, spread, ramp, path_name):
    fig, (ax_img, ax) = plt.subplots(
        2, 1, figsize=(8, 6.2), gridspec_kw={"height_ratios": [1, 5]})
    ax_img.imshow(ramp, aspect="auto",
                  extent=(saturations[0], saturations[-1], 0, 1))
    ax_img.set_yticks([])
    ax_img.set_xticks([])
    ax_img.set_title("Test ramp: gray (S = 0) to orange (S = 0.5), V = 0.6, "
                     f"noise sigma = {NOISE_SIGMA}", fontsize=10.5, color=C.MUTED)

    ax.plot(saturations, spread, "-o", color=C.BLUE, lw=2, ms=6)
    ax.set_xlim(ax_img.get_xlim())
    ax.set_xlabel("saturation S")
    ax.set_ylabel("hue standard deviation (degrees)")
    for s, dx, ha in ((0.0, 10, "left"), (0.02, 10, "left"),
                      (0.1, 10, "left"), (0.5, 0, "right")):
        i = int(np.argmin(np.abs(saturations - s)))
        ax.annotate(f"{spread[i]:.1f} deg", (saturations[i], spread[i]),
                    textcoords="offset points", xytext=(dx, 10), ha=ha,
                    fontsize=10, color=C.INK)
    C.style_axes(ax, grid_axis="both")
    fig.suptitle("Hue becomes undefined as saturation goes to zero")
    fig.tight_layout()
    C.save_figure(fig, path_name)


# --- main ------------------------------------------------------------------
def main():
    rng = np.random.default_rng(C.SEED)
    lit, shaded, truth, _ = make_scene(rng)
    images = {"Unshadowed": lit, "Shadowed": shaded}

    C.print_heading("Color-space thresholds, tuned on the unshadowed image")
    tuned, rows = {}, []
    for space, spec in SPACES.items():
        img = convert(lit, space)
        score, lo, hi, p, m = tune_box(img, truth, spec["channels"])
        tuned[space] = (lo, hi)
        bounds = ", ".join(f"[{a:.2f}, {b:.2f}]" for a, b in zip(lo, hi))
        rows.append([space, spec["names"], bounds, f"{p:g}", f"{m:g}"])
    C.print_table(["color space", "channels", "bounds", "percentile", "margin"],
                  rows, align=["l", "l", "l", "r", "r"])

    masks = {cond: {} for cond in images}
    scores = {cond: {} for cond in images}
    for cond, img_rgb in images.items():
        for space, spec in SPACES.items():
            lo, hi = tuned[space]
            pred = apply_box(convert(img_rgb, space), spec["channels"], lo, hi)
            masks[cond][space] = pred
            scores[cond][space] = C.iou(pred, truth)

    C.print_heading("IoU against the true disc mask")
    C.print_table(
        ["color space", "unshadowed IoU", "shadowed IoU", "change"],
        [[s, f"{scores['Unshadowed'][s]:.3f}", f"{scores['Shadowed'][s]:.3f}",
          f"{scores['Shadowed'][s] - scores['Unshadowed'][s]:+.3f}"]
         for s in SPACES])
    print("HSV hue and saturation do not change when all three channels scale "
          "together, so the shadow leaves them in place. CIELAB a* and b* shrink "
          "toward zero as L* falls, so the shadowed disc leaves the a*, b* box.")

    plot_segmentation(images, masks, truth, scores,
                      "color_space_segmentation.png")

    saturations = np.linspace(0.0, 0.5, 26)
    spread = hue_spread(rng, saturations)
    C.print_heading("Hue spread on a noisy gray-to-orange ramp")
    C.print_table(["saturation", "hue std (deg)"],
                  [[f"{s:.2f}", f"{v:.1f}"] for s, v in
                   zip(saturations[::5], spread[::5])])

    ramp_hsv = np.zeros((40, 400, 3), np.float32)
    ramp_hsv[..., 0] = 30.0
    ramp_hsv[..., 1] = np.linspace(0, 0.5, 400)[None, :]
    ramp_hsv[..., 2] = 0.6
    ramp = cv2.cvtColor(ramp_hsv, cv2.COLOR_HSV2RGB)
    ramp = np.clip(ramp + rng.normal(0, NOISE_SIGMA, ramp.shape), 0, 1)
    plot_hue(saturations, spread, ramp, "color_space_hue_instability.png")


if __name__ == "__main__":
    main()
