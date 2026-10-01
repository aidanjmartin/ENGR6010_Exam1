"""
Gaussian and Laplacian pyramids, aliasing, and coarse-to-fine motion.

Reading: Szeliski, "Computer Vision: Algorithms and Applications," 2nd ed.,
2022, Section 9.1.1 on hierarchical motion estimation, with the pyramid
construction it relies on (paper reference [2]).

Data. A 256 x 256 texture: smooth blobs plus fine diagonal stripes at 0.41
cycles per pixel, close to the full-resolution limit of 0.5. Every component
has a whole number of cycles across the image, so the texture tiles perfectly
and the ideal low-pass filter below is exact.

Experiment 1. Build 4-level Gaussian and Laplacian pyramids with cv2.pyrDown
and cv2.pyrUp, then rebuild the image from the Laplacian pyramid alone. Each
Laplacian level stores exactly what the upsampled coarser level misses, so the
reconstruction error should sit at floating-point round-off.

Experiment 2. Downsample by 2, 4 and 8 in two ways: by keeping every n-th
pixel (no blur), and with the Gaussian pyramid (blur, then keep every second
pixel). Compare both with the ideal result: remove every frequency above the
new sampling limit in the Fourier domain, then subsample. Without the blur, the
fine stripes fold back into false coarse stripes (aliasing).

Experiment 3. Coarse-to-fine illustration. Each pyramid level halves the image
and therefore halves the apparent motion, so a 15 px motion becomes 1.875 px at
the coarsest of four levels, small enough for Lucas-Kanade's linear model.

Run:  python gaussian_pyramid.py
Out:  results/gaussian_pyramid_levels.png, results/gaussian_pyramid_aliasing.png
"""

import cv2
import numpy as np

import common as C
import matplotlib.pyplot as plt

N = 256
LEVELS = 4
STRIPE_CYCLES = (96, 40)        # whole cycles across the image in x and y
MOTION_PX = 15.0
CROP = 16                       # border excluded from error measurements


# --- data ------------------------------------------------------------------
def make_texture(rng):
    """Periodic texture: smooth blobs plus fine stripes, scaled to [0, 1]."""
    freq_x = np.fft.fftfreq(N)[None, :]
    freq_y = np.fft.fftfreq(N)[:, None]
    noise = np.fft.fft2(rng.standard_normal((N, N)))
    blobs = np.real(np.fft.ifft2(noise * np.exp(-(freq_x ** 2 + freq_y ** 2)
                                                 * (2 * np.pi * 6.0) ** 2 / 2)))
    blobs /= np.abs(blobs).max()

    yy, xx = np.mgrid[0:N, 0:N]
    kx, ky = STRIPE_CYCLES
    stripes = np.cos(2 * np.pi * (kx * xx + ky * yy) / N)

    img = 0.5 + 0.25 * blobs + 0.2 * stripes
    return img.astype(np.float64)


def stripe_frequency():
    return float(np.hypot(*STRIPE_CYCLES) / N)


# --- pyramids --------------------------------------------------------------
def gaussian_pyramid(img, levels=LEVELS):
    pyr = [img]
    for _ in range(levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def laplacian_pyramid(gauss):
    lap = []
    for fine, coarse in zip(gauss[:-1], gauss[1:]):
        up = cv2.pyrUp(coarse, dstsize=(fine.shape[1], fine.shape[0]))
        lap.append(fine - up)
    lap.append(gauss[-1])
    return lap


def reconstruct(lap):
    img = lap[-1]
    for band in reversed(lap[:-1]):
        img = cv2.pyrUp(img, dstsize=(band.shape[1], band.shape[0])) + band
    return img


# --- aliasing --------------------------------------------------------------
def ideal_downsample(img, factor):
    """Remove all frequencies above the new limit, then keep every n-th pixel."""
    cutoff = 0.5 / factor
    fx = np.abs(np.fft.fftfreq(img.shape[1]))[None, :]
    fy = np.abs(np.fft.fftfreq(img.shape[0]))[:, None]
    keep = (fx < cutoff) & (fy < cutoff)
    low = np.real(np.fft.ifft2(np.fft.fft2(img) * keep))
    return low[::factor, ::factor]


def rmse(a, b, crop):
    c = max(1, crop)
    return float(np.sqrt(np.mean((a[c:-c, c:-c] - b[c:-c, c:-c]) ** 2)))


# --- figures ---------------------------------------------------------------
def plot_levels(gauss, lap, err, name):
    fig, axes = plt.subplots(2, LEVELS, figsize=(14, 7.2))
    for i in range(LEVELS):
        g = gauss[i]
        axes[0, i].imshow(g, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        C.image_axes(axes[0, i], f"Gaussian level {i}   {g.shape[1]} x {g.shape[0]}")
        band = lap[i]
        if i < LEVELS - 1:
            lim = np.abs(band).max()
            axes[1, i].imshow(band, cmap="gray", vmin=-lim, vmax=lim,
                              interpolation="nearest")
            C.image_axes(axes[1, i], f"Laplacian level {i}   (detail band)")
        else:
            axes[1, i].imshow(band, cmap="gray", vmin=0, vmax=1,
                              interpolation="nearest")
            C.image_axes(axes[1, i], f"Laplacian level {i}   (= Gaussian {i})")
    result = ("exactly (maximum error 0 in float64)" if err == 0
              else f"to within {err:.1e}")
    fig.suptitle(f"The Laplacian pyramid rebuilds the image {result}")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0)
    C.save_figure(fig, name)


def plot_aliasing(examples, factors, rows, name):
    fig = plt.figure(figsize=(15, 5.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.35], wspace=0.25)
    titles = ("Every 2nd pixel, no blur", "Gaussian pyramid (cv2.pyrDown)",
              "Ideal low-pass, then subsample")
    for k, (img, title) in enumerate(zip(examples, titles)):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        C.image_axes(ax, title)

    ax = fig.add_subplot(gs[0, 3])
    naive = [r[1] for r in rows]
    blurred = [r[2] for r in rows]
    ax.plot(factors, naive, "-o", color=C.ORANGE, lw=2, ms=8, label="no blur")
    ax.plot(factors, blurred, "-o", color=C.BLUE, lw=2, ms=8,
            label="Gaussian blur")
    for f, a, b in zip(factors, naive, blurred):
        ax.annotate(f"{a:.3f}", (f, a), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=10, color=C.INK)
        ax.annotate(f"{b:.3f}", (f, b), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=10, color=C.INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(factors)
    ax.set_xticklabels([f"{f}x" for f in factors])
    ax.set_xlabel("downsampling factor")
    ax.set_ylabel("RMSE against ideal low-pass")
    ax.set_ylim(0, max(naive) * 1.25)
    ax.set_xlim(factors[0] / 1.3, factors[-1] * 1.3)
    ax.legend(loc="center right")
    ax.set_title("Error against the ideal result")
    C.style_axes(ax)

    fig.suptitle("Without the blur, 2x downsampling folds the fine stripes into "
                 f"false coarse stripes (RMSE {naive[0]:.3f} vs {blurred[0]:.3f})")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.14)
    C.save_figure(fig, name)


# --- main ------------------------------------------------------------------
def main():
    rng = np.random.default_rng(C.SEED)
    img = make_texture(rng)

    gauss = gaussian_pyramid(img)
    lap = laplacian_pyramid(gauss)
    rebuilt = reconstruct(lap)
    max_err = float(np.abs(rebuilt - img).max())
    rms_err = float(np.sqrt(np.mean((rebuilt - img) ** 2)))

    C.print_heading("Laplacian pyramid reconstruction (4 levels, float64)")
    C.print_table(["quantity", "value"],
                  [["image size", f"{N} x {N}"],
                   ["max absolute error", f"{max_err:.2e}"],
                   ["RMS error", f"{rms_err:.2e}"],
                   ["image range", "[0, 1]"]])
    plot_levels(gauss, lap, max_err, "gaussian_pyramid_levels.png")

    factors = [2 ** k for k in range(1, LEVELS)]
    rows = []
    for k, f in enumerate(factors, start=1):
        ideal = ideal_downsample(img, f)
        naive = img[::f, ::f]
        crop = CROP // f
        rows.append((f, rmse(naive, ideal, crop), rmse(gauss[k], ideal, crop)))

    C.print_heading("Aliasing: downsampled image vs ideal low-pass result")
    print(f"Fine stripes at {stripe_frequency():.3f} cycles/px; the limit after "
          "2x downsampling is 0.25 cycles/px.\n")
    C.print_table(["factor", "RMSE, no blur", "RMSE, Gaussian pyramid",
                   "ratio"],
                  [[f"{f}x", f"{a:.4f}", f"{b:.4f}", f"{a / b:.1f}x"]
                   for f, a, b in rows])

    examples = (img[::2, ::2], gauss[1], ideal_downsample(img, 2))
    plot_aliasing(examples, factors, rows, "gaussian_pyramid_aliasing.png")

    C.print_heading(f"Coarse-to-fine: a {MOTION_PX:g} px motion at each level")
    C.print_table(["level", "image size", "motion (px)"],
                  [[lvl, f"{gauss[lvl].shape[1]} x {gauss[lvl].shape[0]}",
                    f"{MOTION_PX / 2 ** lvl:g}"] for lvl in range(LEVELS)])
    print("Lucas-Kanade linearizes the image around the current estimate, which "
          "holds for about 1-2 px of motion. Only the coarsest level brings "
          "15 px inside that range.")


if __name__ == "__main__":
    main()
