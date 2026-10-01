"""
Shared helpers for the five Q4 synthetic-data scripts.

Holds the fixed random seed, the output paths, the figure style, the import
hooks that load the reading-assignment code unchanged from
`reading_assignments/`, and small helpers for intersection over union (IoU) and
Markdown results tables. `run_all.py` copies every printed table into
`q4_data_sim/results/SUMMARY.md`, so all tables print as Markdown.

Import this module before matplotlib.pyplot: it selects the non-interactive Agg
backend so that every script runs headless.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# --- reproducibility --------------------------------------------------------
SEED = 6010

# --- paths ------------------------------------------------------------------
SYNTHETIC_DIR = os.path.dirname(os.path.abspath(__file__))
Q4_DIR = os.path.dirname(SYNTHETIC_DIR)
REPO_ROOT = os.path.dirname(Q4_DIR)
RESULTS_DIR = os.path.join(Q4_DIR, "results")
RA1_DIR = os.path.join(REPO_ROOT, "reading_assignments", "ra1_optical_flow")
RA2_DIR = os.path.join(REPO_ROOT, "reading_assignments", "ra2_grabcut")


def use_reading_assignment(path):
    """Make a reading-assignment folder importable without copying its code."""
    if path not in sys.path:
        sys.path.insert(0, path)


# --- figure style -----------------------------------------------------------
# Categorical slots 1-5 of the validated reference palette, in fixed order.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e5e4e0"
AXIS = "#d5d4d0"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titlecolor": INK,
    "axes.labelcolor": MUTED,
    "axes.labelsize": 11,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "figure.titlesize": 14,
})


def style_axes(ax, grid_axis="y"):
    """Recessive grid and axes: data first, scaffolding second."""
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(length=0)


def image_axes(ax, title=None):
    """An axes that shows an image: no ticks, no frame."""
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    if title:
        ax.set_title(title, fontsize=11)


def save_figure(fig, name):
    """Save a figure to q4_data_sim/results/ at 150 dpi and close it."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved: `results/{name}`")
    return path


# --- metrics ----------------------------------------------------------------
def iou(pred, truth):
    """Intersection over union of two boolean masks."""
    pred = np.asarray(pred, bool)
    truth = np.asarray(truth, bool)
    union = np.logical_or(pred, truth).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, truth).sum() / union)


# --- printing ---------------------------------------------------------------
def print_heading(text):
    print(f"\n### {text}\n")


def print_table(headers, rows, align=None):
    """Print a Markdown table. Cells are printed exactly as given."""
    align = align or ["l"] + ["r"] * (len(headers) - 1)
    marks = {"l": ":---", "r": "---:", "c": ":---:"}
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(marks[a] for a in align) + " |")
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")
    print()
