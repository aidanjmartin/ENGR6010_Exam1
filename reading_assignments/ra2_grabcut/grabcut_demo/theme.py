"""
Shared visual language for the GrabCut demo.

Every screen in both parts of the demo pulls its colors, fonts and axis styling
from here, so the math walkthrough and the live segmentation tool read as one
piece of software. Two accents only:

    BLUE  foreground / primary structure
    GOLD  the thing you are supposed to be looking at right now

Everything else is a neutral ramp from the near-black background up to off-white
text. The background class is deliberately drawn in a *neutral* (SLATE), not a
third accent, so "foreground" is always the only saturated thing on screen.
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --- palette ---------------------------------------------------------------
BG        = "#0E1117"   # page background, near-black with a blue cast
PANEL     = "#161B22"   # raised panel fill
PANEL_EDGE= "#232A35"   # hairline panel border
GRID      = "#252C38"   # gridlines, axis rules

TEXT      = "#E6EDF3"   # primary type
MUTED     = "#8B949E"   # secondary type, captions
DIM       = "#565E68"   # tertiary / not-yet-revealed type

BLUE      = "#4CA6FF"   # accent 1 - foreground class, primary structure
BLUE_DEEP = "#1F5FA8"   # blue, pushed back
GOLD      = "#FFC857"   # accent 2 - active highlight, the min-cut
GOLD_DEEP = "#8A6A20"   # gold, pushed back

SLATE     = "#7D8899"   # background class (neutral, not an accent)
SLATE_DEEP= "#3C444F"

FG, BGC = BLUE, SLATE   # semantic aliases used by the segmentation screens

# --- typography ------------------------------------------------------------
_SANS_PREFS = ["Helvetica Neue", "Helvetica", "Avenir Next", "Arial", "DejaVu Sans"]


def _pick_sans():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in _SANS_PREFS:
        if name in have:
            return name
    return "DejaVu Sans"


SANS = _pick_sans()

FIGSIZE = (16, 9)       # 16:9 for a projector
DPI = 100


def apply_theme():
    """Install the demo's look as matplotlib defaults."""
    plt.rcdefaults()
    matplotlib.rcParams.update({
        "figure.facecolor":  BG,
        "figure.edgecolor":  BG,
        "savefig.facecolor": BG,
        "savefig.edgecolor": BG,
        "axes.facecolor":    BG,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   MUTED,
        "text.color":        TEXT,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "grid.color":        GRID,
        "font.family":       "sans-serif",
        "font.sans-serif":   [SANS, "DejaVu Sans"],
        "font.size":         13,
        "mathtext.fontset":  "cm",       # real Computer Modern for equations
        "axes.titlesize":    17,
        "axes.labelsize":    12,
        "figure.dpi":        DPI,
        "savefig.dpi":       DPI,
        "toolbar":           "None",     # no matplotlib chrome
        "keymap.save":       [],         # free up 's', 'f', 'g', etc. for us
        "keymap.fullscreen": [],
        "keymap.grid":       [],
        "keymap.home":       [],
        "keymap.back":       [],
        "keymap.forward":    [],
        "keymap.pan":        [],
        "keymap.zoom":       [],
        "keymap.quit":       ["ctrl+w", "cmd+w"],
        "keymap.xscale":     [],
        "keymap.yscale":     [],
    })


# --- axis helpers ----------------------------------------------------------
def blank(ax, keep_facecolor=False):
    """Strip every piece of default matplotlib chrome from an axes."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if not keep_facecolor:
        ax.set_facecolor("none")
    ax.set_navigate(False)
    return ax


def panel(ax, title=None, pad=0.012):
    """A raised card: subtle fill, hairline border, optional small-caps title."""
    blank(ax)
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(PANEL_EDGE)
        s.set_linewidth(1.0)
    if title:
        ax.set_title(title.upper(), color=MUTED, fontsize=10.5, pad=10,
                     loc="left", fontweight="medium")
        try:
            ax.title.set_fontstretch("semi-expanded")
        except Exception:
            pass
    return ax


def chart(ax, title=None):
    """A minimal plot frame: left+bottom rules only, horizontal gridlines."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("none")
    for name, s in ax.spines.items():
        on = name in ("left", "bottom")
        s.set_visible(on)
        s.set_color(GRID)
        s.set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=10, length=3, width=0.8)
    ax.grid(True, axis="y", color=GRID, lw=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title.upper(), color=MUTED, fontsize=10.5, pad=10, loc="left")
    return ax


# --- easing ----------------------------------------------------------------
def smooth(t):
    """Cubic smoothstep. 3b1b's default easing; no visible start/stop jerk."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def smoother(t):
    """Quintic smoothstep - gentler still, for long camera-like moves."""
    t = min(max(t, 0.0), 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def lerp(a, b, t):
    return a + (b - a) * t


def stagger(t, i, n, overlap=0.65):
    """
    Progress of item `i` of `n` within an overall progress `t`.

    Items start in sequence but their animations overlap, which is what makes a
    list of things appear to *flow* in rather than tick in one at a time.
    """
    if n <= 1:
        return smooth(t)
    span = 1.0 / (1.0 + (n - 1) * (1.0 - overlap))
    start = i * span * (1.0 - overlap)
    return smooth((t - start) / span) if span > 0 else smooth(t)


def fade(artists, alpha):
    for a in artists:
        try:
            a.set_alpha(alpha)
        except Exception:
            pass
