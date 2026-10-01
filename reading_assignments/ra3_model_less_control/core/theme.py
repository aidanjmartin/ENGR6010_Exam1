"""Palette, type scale and spacing. Every number here is in virtual pixels."""

VW, VH = 1920, 1080
MARGIN = 40            # nothing registered may come closer than this to an edge
GUTTER = 96            # standard left/right content inset

# ---- palette ------------------------------------------------------------------------
BG = (11, 13, 20)
BG_PANEL = (16, 19, 29)
PANEL_EDGE = (38, 44, 62)
GRID = (26, 30, 44)
GRID_MAJOR = (36, 42, 60)
TEXT = (232, 236, 243)
TEXT_DIM = (150, 159, 180)
TEXT_FAINT = (98, 107, 128)

# Meanings. Each colour is always paired with a line style or a label as well.
GRADIENT = (247, 196, 72)     # first order: gradient descent path, gradient arrows
CURVATURE = (88, 176, 240)    # second order: Newton path, Hessian, true contours
ESTIMATE = (64, 216, 168)     # estimates: BFGS ellipse, J-hat, model-less robot
TRUTH = (238, 232, 216)       # ground truth: drawn dashed
MODEL = (236, 112, 184)       # assumed model: the model-based robot
WARNING = (255, 94, 74)       # error growing, saturation, infeasible (with an icon)

# Roles used inside equations and diagrams.
STEP = (172, 146, 255)        # what you did: s, dy
OBSERVED = TRUTH              # what you saw: y, dx (a measurement of the truth)
TENSION = (178, 190, 212)
TARGET = (246, 246, 250)

# Jacobian columns keep one hue family per source: insertion, left, right.
ESTIMATE_COLS = [(160, 244, 212), (64, 216, 168), (52, 170, 204)]
MODEL_COLS = [(252, 176, 216), (236, 112, 184), (186, 110, 238)]
TRUTH_COLS = [(250, 247, 238), (232, 222, 196), (206, 198, 178)]
COL_NAMES = ["insertion", "left tendon", "right tendon"]
COL_SHORT = ["ins", "L", "R"]

# ---- type scale ---------------------------------------------------------------------
TITLE = 62
SUBTITLE = 36
HEADING = 44
CAPTION = 36
BODY = 32
LABEL = 28
SMALL = 24

MATH_BIG = 48
MATH = 42
MATH_SMALL = 36

# ---- standard regions -----------------------------------------------------------------
HEADER_Y = 56                 # top of act titles
CAPTION_CY = 990              # vertical centre of the caption band
CONTENT_TOP = 150
CONTENT_BOTTOM = 930


def mix(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def dim(c, t):
    """Blend a colour toward the background."""
    return mix(BG, c, t)
