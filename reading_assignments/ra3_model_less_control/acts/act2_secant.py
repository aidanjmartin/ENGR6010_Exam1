"""Act 2: estimate what you cannot compute. Secants, BFGS, and Broyden's update."""
from __future__ import annotations

import math

import numpy as np
import pygame

from acts.act1_curvature import PLOT, PathAnim, draw_plot_frame, to_px
from acts.base import Act
from core import draw, text, theme
from core.anim import clamp, ease_in_out, ease_out, fade, lerp, seg
from core.mathtext import Eq, pulse
from core.widgets import legend
from sim import optim

RIGHT_X = 1010
GAMMA = 25.0

# ---- beat 1: a 1-D function and its secant --------------------------------------------
CURVE = pygame.Rect(150, 160, 1060, 740)
XR = (-0.2, 4.3)
YR = (0.4, 4.9)
XA = 1.0          # the first observation stays put
XB0 = 3.8         # the second one slides in toward it ...
XB1 = 1.42        # ... and stops close enough that the tangent is almost reached


def g_fn(x):
    return 1.2 + 0.35 * (x - 1) + 0.28 * (x - 1) ** 2


def g_slope(x):
    return 0.35 + 0.56 * (x - 1)


def c_px(x, y):
    return (CURVE.left + (x - XR[0]) / (XR[1] - XR[0]) * CURVE.w,
            CURVE.bottom - (y - YR[0]) / (YR[1] - YR[0]) * CURVE.h)


def ellipse_matrix(B, radius):
    """Pixel-space matrix whose image of the unit circle is {d : d'Bd = const},
    scaled so the longest semi-axis is `radius` pixels."""
    w, V = np.linalg.eigh(B)
    axes = 1.0 / np.sqrt(np.maximum(w, 1e-9))
    axes = axes / axes.max() * radius
    M = V @ np.diag(axes)
    return np.array([[1, 0], [0, -1]]) @ M      # flip y for screen space


class SecantAct(Act):
    number = 2
    title = "Estimate what you cannot compute"
    captions = [
        "Two observations give you a slope estimate.",
        "Adjust the estimate so it explains what you just observed.",
        "The smallest change that explains the newest measurement.",
    ]
    settle = 9.0

    # BFGS timeline (seconds from beat start)
    T_SHOW = 0.5
    STEP_T = [(1.2, 1.0), (4.3, 1.0)]     # (start, duration) of each move
    MORPH_T = [(2.9, 1.1), (6.0, 1.1)]    # estimate update after each move

    def __init__(self):
        super().__init__()
        self.eq_slope = Eq([r"\mathrm{slope} \;\approx\; \frac{", (r"\Delta g", "dg"), r"}{", (r"\Delta x", "dx"), r"}"],
                           size=theme.MATH_BIG + 6, colors={"dg": theme.OBSERVED, "dx": theme.STEP})
        self.eq_secant = Eq([(r"B", "B"), r"\,", (r"s", "s"), r"\;=\;", (r"y", "y")], size=theme.MATH_BIG + 16,
                            colors={"B": theme.ESTIMATE, "s": theme.STEP, "y": theme.OBSERVED})
        self.eq_row1 = Eq([(r"B_{\mathrm{new}}", "B"), r"\,", (r"s", "s"), r"\;=\;", (r"y", "y")],
                          size=theme.MATH_BIG + 4, colors={"B": theme.ESTIMATE, "s": theme.STEP, "y": theme.OBSERVED})
        self.eq_row2 = Eq([(r"\hat{J}_{\mathrm{new}}", "J"), r"\,", (r"\Delta y", "dy"), r"\;=\;", (r"\Delta x", "dx")],
                          size=theme.MATH_BIG + 4, colors={"J": theme.ESTIMATE, "dy": theme.STEP, "dx": theme.OBSERVED})
        self.eq_broyden = Eq([(r"\hat{J}_{\mathrm{new}}", "J"), r"\;=\;", (r"\hat{J}", "J0"), r"\;+\;\frac{",
                              r"(", (r"\Delta x", "edx"), r"-", (r"\hat{J}\,\Delta y", "eJ"), r")\;",
                              (r"\Delta y^{\mathsf{T}}", "dyT"),
                              r"}{", (r"\Delta y^{\mathsf{T}}\Delta y", "den"), r"}"],
                             size=theme.MATH_BIG + 6,
                             colors={"J": theme.ESTIMATE, "J0": theme.ESTIMATE, "edx": theme.OBSERVED,
                                     "eJ": theme.ESTIMATE, "dyT": theme.STEP, "den": theme.STEP})
        self.lbl_x1 = Eq([r"x_1"], size=theme.MATH_SMALL, color=theme.TEXT_DIM)
        self.lbl_x2 = Eq([r"x_2"], size=theme.MATH_SMALL, color=theme.TEXT_DIM)
        self.lbl_gx = Eq([r"g(x)"], size=theme.MATH, color=theme.TRUTH)
        self.lbl_x = Eq([r"x"], size=theme.MATH_SMALL, color=theme.TEXT_DIM)
        self.lbl_s = Eq([r"s"], size=theme.MATH, color=theme.STEP)
        self.lbl_y = Eq([r"y"], size=theme.MATH, color=theme.OBSERVED)
        self.lbl_dx = Eq([r"\Delta x"], size=theme.MATH, color=theme.STEP)
        self.lbl_dg = Eq([r"\Delta g"], size=theme.MATH, color=theme.OBSERVED)
        self.lbl_pred = Eq([r"\hat{J}\,\Delta y"], size=theme.MATH, color=theme.ESTIMATE)
        self.lbl_meas = Eq([r"\Delta x"], size=theme.MATH, color=theme.OBSERVED)
        path, Bs, ss, ys = optim.bfgs(GAMMA)
        self.bfgs_path, self.Bs, self.ss, self.ys = path, Bs, ss, ys
        self.gd = PathAnim(optim.gradient_descent(GAMMA))

    def equations(self):
        return [self.eq_slope, self.eq_secant, self.eq_row1, self.eq_row2, self.eq_broyden,
                self.lbl_x1, self.lbl_x2, self.lbl_gx, self.lbl_x, self.lbl_s, self.lbl_y,
                self.lbl_dx, self.lbl_dg, self.lbl_pred, self.lbl_meas]

    def builds_in(self, beat):
        return 2 if beat == 2 else 0

    def handle(self, ev):
        return False

    # ---- drawing ---------------------------------------------------------------------------
    def draw(self, c):
        self.draw_header(c)
        [self._draw_secant, self._draw_bfgs, self._draw_compare][self.beat](c)
        self.draw_caption(c)

    # beat 1 -------------------------------------------------------------------------------
    def _draw_secant(self, c):
        t = self.t
        pygame.draw.rect(c, theme.BG_PANEL, CURVE, border_radius=4)
        old = c.get_clip()
        c.set_clip(CURVE)
        for gx in np.arange(0, 4.5, 0.5):
            px = c_px(gx, 0)[0]
            pygame.draw.line(c, theme.GRID, (px, CURVE.top), (px, CURVE.bottom), 1)
        for gy in np.arange(0.5, 5, 0.5):
            py = c_px(0, gy)[1]
            pygame.draw.line(c, theme.GRID, (CURVE.left, py), (CURVE.right, py), 1)
        xs = np.linspace(XR[0], XR[1], 200)
        pts = [c_px(x, g_fn(x)) for x in xs]
        n = max(2, int(len(pts) * seg(t, 0.0, 0.9, ease_out)))
        draw.polyline(c, theme.TRUTH, pts[:n], 4)

        xb = lerp(XB0, XB1, seg(t, 1.6, 3.2))
        A = (XA, g_fn(XA))
        B = (xb, g_fn(xb))
        slope = (B[1] - A[1]) / (B[0] - A[0])
        pa = fade(t, 0.8)
        # tangent at A (the truth), dashed
        ta = fade(t, 0.9)
        if ta > 0:
            m = g_slope(XA)
            draw.dashed_line(c, theme.dim(theme.TRUTH, 0.7 * ta), c_px(XR[0], A[1] + m * (XR[0] - A[0])),
                             c_px(XR[1], A[1] + m * (XR[1] - A[0])), 3, 16, 12)
        # secant through A and B, the estimate
        if pa > 0:
            x0, x1 = XR[0], XR[1]
            draw.line(c, theme.dim(theme.ESTIMATE, pa), c_px(x0, A[1] + slope * (x0 - A[0])),
                      c_px(x1, A[1] + slope * (x1 - A[0])), 4)
            corner = c_px(B[0], A[1])
            span = abs(c_px(B[0], 0)[0] - c_px(A[0], 0)[0])
            la = pa * clamp((span - 30) / 60)
            draw.line(c, theme.dim(theme.STEP, pa), c_px(*A), corner, 5)
            draw.line(c, theme.dim(theme.OBSERVED, pa), corner, c_px(*B), 5)
            if la > 0:
                self.lbl_dx.draw(c, ((c_px(*A)[0] + corner[0]) / 2, corner[1] + 12), "midtop", la, register=False)
                self.lbl_dg.draw(c, (corner[0] + 22, (corner[1] + c_px(*B)[1]) / 2 + 14), "midleft", la,
                                 register=False)
            for P in (A, B):
                draw.glow(c, theme.TRUTH, c_px(*P), 26, 0.5 * pa)
                draw.circle(c, theme.dim(theme.TRUTH, pa), c_px(*P), 10)
                draw.circle(c, theme.BG_PANEL, c_px(*P), 4)
        c.set_clip(old)
        pygame.draw.rect(c, theme.PANEL_EDGE, CURVE, 2, border_radius=4)
        gp = c_px(2.9, g_fn(2.9))
        self.lbl_gx.draw(c, (gp[0] - 22, gp[1] - 10), "bottomright", fade(t, 0.5), register=False)
        self.lbl_x.draw(c, (CURVE.right - 14, CURVE.bottom - 12), "bottomright", register=False)

        x = CURVE.right + 70
        self.eq_slope.draw(c, (x, 300), "midleft", fade(t, 0.9))
        va = fade(t, 1.0)
        text.blit(c, "estimate", theme.LABEL, theme.TEXT_DIM, (x, 430), "midleft", va, name="est-lbl")
        text.blit(c, f"{slope:.2f}", theme.HEADING + 6, theme.ESTIMATE, (x + 520, 430), "midright", va, name="est")
        text.blit(c, "true slope", theme.LABEL, theme.TEXT_DIM, (x, 500), "midleft", va, name="true-lbl")
        text.blit(c, f"{g_slope(XA):.2f}", theme.HEADING + 6, theme.TRUTH, (x + 520, 500), "midright", va, name="true")
        legend(c, [("secant: the estimate", theme.ESTIMATE, "line"),
                   ("tangent: the truth", theme.TRUTH, "dash"),
                   ("what you changed", theme.STEP, "line"),
                   ("what you observed", theme.OBSERVED, "line")], (x + 4, 640), fade(t, 0.6))

    # beat 2 -------------------------------------------------------------------------------
    def _bfgs_state(self, t):
        """(position, B estimate, index of last completed step, morph progress)."""
        P, Bs = self.bfgs_path, self.Bs
        pos, B, k = P[0], Bs[0], 0
        for i, ((s0, d), (m0, md)) in enumerate(zip(self.STEP_T, self.MORPH_T)):
            if i + 1 >= len(P):
                break
            f = seg(t, s0, d)
            if f <= 0:
                break
            pos = P[i] + f * (P[i + 1] - P[i])
            mf = seg(t, m0, md)
            B = Bs[i] + mf * (Bs[i + 1] - Bs[i])
            k = i + 1 if f >= 1 else i
        return np.asarray(pos), B, k

    def _draw_bfgs(self, c):
        t = self.t
        draw_plot_frame(c, GAMMA, self.lbl_x1, self.lbl_x2)
        old = c.get_clip()
        c.set_clip(PLOT.inflate(-4, -4))
        # gradient descent ghost for comparison
        gp = [to_px(p) for p in self.gd.path]
        ga = 0.55 * fade(t, 0.2)
        for a, b in zip(gp[:-1], gp[1:]):
            draw.line(c, theme.mix(theme.BG_PANEL, theme.GRADIENT, ga), a, b, 2)
        pos, B, k = self._bfgs_state(t)
        # completed BFGS segments
        for i in range(len(self.bfgs_path) - 1):
            s0, d = self.STEP_T[i] if i < len(self.STEP_T) else (1e9, 1)
            f = seg(t, s0, d)
            if f <= 0:
                continue
            a = np.array(to_px(self.bfgs_path[i]))
            b = np.array(to_px(self.bfgs_path[i] + f * (self.bfgs_path[i + 1] - self.bfgs_path[i])))
            draw.arrow(c, theme.STEP, a, b, 5, 22) if f > 0.15 else draw.line(c, theme.STEP, a, b, 5)
            if f >= 1:
                mid = (a + b) / 2
                nrm = np.array([-(b - a)[1], (b - a)[0]])
                nrm = nrm / max(np.linalg.norm(nrm), 1e-9)
                if nrm[1] > 0:
                    nrm = -nrm
                self.lbl_s.draw(c, mid + nrm * 30, "center", fade(t, s0 + d), register=False)
            # observed gradient change y at the new iterate
            if f >= 1:
                ya = fade(t, s0 + d + 0.1)
                yv = self.ys[i]
                L = np.linalg.norm(b - a)
                v = np.array([yv[0], -yv[1]]) / max(np.linalg.norm(yv), 1e-9) * min(150, max(90, 0.8 * L))
                draw.arrow(c, theme.dim(theme.OBSERVED, ya), b, b + v, 4, 18, dashed=True)
                off = v / max(np.linalg.norm(v), 1e-9) * 30
                self.lbl_y.draw(c, b + v + off, "center", ya, register=False)
        for p in self.bfgs_path[:k + 1]:
            draw.circle(c, theme.ESTIMATE, to_px(p), 6)
        c.set_clip(old)
        # Curvature ellipses at the current iterate. They are an overlay on the plot,
        # so they may extend past its frame rather than being cut off.
        ea = fade(t, self.T_SHOW)
        cpx = to_px(pos)
        if ea > 0:
            Mtrue = ellipse_matrix(optim.hessian(GAMMA), 105)
            Mest = ellipse_matrix(B, 105)
            draw.dashed_polyline(c, theme.dim(theme.CURVATURE, ea), draw.ellipse_points(cpx, Mtrue, 120), 3,
                                 12, 8, closed=True)
            draw.polyline(c, theme.dim(theme.ESTIMATE, ea), draw.ellipse_points(cpx, Mest, 120), 4, closed=True)
        draw.glow(c, theme.ESTIMATE, cpx, 30, 0.6)
        draw.circle(c, theme.ESTIMATE, cpx, 9)

        x = RIGHT_X
        legend(c, [("BFGS iterates", theme.ESTIMATE, "dot"),
                   ("estimated curvature B", theme.ESTIMATE, "ellipse"),
                   ("true curvature H", theme.CURVATURE, "dash"),
                   ("gradient descent", theme.GRADIENT, "line")], (x + 4, 190), fade(t, 0.2))
        # the secant condition, highlighted while the estimate updates
        hl = 0.0
        for m0, md in self.MORPH_T:
            if m0 - 0.4 <= t <= m0 + md + 0.3:
                hl = pulse(t)
        self.eq_secant.draw(c, (x, 470), "midleft", fade(t, 0.6), highlight={"B": hl})
        text.blit(c, "B must turn the step s into the observed change y", theme.LABEL, theme.TEXT_DIM,
                  (x, 548), "midleft", fade(t, 0.8), name="secant-note")
        # step counters
        a = fade(t, 0.3)
        text.blit(c, "steps to reach the center", theme.LABEL, theme.TEXT_DIM, (x, 660), "midleft", a, name="steps")
        text.blit(c, "BFGS", theme.BODY, theme.ESTIMATE, (x, 722), "midleft", a, name="bfgs")
        done = t >= self.STEP_T[-1][0] + self.STEP_T[-1][1]
        text.blit(c, f"{k}", theme.HEADING + 8, theme.ESTIMATE if done else theme.TEXT, (x + 700, 722),
                  "midright", a, name="bfgs-n")
        text.blit(c, "Gradient descent", theme.BODY, theme.GRADIENT, (x, 792), "midleft", a, name="gd")
        text.blit(c, f"{len(self.gd.path) - 1}", theme.HEADING + 8, theme.GRADIENT, (x + 700, 792), "midright", a,
                  name="gd-n")

    # beat 3 -------------------------------------------------------------------------------
    def _draw_compare(self, c):
        t, b = self.t, self.build
        x_lbl, x_eq = 150, 470
        a1 = fade(t, 0.2)
        text.blit(c, "Curvature", theme.BODY, theme.CURVATURE, (x_lbl, 260), "midleft", a1, name="row1")
        self.eq_row1.draw(c, (x_eq, 260), "midleft", a1)
        text.blit(c, "Hessian of f", theme.SMALL + 2, theme.TEXT_DIM, (x_lbl, 304), "midleft", a1, name="row1s")
        a2 = fade(self.build_t, 0.1) if b >= 1 else 0.0
        a2 = 1.0 if b >= 2 else a2
        text.blit(c, "Kinematics", theme.BODY, theme.ESTIMATE, (x_lbl, 400), "midleft", a2, name="row2")
        self.eq_row2.draw(c, (x_eq, 400), "midleft", a2)
        text.blit(c, "robot Jacobian", theme.SMALL + 2, theme.TEXT_DIM, (x_lbl, 444), "midleft", a2, name="row2s")
        if b >= 1:
            # Tie the roles across the two rows.
            for k1, k2 in (("B", "J"), ("s", "dy"), ("y", "dx")):
                r1 = self.eq_row1.key_rect(k1, (x_eq, 260), "midleft")
                r2 = self.eq_row2.key_rect(k2, (x_eq, 400), "midleft")
                if r1 and r2:
                    col = self.eq_row1.colors[k1]
                    draw.dashed_line(c, theme.dim(col, 0.45 * a2), (r1.centerx, r1.bottom + 10),
                                     (r2.centerx, r2.top - 10), 2, 6, 6)
        if b >= 2:
            bt = self.build_t
            a3 = fade(bt, 0.1)
            pos = (x_lbl, 640)
            hl = seg(bt, 0.8, 0.5) * pulse(bt - 0.8)
            err = ("edx", "eJ")
            r = self.eq_broyden.draw(c, pos, "midleft", a3, highlight={err: hl},
                                     colors=None)
            er = self.eq_broyden.key_rect(err, pos, "midleft")
            if er is not None:
                ba = seg(bt, 0.9, 0.5)
                if ba > 0:
                    draw.brace_h(c, theme.dim(theme.TEXT, ba), er.left + 4, er.right - 4, r.bottom + 10, 16, 3)
                    text.blit(c, "prediction error", theme.LABEL, theme.dim(theme.TEXT, 1.0), (er.centerx, r.bottom + 40),
                              "midtop", ba, name="pred-err")
            text.blit(c, "Broyden's update  ·  Yip and Camarillo, eq. (10)", theme.SMALL + 2, theme.TEXT_DIM,
                      (x_lbl, 820), "midleft", fade(bt, 1.2), name="tag")
            self._draw_vector_diagram(c, bt)

    def _draw_vector_diagram(self, c, bt):
        """Predicted J-hat dy vs measured dx; the update rotates one onto the other."""
        box = pygame.Rect(1300, 170, 520, 600)
        a = fade(bt, 0.3)
        if a <= 0:
            return
        draw.panel(c, box, theme.dim(theme.BG_PANEL, a), theme.dim(theme.PANEL_EDGE, a))
        o = np.array([box.left + 90, box.bottom - 90])
        meas = np.array([230.0, -300.0])
        pred0 = np.array([330.0, -60.0])
        f = seg(bt, 2.4, 1.4)
        ang0, ang1 = math.atan2(pred0[1], pred0[0]), math.atan2(meas[1], meas[0])
        L0, L1 = np.linalg.norm(pred0), np.linalg.norm(meas)
        ang, L = lerp(ang0, ang1, f), lerp(L0, L1, f)
        pred = np.array([math.cos(ang), math.sin(ang)]) * L
        if f > 0:
            draw.arrow(c, theme.dim(theme.ESTIMATE, 0.3 * a), o, o + pred0, 3, 18, dashed=True)
            text.blit(c, "before", theme.SMALL, theme.dim(theme.ESTIMATE, 0.6), o + pred0 + np.array([0, 28]),
                      "midtop", a * f, kind="scene", name="before")
        draw.arrow(c, theme.dim(theme.OBSERVED, a), o, o + meas, 4, 22, dashed=True)
        draw.arrow(c, theme.dim(theme.ESTIMATE, a), o, o + pred, 5, 22)
        ea = a * (1 - f)
        if ea > 0.02:
            draw.arrow(c, theme.dim(theme.TEXT, ea), o + pred, o + meas, 3, 16)
        self.lbl_meas.draw(c, o + meas + np.array([-18, -8]), "bottomright", a, register=False)
        lp = o + pred + pred / max(np.linalg.norm(pred), 1) * 34
        self.lbl_pred.draw(c, lp + np.array([6, 0]), "midleft" if f < 0.5 else "bottomleft", a, register=False)
        text.blit(c, "predicted", theme.SMALL + 2, theme.ESTIMATE, (box.left + 30, box.top + 26), "topleft", a,
                  name="diag1")
        text.blit(c, "measured", theme.SMALL + 2, theme.OBSERVED, (box.left + 30, box.top + 62), "topleft", a,
                  name="diag2")
        text.blit(c, "after the update, the prediction matches", theme.SMALL, theme.TEXT_DIM,
                  (box.centerx, box.bottom - 30), "center", a * seg(bt, 3.6, 0.6), name="diag3")
        draw.circle(c, theme.dim(theme.TEXT, a), o, 6)

    def screenshot_script(self):
        return [(0.0, self.reveal_all)]
