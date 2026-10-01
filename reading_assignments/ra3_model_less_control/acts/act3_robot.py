"""Act 3: model-less control of a continuum robot (Yip and Camarillo 2014)."""
from __future__ import annotations

import math

import numpy as np
import pygame

from acts import robotdraw as rd
from acts.base import Act
from core import draw, text, theme
from core.anim import Tween, clamp, ease_in_out, ease_out, fade, lerp, seg
from core.mathtext import Eq, pulse
from core.widgets import Slider, hbar, sparkline
from sim import controller as ctl
from sim import robot as rb

# ---- layout (virtual px) ----------------------------------------------------------------
PANEL_TOP, PANEL_H = 132, 550
LEFT = pygame.Rect(80, PANEL_TOP, 870, PANEL_H)
RIGHT = pygame.Rect(970, PANEL_TOP, 870, PANEL_H)
SOLO = pygame.Rect(470, PANEL_TOP, 980, 770)
DUO_CENTER, DUO_SCALE = (-4.0, 20.0), 3.6
SOLO_CENTER, SOLO_SCALE = (40.0, 40.0), 5.4
STRIP_Y = PANEL_TOP + PANEL_H + 16
SLIDER_Y = 910
BEND_MAX = math.radians(120)

HOME_TARGET = rb.local_tip(rb.L0 + rb.Y_HOME[0], 0.0)


def tension_frac(tau):
    return tau / 2.6


class RobotAct(Act):
    number = 3
    title = "Model-less control"
    captions = [
        "Each arrow: how the tip moves when one actuator moves.",
        "Build the first estimate by moving each actuator a little.",
        "When the model is right, both controllers work.",
        "Estimate the Jacobian online, and the bend stops mattering.",
        "Smoothing trades jitter for lag. The paper names this as a limitation.",
    ]
    settle = 10.0

    def __init__(self):
        super().__init__()
        self.sim = ctl.TwinSim()
        self.phi_tw = Tween(0.0, 3.0)
        self.bend_target = 0.0
        self.show_truth = False
        self.noise_on = False
        self.meas_hist = {"model": [], "modelless": []}
        self.layout_t = 1.0
        self.wiggle_cols = ctl.Controller.wiggle_columns(rb.Plant())
        self.w_est = ctl.column_weights(self.wiggle_cols)
        self.sliders = {}
        self._make_sliders()
        c = theme
        self.eq_jac = Eq([(r"\Delta x", "dx"), r"\;\approx\;", (r"J", "J"), r"\,", (r"\Delta y", "dy")],
                         size=c.MATH_BIG + 8, colors={"dx": c.OBSERVED, "J": c.TRUTH, "dy": c.STEP})
        self.eq_col = Eq([(r"\hat{J}_i", "J"), r"\;=\;\frac{", (r"\Delta x", "dx"), r"}{", (r"\Delta y_i", "dy"), r"}"],
                         size=c.MATH_BIG + 8, colors={"J": c.ESTIMATE, "dx": c.OBSERVED, "dy": c.STEP})
        self.eq_qp = {}
        for key, col in (("model", c.MODEL), ("modelless", c.ESTIMATE)):
            J = r"J_{\mathrm{model}}" if key == "model" else r"\hat{J}"
            self.eq_qp[key] = Eq([r"\min_{\Delta y}\ \Vert", (r"\tau", "tau"), r"+K\Delta y\Vert^2\quad\mathrm{s.t.}\quad",
                                  (J, "J"), r"W\Delta y=", (r"\Delta x_d", "dxd"), r",\ \ ", (r"\tau", "tau2"),
                                  r"+K\Delta y\geq", (r"\tau_{\min}", "tmin")],
                                 size=30, colors={"tau": c.TENSION, "tau2": c.TENSION, "J": col,
                                                  "dxd": c.TARGET, "tmin": c.WARNING})
        self.lbl_tau = [Eq([r"\tau_L"], size=28, color=c.TENSION), Eq([r"\tau_R"], size=28, color=c.TENSION)]
        self.lbl_phi = Eq([r"\varphi"], size=34, color=c.TEXT_DIM)

    def equations(self):
        return [self.eq_jac, self.eq_col, *self.eq_qp.values(), *self.lbl_tau, self.lbl_phi]

    # ---- sliders ---------------------------------------------------------------------------
    def _make_sliders(self):
        c = theme
        y = SLIDER_Y
        self.sliders["phi"] = Slider(150, y, 300, "endoscope bend φ", 0.0, 120.0, 0.0, "{:.0f}°",
                                     c.TEXT, on_change=self._phi_drag)
        self.sliders["sigma"] = Slider(510, y, 280, "sensor noise σ", 0.0, 2.0, 0.0, "{:.1f} mm",
                                       c.WARNING, on_change=self._sigma)
        self.sliders["alpha"] = Slider(850, y, 280, "smoothing α", 0.1, 1.0, 0.5, "{:.2f}",
                                       c.ESTIMATE, on_change=self._alpha)
        self.sliders["thr"] = Slider(1190, y, 280, "update threshold", 0.0, 3.0, 0.5, "{:.1f} mm",
                                     c.ESTIMATE, on_change=self._thr)
        self.sliders["g"] = Slider(1530, y, 260, "tendon friction", 0.5, 1.0, rb.G_TRUE, "g = {:.2f}",
                                   c.TRUTH, on_change=self._g)

    def _phi_drag(self, v):
        self.phi_tw.set(math.radians(v), instant=True)
        self.bend_target = math.radians(v)

    def _sigma(self, v):
        self.sim.sigma = v
        self.noise_on = v > 0

    def _alpha(self, v):
        self.sim.rigs["modelless"].ctrl.alpha = v

    def _thr(self, v):
        self.sim.rigs["modelless"].ctrl.threshold = v

    def _g(self, v):
        self.sim.set_g(v)

    def _visible_sliders(self):
        if self.beat == 3:
            return ["phi"]
        if self.beat == 4:
            return ["phi", "sigma", "alpha", "thr", "g"]
        return []

    # ---- beats -------------------------------------------------------------------------------
    def on_beat(self):
        b = self.beat
        if b >= 2:
            self.layout_t = 0.0 if b == 2 else 1.0
            self.sim = ctl.TwinSim()
            self.sim.reset()
            self.phi_tw = Tween(0.0, 3.0)
            self.bend_target = 0.0
            self.meas_hist = {"model": [], "modelless": []}
            self.show_truth = b >= 3
            self.noise_on = False
            for k, s in self.sliders.items():
                s.dragging = False
            self.sliders["phi"].set(0.0)
            self.sliders["sigma"].set(0.0)
            self.sliders["alpha"].set(0.5)
            self.sliders["thr"].set(0.5)
            self.sliders["g"].set(rb.G_TRUE)
            if b >= 3:
                self.sim.toggle_trajectory()
            if b == 4:
                self._set_noise(True)

    def _set_noise(self, on):
        self.noise_on = on
        v = 0.6 if on else 0.0
        self.sliders["sigma"].set(v)
        self.sim.sigma = v

    def toggle_bend(self):
        self.bend_target = 0.0 if self.bend_target > 0.5 * BEND_MAX else BEND_MAX
        dur = 0.8 if self.beat == 4 else 3.0
        self.phi_tw.set(self.bend_target, dur)

    def handle(self, ev):
        if self.beat < 2:
            return False
        for k in self._visible_sliders():
            if self.sliders[k].handle(ev):
                return True
        if ev.kind == "down" and ev.pos is not None:
            for key, view in self._views().items():
                if view.rect.collidepoint(ev.pos):
                    self.sim.click_world(view.mm(ev.pos))
                    return True
        if ev.kind == "key":
            k = ev.key
            if k == pygame.K_t:
                self.sim.toggle_trajectory()
            elif k == pygame.K_b and self.beat >= 2:
                self.toggle_bend()
            elif k == pygame.K_j:
                self.show_truth = not self.show_truth
            elif k == pygame.K_n:
                self._set_noise(not self.noise_on)
            else:
                return False
            return True
        return False

    def update(self, dt):
        super().update(dt)
        if self.beat >= 2:
            self.layout_t = min(1.0, self.layout_t + dt / 1.2)
            self.phi_tw.update(dt)
            phi = self.phi_tw.value
            self.sim.set_phi(phi)
            if not self.sliders["phi"].dragging:
                self.sliders["phi"].set(math.degrees(phi))
            self.sim.advance(dt)
            for key, r in self.sim.rigs.items():
                if self.sim.sigma > 0:
                    h = self.meas_hist[key]
                    h.append(r.x_meas.copy())
                    del h[:-24]
                else:
                    self.meas_hist[key] = []

    # ---- geometry --------------------------------------------------------------------------------
    def _views(self):
        duo_r = rd.View(RIGHT, DUO_CENTER, DUO_SCALE)
        duo_l = rd.View(LEFT, DUO_CENTER, DUO_SCALE)
        solo = rd.View(SOLO, SOLO_CENTER, SOLO_SCALE)
        if self.beat < 2:
            return {"modelless": solo}
        f = ease_in_out(self.layout_t)
        return {"model": duo_l, "modelless": rd.View.lerp(solo, duo_r, f)}

    # ---- drawing --------------------------------------------------------------------------------
    def draw(self, c):
        self.draw_header(c)
        if self.beat < 2:
            self._draw_solo(c)
        else:
            self._draw_duo(c)
        self.draw_caption(c)

    def _panel_base(self, c, view, phi, alpha=1.0, reach=True):
        draw.panel(c, view.rect, theme.dim(theme.BG_PANEL, alpha), theme.dim(theme.PANEL_EDGE, alpha), 16)
        old = c.get_clip()
        c.set_clip(view.rect.inflate(-4, -4))
        rd.grid(c, view, alpha)
        if reach:
            outline = [rb.world_from_local(p, phi) for p in rb.reach_outline_local()]
            pts = view.pts(np.array(outline))
            pygame.draw.polygon(c, theme.mix(theme.BG_PANEL, theme.TRUTH, 0.045 * alpha), [tuple(p) for p in pts])
        rd.endoscope(c, view, phi, alpha)
        c.set_clip(old)

    def _title_plate(self, c, title, rect, color, alpha):
        """Panel title on a plate, so robot parts passing underneath never collide with it."""
        w, h = text.size_of(title, theme.LABEL)
        plate = pygame.Rect(rect.left + 3, rect.top + 3, w + 44, h + 22)
        pygame.draw.rect(c, theme.dim(theme.BG_PANEL, alpha), plate, border_top_left_radius=14,
                         border_bottom_right_radius=14)
        text.blit(c, title, theme.LABEL, theme.dim(color, alpha), (rect.left + 24, rect.top + 14), "topleft",
                  name="panel-title")

    def _exit_angle(self, c, view, phi, alpha):
        """Dashed line where the model thinks the instrument exits, and the true bend."""
        E, psi = rb.exit_pose(phi)
        Ep = np.array(view.px(E))
        L = 110
        d0 = np.array([math.cos(rb.PSI0), -math.sin(rb.PSI0)])
        draw.dashed_line(c, theme.dim(theme.TEXT_FAINT, alpha), Ep, Ep + d0 * L, 2, 8, 7)
        if phi > math.radians(4):
            arc = [Ep + 62 * np.array([math.cos(rb.PSI0 + a), -math.sin(rb.PSI0 + a)])
                   for a in np.linspace(0, phi, 30)]
            draw.polyline(c, theme.dim(theme.TEXT_DIM, alpha), arc, 3)
            mid = rb.PSI0 + phi / 2
            p = Ep + 92 * np.array([math.cos(mid), -math.sin(mid)])
            self.lbl_phi.draw(c, p, "center", alpha, register=False)

    # solo beats ------------------------------------------------------------------------------------
    def _solo_state(self, t):
        """Actuators and arrow growth for the scripted beats 3.1 and 3.2."""
        y = rb.Y_HOME.copy()
        grow = [0.0, 0.0, 0.0]
        active = -1
        if self.beat == 0:
            for i in range(3):
                t0 = 0.8 + 2.4 * i
                grow[i] = seg(t, t0, 0.7, ease_out)
                u = (t - t0) / 2.0
                if 0 <= u <= 1:
                    active = i
                    env = math.sin(math.pi * u)
                    if i == 0:
                        y[0] += 14.0 * env * math.sin(2 * math.pi * u)
                    else:
                        y[i] += 9.0 * env
        else:
            for i in range(3):
                t0 = 0.8 + 2.0 * i
                amp = (5.0, 3.0, 3.0)[i]
                u = t - t0
                if 0 <= u <= 1.3:
                    active = i
                    y[i] += amp * (seg(u, 0.0, 0.3) - seg(u, 0.85, 0.35))
                grow[i] = seg(t, t0 + 0.45, 0.6, ease_out)
        return y, grow, active

    def _draw_solo(self, c):
        t = self.t
        view = self._views()["modelless"]
        self._panel_base(c, view, 0.0, reach=False)
        y, grow, active = self._solo_state(t)
        old = c.get_clip()
        c.set_clip(view.rect.inflate(-4, -4))
        tip = rd.instrument(c, view, y, 0.0, rb.G_TRUE, theme.ESTIMATE, glow=0.6)
        if self.beat == 0:
            J = rb.jacobian(y, 0.0, rb.G_TRUE) * self.w_est
            rd.jacobian_arrows(c, view, tip, J, theme.TRUTH_COLS, dashed=True, grow=grow, length=120,
                               label_size=theme.LABEL, width=5)
        else:
            # A twitch: the tip moves (observed), and that column of the estimate appears.
            if active >= 0:
                y0 = rb.Y_HOME
                p0 = np.array(view.px(rb.tip(y0, 0.0, rb.G_TRUE)))
                p1 = np.array(view.px(rb.tip(y, 0.0, rb.G_TRUE)))
                if np.linalg.norm(p1 - p0) > 6:
                    draw.arrow(c, theme.OBSERVED, p0, p1, 3, 12)
            J = self.wiggle_cols * self.w_est
            tip0 = np.array(view.px(rb.tip(rb.Y_HOME, 0.0, rb.G_TRUE)))
            rd.jacobian_arrows(c, view, tip0, J, theme.ESTIMATE_COLS, grow=grow, length=120,
                               label_size=theme.LABEL, width=5)
        c.set_clip(old)
        self._title_plate(c, "Model-less: estimates its Jacobian" if self.beat == 1 else "A tendon-driven continuum robot",
                          view.rect, theme.ESTIMATE if self.beat == 1 else theme.TEXT_DIM, 1.0)
        # left column: the relation being built
        lx = 96
        if self.beat == 0:
            self.eq_jac.draw(c, (lx, 360), "midleft", fade(t, 0.3))
            text.blit(c, "one column", theme.LABEL, theme.TEXT_DIM, (lx, 440), "midleft", fade(t, 0.6), name="l1")
            text.blit(c, "per actuator", theme.LABEL, theme.TEXT_DIM, (lx, 478), "midleft", fade(t, 0.6), name="l2")
        else:
            hl = pulse(t) if active >= 0 else 0.0
            self.eq_col.draw(c, (lx, 360), "midleft", fade(t, 0.3), highlight={"J": hl})
            n = sum(1 for g in grow if g >= 1)
            text.blit(c, f"columns measured: {n} of 3", theme.LABEL, theme.TEXT_DIM, (lx, 470), "midleft",
                      fade(t, 0.6), name="cols")
        self._draw_actuators(c, y, active, grow)

    def _draw_actuators(self, c, y, active, grow):
        x0 = SOLO.right + 40
        cols = theme.TRUTH_COLS if self.beat == 0 else theme.ESTIMATE_COLS
        text.blit(c, "actuators", theme.LABEL, theme.TEXT_DIM, (x0, 250), "midleft", name="act-title")
        for i in range(3):
            cy = 330 + i * 150
            on = i == active
            col = cols[i]
            base = theme.TEXT if on else theme.TEXT_DIM
            box = pygame.Rect(x0, cy - 36, 72, 72)
            draw.panel(c, box, theme.BG_PANEL, col if on else theme.PANEL_EDGE, 12, 2)
            cx, cyy = box.center
            if i == 0:
                pygame.draw.line(c, base, (cx, cyy - 24), (cx, cyy + 24), 3)
                off = (y[0] - rb.Y_HOME[0]) * 1.2
                pygame.draw.rect(c, col, (cx - 13, cyy - 7 - off, 26, 14), border_radius=4)
            else:
                draw.ring(c, base, (cx, cyy), 20, 3)
                ang = (y[i] - rb.Y_HOME[i]) * 0.5 + (0 if i == 1 else math.pi)
                draw.line(c, col, (cx, cyy), (cx + 20 * math.cos(ang), cyy - 20 * math.sin(ang)), 4)
                draw.line(c, base, (cx + (-20 if i == 1 else 20), cyy), (cx + (-20 if i == 1 else 20), cyy + 34), 2)
            text.blit(c, theme.COL_NAMES[i], theme.LABEL, col if on else theme.TEXT, (x0 + 96, cy - 14), "midleft",
                      name="act-name")
            text.blit(c, f"{theme.COL_SHORT[i]}  ·  {y[i] - rb.Y_HOME[i]:+.1f} mm", theme.SMALL,
                      theme.TEXT_DIM, (x0 + 96, cy + 22), "midleft", name="act-val")

    # duo beats ------------------------------------------------------------------------------------
    def _draw_duo(self, c):
        t = self.t
        views = self._views()
        la = seg(self.layout_t, 0.45, 0.55) if self.beat == 2 else 1.0
        phi = self.sim.phi
        tgt = self.sim.target_world()
        for key in ("model", "modelless"):
            view = views[key]
            alpha = la if key == "model" else 1.0
            if alpha <= 0.01:
                continue
            rig = self.sim.rigs[key]
            color = theme.MODEL if key == "model" else theme.ESTIMATE
            self._panel_base(c, view, phi, alpha)
            old = c.get_clip()
            c.set_clip(view.rect.inflate(-4, -4))
            if self.beat >= 3:
                self._exit_angle(c, view, phi, alpha)
            if self.sim.traj_on:
                sq = [rb.world_from_local(p, phi) for p in self.sim.square_outline_local()]
                draw.dashed_polyline(c, theme.dim(theme.TARGET, 0.28 * alpha), view.pts(np.array(sq)), 2, 10, 8,
                                     closed=True)
            rd.trail(c, view, rig.trail, color, alpha)
            if self.meas_hist[key]:
                for p in self.meas_hist[key]:
                    draw.circle(c, theme.dim(theme.OBSERVED, 0.6 * alpha), view.px(p), 3)
            tip = rd.instrument(c, view, rig.plant.y, phi, rig.plant.g, color, alpha, glow=0.5)
            rd.target(c, view, tgt, alpha, t)
            A = rig.ctrl.A
            if self.show_truth:
                Jt = rig.plant.jacobian() * rig.ctrl.w
                rd.jacobian_arrows(c, view, tip, Jt, theme.TRUTH_COLS, alpha, dashed=True, labels=False, length=64)
            cols = theme.MODEL_COLS if key == "model" else theme.ESTIMATE_COLS
            rd.jacobian_arrows(c, view, tip, A, cols, alpha, length=64)
            c.set_clip(old)
            title = "Model-based: trusts its kinematics" if key == "model" else "Model-less: estimates its Jacobian"
            self._title_plate(c, title, view.rect, color, alpha)
            sa = self._strip_alpha()
            if sa > 0.01:
                self._draw_strip(c, key, view.rect, alpha * sa)
                if key == "model":
                    self._draw_warning(c, rig, view.rect)
        sa = self._strip_alpha()
        if sa > 0.01:
            self._draw_legend(c, sa)
            for k in self._visible_sliders():
                self.sliders[k].draw(c, sa * fade(t, 0.2))

    def _strip_alpha(self):
        """Readouts fade in once the panels have settled into place."""
        if self.beat != 2:
            return 1.0
        return fade(self.t, 1.25, 0.5)

    def _draw_legend(self, c, a=1.0):
        items = [("assumed model", theme.MODEL, "line"), ("estimate", theme.ESTIMATE, "line")]
        if self.show_truth:
            items.append(("truth", theme.TRUTH, "dash"))
        items.append(("target", theme.TARGET, "ring"))
        x = 760
        y = 92
        for label, col, style in items:
            col = theme.dim(col, a)
            if style == "dash":
                draw.dashed_line(c, col, (x, y), (x + 40, y), 3, 9, 6)
            elif style == "ring":
                draw.crosshair(c, col, (x + 20, y), 9, 2)
            else:
                draw.line(c, col, (x, y), (x + 40, y), 4)
            r = text.blit(c, label, theme.SMALL, theme.TEXT_DIM, (x + 54, y), "midleft", a, name="legend")
            x = r.right + 40

    def _draw_strip(self, c, key, rect, alpha):
        rig = self.sim.rigs[key]
        color = theme.MODEL if key == "model" else theme.ESTIMATE
        x, y = rect.left + 8, STRIP_Y
        err = self.sim.error(key)
        bad = err > 10.0
        text.blit(c, "tip error", theme.SMALL, theme.TEXT_DIM, (x, y), "topleft", alpha, name="err-lbl")
        text.blit(c, f"{err:.1f} mm", theme.HEADING, theme.WARNING if bad else theme.TEXT, (x, y + 32), "topleft",
                  alpha, name="err")
        sparkline(c, rig.err_hist, (x + 200, y + 8, 250, 64), theme.WARNING if bad else color, 40.0, alpha)
        # tendon tension bars with the tau_min line
        bx, bw = rect.right - 330, 300
        tau = rb.tensions(rig.plant.y)
        for j in range(2):
            by = y + 8 + j * 34
            self.lbl_tau[j].draw(c, (bx - 16, by + 7), "midright", alpha, register=False)
            low = tau[j] < rb.TAU_MIN - 0.02
            hbar(c, (bx, by, bw, 14), tension_frac(tau[j]), theme.WARNING if low else theme.TENSION, alpha)
        mx = bx + bw * tension_frac(rb.TAU_MIN)
        draw.dashed_line(c, theme.dim(theme.WARNING, alpha), (mx, y), (mx, y + 58), 2, 5, 4)
        text.blit(c, "\u03c4 min", theme.SMALL - 2, theme.dim(theme.WARNING, alpha), (mx, y + 62), "midtop",
                  name="taumin")
        text.blit(c, "tendon tension", theme.SMALL - 2, theme.TEXT_DIM, (bx + bw, y + 62), "topright", alpha,
                  name="tension-lbl")
        if self.beat == 3:
            self.eq_qp[key].draw(c, (rect.centerx, y + 124), "center", alpha * fade(self.t, 0.4))

    def _draw_warning(self, c, rig, rect):
        a = clamp((rig.positive_feedback - 0.35) / 0.3)
        if a <= 0.01:
            return
        y = rect.bottom - 38
        msg = "Positive feedback: the model moves the tip the wrong way."
        w, h = text.size_of(msg, theme.SMALL + 2)
        plate = pygame.Rect(rect.left + 3, y - h / 2 - 12, w + 90, h + 24)
        veil = pygame.Surface(plate.size, pygame.SRCALPHA)
        veil.fill((*theme.BG_PANEL, int(225 * a)))
        c.blit(veil, plate)
        draw.warning_icon(c, theme.dim(theme.WARNING, a), (rect.left + 40, y), 30)
        text.blit(c, msg, theme.SMALL + 2, theme.WARNING, (rect.left + 68, y), "midleft", a, name="warning")

    # ---- harness ------------------------------------------------------------------------------------
    def screenshot_script(self):
        if self.beat == 2:
            def click():
                v = self._views()["model"]
                self.sim.set_target_local(np.array([78.0, 22.0]))
            return [(0.0, click), (4.0, self.sim.toggle_trajectory)]
        if self.beat == 3:
            return [(1.0, self.toggle_bend)]
        if self.beat == 4:
            def setup():
                self.sliders["alpha"].set(0.3, notify=True)
            return [(0.0, setup), (2.0, self.toggle_bend)]
        return []
