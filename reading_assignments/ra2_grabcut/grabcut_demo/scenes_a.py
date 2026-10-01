"""
Part A - the math walkthrough.

Content area for every scene is x in [0.042, 0.958], y in [0.07, 0.84]; the
band above that belongs to the runner's title block. Scenes build artists in
`build()` and only mutate them in `draw()`.
"""

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from . import draw as D
from . import theme as T
from . import toy as TOY
from .graphview import GraphView
from .scene import Scene

# shared toy solution - computed once, used by several scenes
TOY_RUN = None


def toy_run():
    global TOY_RUN
    if TOY_RUN is None:
        TOY_RUN = TOY.run_toy_grabcut(iters=5)
    return TOY_RUN


L, R_, B, TOPY = 0.042, 0.958, 0.07, 0.84


# ===========================================================================
class TitleScene(Scene):
    title = ""
    subtitle = ""
    n_beats = 3
    beat_seconds = 1.0

    def build(self):
        fig = self.fig
        self.runner.hud_title.set_alpha(0)
        self.runner.hud_sub.set_alpha(0)
        self.runner.hud_rule.set_alpha(0)

        ax = self.add_axes([0.07, 0.10, 0.48, 0.74])
        T.blank(ax)
        self.t1 = ax.text(0, 0.72, "GrabCut", fontsize=74, color=T.TEXT,
                          transform=ax.transAxes, va="center", alpha=0)
        self.t2 = ax.text(0, 0.575,
                          "Interactive foreground extraction\nusing iterated graph cuts",
                          fontsize=23, color=T.BLUE, transform=ax.transAxes,
                          va="center", linespacing=1.5, alpha=0)
        self.rule = ax.plot([0, 0.46], [0.44, 0.44], color=T.PANEL_EDGE, lw=1.2,
                            transform=ax.transAxes, alpha=0)[0]
        self.t3 = ax.text(0, 0.375,
                          "Rother, Kolmogorov and Blake   ·   ACM SIGGRAPH 2004",
                          fontsize=14, color=T.MUTED, transform=ax.transAxes,
                          va="center", alpha=0)
        self.t4 = ax.text(0, 0.19,
                          "A bounding box is almost no information.\n"
                          "GrabCut turns it into a segmentation by alternating\n"
                          "two subproblems it can each solve exactly.",
                          fontsize=17, color=T.TEXT, transform=ax.transAxes,
                          va="center", linespacing=1.75, alpha=0)

        # teaser: the toy image with its final cut drawing itself
        gax = self.add_axes([0.615, 0.17, 0.315, 0.56])
        r = toy_run()
        self.grid = D.ImageGrid(gax, r["img"], gap=0.09)
        self.outline = D.mask_outline(gax, r["history"][-1]["alpha"], lw=3.0)
        self.outline.set_alpha(0)
        self.cap = gax.text(0.5, -0.055, "an 8 × 8 toy image, and the cut GrabCut finds",
                            transform=gax.transAxes, ha="center", va="top",
                            fontsize=12, color=T.DIM, alpha=0)
        self.grid.set_image_alpha(0)
        self.final_mask = r["history"][-1]["alpha"]

    def draw(self, beat, t):
        e = T.smooth(t)
        if beat == 0:
            self.t1.set_alpha(e)
            self.t2.set_alpha(max(0.0, (e - 0.3) / 0.7))
            for a in (self.rule, self.t3, self.t4, self.cap):
                a.set_alpha(0)
            self.grid.set_image_alpha(0)
            self.grid.clear_mask()
            self.outline.set_alpha(0)
        elif beat == 1:
            self.t1.set_alpha(1)
            self.t2.set_alpha(1)
            self.rule.set_alpha(e)
            self.t3.set_alpha(e)
            self.t4.set_alpha(0)
            self.grid.set_image_alpha(e)
            self.cap.set_alpha(e * 0.9)
            self.grid.clear_mask()
            self.outline.set_alpha(0)
        else:
            for a in (self.t1, self.t2, self.rule, self.t3):
                a.set_alpha(1)
            self.grid.set_image_alpha(1)
            self.cap.set_alpha(0.9)
            self.t4.set_alpha(e)
            self.grid.set_mask(self.final_mask, strength=e)
            self.outline.set_alpha(e)

    def leave(self):
        self.runner.hud_title.set_alpha(1)
        self.runner.hud_sub.set_alpha(1)
        self.runner.hud_rule.set_alpha(1)
        super().leave()

    def export_beats(self):
        return {2: "01_title"}


# ===========================================================================
class SetupScene(Scene):
    title = "The problem, and the only thing the user tells us"
    subtitle = "Pixels, labels, and the trimap that a bounding box induces"
    n_beats = 4

    def build(self):
        r = toy_run()
        self.r = r
        ax = self.add_axes([L, 0.10, 0.44, 0.68])
        T.blank(ax)
        self.defs = D.DefList(ax, [
            (r"$z \;=\; (z_1,\, z_2,\, \dots,\, z_N)$",
             "the image: N pixels, each an RGB triple"),
            (r"$\alpha \;=\; (\alpha_1,\dots,\alpha_N),\quad \alpha_n \in \{0,1\}$",
             "the labelling: 0 is background, 1 is foreground.\n"
             "This vector is the entire answer we are after."),
            (r"$n \in T_B \;\Rightarrow\; \alpha_n = 0$",
             "the box pins everything outside it to background\n"
             "and leaves the inside unknown. That is the whole\n"
             "of the supervision."),
        ], x=0.0, y0=0.86, row_h=0.325, gap=0.095, eq_size=26)

        gax = self.add_axes([0.575, 0.145, 0.295, 0.60])
        self.grid = D.ImageGrid(gax, r["img"], gap=0.09)
        self.gax = gax
        x0, y0, x1, y1 = r["rect"]
        n = r["img"].shape[0]
        self.box = Rectangle((y0 - 0.06, n - 1 - x1 - 0.06),
                             (y1 - y0) + 1.12, (x1 - x0) + 1.12,
                             facecolor="none", edgecolor=T.GOLD, lw=2.6,
                             zorder=9, alpha=0)
        gax.add_patch(self.box)
        self.boxlab = gax.text(0.5, 1.055, "the user's bounding box",
                               transform=gax.transAxes, ha="center", va="bottom",
                               fontsize=13, color=T.GOLD, alpha=0)
        self.leg = [D.legend_dot(gax, 0.02, -0.105, T.SLATE_DEEP,
                                 r"$T_B$    definite background", size=12.5),
                    D.legend_dot(gax, 0.02, -0.185, T.GOLD,
                                 r"$T_U$    unknown", size=12.5)]
        for d, t_ in self.leg:
            d.set_alpha(0)
            t_.set_alpha(0)
        self.tri = r["trimap"] == 1

    def draw(self, beat, t):
        e = T.smooth(t)
        self.defs.reveal(beat + 1 if beat < 3 else 3, t if beat < 3 else 1.0)
        self.grid.set_image_alpha(1.0)
        if beat <= 1:
            self.grid.clear_mask()
            self.box.set_alpha(0)
            self.boxlab.set_alpha(0)
        elif beat == 2:
            self.box.set_alpha(e)
            self.boxlab.set_alpha(e)
            self.grid.clear_mask()
        else:
            self.box.set_alpha(1)
            self.boxlab.set_alpha(1)
            self.defs.focus(2)
            self.grid.set_mask(self.tri, strength=e, bg_alpha=0.78)
            for d, t_ in self.leg:
                d.set_alpha(e)
                t_.set_alpha(e)

    def export_beats(self):
        return {3: "02_trimap"}


# ===========================================================================
class EnergyScene(Scene):
    title = "One number scores a whole labelling"
    subtitle = "The Gibbs energy, and the two questions it asks"
    n_beats = 5

    def seconds(self, beat):
        return 1.25 if beat == 1 else self.beat_seconds

    def build(self):
        ax = self.add_axes([L, 0.08, R_ - L, TOPY - 0.08])
        T.blank(ax)
        self.ax = ax
        self.lead = ax.text(0.5, 0.935, "Give me a labelling and I will give you a cost.",
                            transform=ax.transAxes, ha="center", va="center",
                            fontsize=19, color=T.MUTED, alpha=0)
        # The left-hand side slides from the centre of the screen into its place
        # in the full equation while the right-hand side fades in beside it.
        # Cross-fading two centred strings of different widths just smears.
        self.E_X0, self.E_X1 = 0.5, 0.271
        self.e_lhs = ax.text(self.E_X0, 0.715, r"$E(\alpha,\,k,\,\theta,\,z)$",
                             transform=ax.transAxes, ha="center", va="center",
                             fontsize=42, color=T.TEXT, alpha=0)
        self.e_rhs = ax.text(0.391, 0.715,
                             r"$=\; U(\alpha,\,k,\,\theta,\,z) \;+\; V(\alpha,\,z)$",
                             transform=ax.transAxes, ha="left", va="center",
                             fontsize=42, color=T.TEXT, alpha=0)
        # measured centres of E, U and V inside the rendered equation
        self.e_x, self.u_x, self.v_x = 0.271, 0.549, 0.776
        self.e_br = D.brace(ax, self.e_x - 0.105, self.e_x + 0.105, 0.628,
                            color=T.MUTED, lw=1.3)
        self.u_br = D.brace(ax, self.u_x - 0.107, self.u_x + 0.107, 0.628,
                            color=T.BLUE)
        self.v_br = D.brace(ax, self.v_x - 0.062, self.v_x + 0.058, 0.628,
                            color=T.GOLD)
        self.e_head = ax.text(self.e_x, 0.567, "the cost of one labelling",
                              transform=ax.transAxes, ha="center", va="center",
                              fontsize=14.5, color=T.MUTED, alpha=0)
        self.u_tag = ax.text(self.u_x, 0.472,
                             "Does each pixel's colour look like\nthe label it was given?",
                             transform=ax.transAxes, ha="center", va="center",
                             fontsize=14.5, color=T.MUTED, alpha=0, linespacing=1.6)
        self.v_tag = ax.text(self.v_x, 0.472,
                             "Do neighbouring pixels\nagree with each other?",
                             transform=ax.transAxes, ha="center", va="center",
                             fontsize=14.5, color=T.MUTED, alpha=0, linespacing=1.6)
        self.u_head = ax.text(self.u_x, 0.567, "data term", transform=ax.transAxes,
                              ha="center", va="center", fontsize=14.5, color=T.BLUE,
                              alpha=0)
        self.v_head = ax.text(self.v_x, 0.567, "smoothness term", transform=ax.transAxes,
                              ha="center", va="center", fontsize=14.5, color=T.GOLD,
                              alpha=0)
        self.goal_head = ax.text(0.5, 0.235, "SO THE SEGMENTATION WE WANT IS",
                                 transform=ax.transAxes, ha="center", va="center",
                                 fontsize=11, color=T.DIM, alpha=0)
        self.goal = ax.text(0.5, 0.145,
                            r"$\hat{\alpha} \;=\; \arg\min_{\alpha}\;"
                            r"\min_{k,\,\theta}\; E(\alpha,\,k,\,\theta,\,z)$",
                            transform=ax.transAxes, ha="center", va="center",
                            fontsize=30, color=T.TEXT, alpha=0)
        self.goal_cap = ax.text(0.5, 0.035,
                                "Neither minimisation is convex. The rest of the talk is "
                                "how to do it anyway.",
                                transform=ax.transAxes, ha="center", va="center",
                                fontsize=14, color=T.DIM, alpha=0)

    def draw(self, beat, t):
        e = T.smooth(t)
        eq_parts = (self.e_lhs, self.e_rhs)
        braces = (self.e_br, self.e_head, self.u_br, self.u_head, self.u_tag,
                  self.v_br, self.v_head, self.v_tag)
        goal = (self.goal, self.goal_head, self.goal_cap)

        if beat == 0:
            self.lead.set_alpha(e)
            self.e_lhs.set_alpha(max(0.0, (e - 0.35) / 0.65))
            self.e_lhs.set_position((self.E_X0, 0.715))
            self.e_rhs.set_alpha(0)
            for a in braces + goal:
                a.set_alpha(0)
        elif beat == 1:
            # slide first, then fade the right-hand side in: overlapping the two
            # puts U(...) underneath the sliding E(...)
            slide = T.smoother(min(t / 0.62, 1.0))
            self.lead.set_alpha(1 - e * 0.45)
            self.e_lhs.set_alpha(1)
            self.e_lhs.set_position((T.lerp(self.E_X0, self.E_X1, slide), 0.715))
            self.e_rhs.set_alpha(T.smooth(max(0.0, (t - 0.62) / 0.38)))
            for a in braces + goal:
                a.set_alpha(0)
        else:
            self.lead.set_alpha(0.55 if beat < 4 else 0.55 * (1 - e))
            for a in eq_parts:
                a.set_alpha(1)
            self.e_lhs.set_position((self.E_X1, 0.715))
            if beat == 2:
                self.e_br.set_alpha(e * 0.8)
                self.e_head.set_alpha(e * 0.8)
                for a in (self.u_br, self.u_head, self.u_tag):
                    a.set_alpha(e)
                for a in (self.v_br, self.v_head, self.v_tag) + goal:
                    a.set_alpha(0)
            elif beat == 3:
                for a in (self.e_br, self.e_head):
                    a.set_alpha(0.8)
                for a in (self.u_br, self.u_head, self.u_tag):
                    a.set_alpha(1)
                for a in (self.v_br, self.v_head, self.v_tag):
                    a.set_alpha(e)
                for a in goal:
                    a.set_alpha(0)
            else:
                for a in (self.e_br, self.e_head):
                    a.set_alpha(0.8)
                for a in braces[2:]:
                    a.set_alpha(1)
                self.goal_head.set_alpha(e)
                self.goal.set_alpha(e)
                self.goal_cap.set_alpha(max(0.0, (e - 0.4) / 0.6))

    def export_beats(self):
        return {4: "03_energy"}


# ===========================================================================
_PHOTO = None


def photo_models(K=5):
    """
    The colour model for Part A's scatter comes from the *same photo* Part B
    will segment, fit from the same kind of bounding box. The scatter on this
    screen is therefore literally the model the live tool starts from.
    """
    global _PHOTO
    if _PHOTO is None:
        from . import realimg as RI
        img = RI.load_image()
        rect = RI.default_rect(img)
        inside = RI.rect_trimap(img, rect)
        zf, zb = RI.sample_pixels(img, inside, n=1800, seed=1)
        fg0, bg0 = RI.fit_pair(zf, zb, K, seed=3)
        stages = [(fg0.params(), bg0.params())]
        for _ in range(3):                       # a few refits, to animate
            fg0.refit(zf)
            bg0.refit(zb)
            stages.append((fg0.params(), bg0.params()))
        _PHOTO = dict(img=img, rect=rect, zf=zf, zb=zb, stages=stages,
                      fg=fg0, bg=bg0)
    return _PHOTO


class DataTermScene(Scene):
    title = "The data term: does this colour belong to this label?"
    subtitle = ("Two Gaussian mixtures in RGB, K = 5 components each, "
                "one for foreground and one for background")
    n_beats = 6

    def build(self):
        from . import realimg as RI
        P = photo_models()
        self.P = P

        # --- left: the equation, built in three pieces -------------------
        ax = self.add_axes([L, 0.08, 0.50, TOPY - 0.08])
        T.blank(ax)
        self.ax = ax
        self.head = ax.text(0.0, 0.905,
                            "Every pixel is charged the negative log likelihood\n"
                            "of its colour under the model it was assigned to.",
                            transform=ax.transAxes, fontsize=15.5, color=T.MUTED,
                            va="center", alpha=0, linespacing=1.65)
        self.dlhs = ax.text(0.0, 0.705, r"$D(\alpha_n,\,k_n,\,\theta,\,z_n) \;=\;$",
                            transform=ax.transAxes, fontsize=23, color=T.TEXT,
                            va="center", alpha=0)
        self.parts = [
            ax.text(0.045, 0.565, r"$-\log \pi(\alpha_n,k_n)$",
                    transform=ax.transAxes, fontsize=23, color=T.TEXT,
                    va="center", alpha=0),
            ax.text(0.045, 0.425,
                    r"$+\;\frac{1}{2}\log\det\Sigma(\alpha_n,k_n)$",
                    transform=ax.transAxes, fontsize=23, color=T.TEXT,
                    va="center", alpha=0),
            ax.text(0.045, 0.275,
                    r"$+\;\frac{1}{2}\,[\,z_n-\mu(\alpha_n,k_n)\,]^{\top}\,"
                    r"\Sigma(\alpha_n,k_n)^{-1}\,[\,z_n-\mu(\alpha_n,k_n)\,]$",
                    transform=ax.transAxes, fontsize=19.5, color=T.TEXT,
                    va="center", alpha=0),
        ]
        self.notes = [
            ax.text(0.545, 0.565, "how big is this component?",
                    transform=ax.transAxes, fontsize=13, color=T.DIM,
                    va="center", alpha=0),
            ax.text(0.545, 0.425, "how spread out is it?",
                    transform=ax.transAxes, fontsize=13, color=T.DIM,
                    va="center", alpha=0),
            ax.text(0.045, 0.185, "how far is this colour from its centre,\n"
                                  "measured in the ellipse's own units?",
                    transform=ax.transAxes, fontsize=13, color=T.DIM,
                    va="top", alpha=0, linespacing=1.6),
        ]
        self.usum = ax.text(0.0, 0.035,
                            r"$U(\alpha,k,\theta,z) \;=\; \sum_n "
                            r"D(\alpha_n,\,k_n,\,\theta,\,z_n)$",
                            transform=ax.transAxes, fontsize=22, color=T.TEXT,
                            va="center", alpha=0)

        # --- right top: the photo and its box ----------------------------
        pax = self.add_axes([0.615, 0.618, 0.163, 0.205])
        T.blank(pax)
        pax.imshow(P["img"])
        x, y, w, h = P["rect"]
        pax.add_patch(Rectangle((x, y), w, h, facecolor="none",
                                edgecolor=T.GOLD, lw=2.0))
        pax.set_title("THE BOX SPLITS THE PIXELS", color=T.MUTED, fontsize=10.5,
                      loc="left", pad=9)
        self.pax = pax
        self.pcap = self.fig_text(0.795, 0.715,
                                  "Inside the box is not the object -\n"
                                  "it is the object plus a lot of\n"
                                  "background. Pulling those apart\n"
                                  "is what the iteration is for.",
                                  fontsize=12.5, color=T.DIM, va="center",
                                  ha="left", linespacing=1.65, alpha=0)

        # --- right: the colour-space scatter -----------------------------
        i, j = RI.PROJ
        sax = self.add_axes([0.615, 0.095, 0.343, 0.455])
        self.sax = sax
        T.panel(sax)
        allz = np.vstack([P["zf"], P["zb"]])
        for setlim, k in ((sax.set_xlim, i), (sax.set_ylim, j)):
            lo, hi = np.percentile(allz[:, k], [0.4, 99.6])
            pad = 0.17 * (hi - lo) + 0.02
            setlim(lo - pad, hi + pad)
        # axis names live inside the panel: outside, they fight the caption row
        sax.text(0.985, 0.028, f"{RI.PROJ_NAMES[0]} channel  →",
                 transform=sax.transAxes, ha="right", va="bottom",
                 fontsize=11.5, color=T.DIM, family="DejaVu Sans")
        sax.text(0.048, 0.030, f"{RI.PROJ_NAMES[1]} channel  →",
                 transform=sax.transAxes, ha="left", va="bottom", rotation=90,
                 fontsize=11.5, color=T.DIM, family="DejaVu Sans")
        sax.set_title(f"RGB COLOUR SPACE, PROJECTED ONTO "
                      f"{RI.PROJ_NAMES[0]} AND {RI.PROJ_NAMES[1]}"
                      "   ·   1.6σ ELLIPSES, K = 5 PER CLASS",
                      color=T.MUTED, fontsize=10.5, loc="left", pad=10)
        self.sc_b = sax.scatter(P["zb"][:, i], P["zb"][:, j], s=6,
                                c=T.SLATE, alpha=0.0, linewidths=0, zorder=3)
        self.sc_f = sax.scatter(P["zf"][:, i], P["zf"][:, j], s=6,
                                c=T.BLUE, alpha=0.0, linewidths=0, zorder=4)
        self.ell_b = D.EllipseSet(sax, 5, T.SLATE, proj=(i, j), zorder=6)
        self.ell_f = D.EllipseSet(sax, 5, T.BLUE, proj=(i, j), zorder=8)
        self.leg = [D.legend_dot(sax, 0.075, 0.935, T.BLUE, "inside the box", 12,
                                 r=0.012),
                    D.legend_dot(sax, 0.075, 0.868, T.SLATE, "outside the box", 12,
                                 r=0.012)]
        for d, t_ in self.leg:
            d.set_alpha(0)
            t_.set_alpha(0)
        self.ellcap = sax.text(0.985, 0.05, "", transform=sax.transAxes,
                               ha="right", va="bottom", fontsize=11,
                               color=T.DIM, alpha=0)

    def _ellipses(self, stage_f, stage_t, alpha=1.0):
        from .gmm import blend_params
        P = self.P
        a = P["stages"][stage_f]
        b = P["stages"][min(stage_t, len(P["stages"]) - 1)]
        wf, mf, cf, _ = blend_params(a[0], b[0], T.smooth(self._tt))
        wb, mb, cb, _ = blend_params(a[1], b[1], T.smooth(self._tt))
        self.ell_f.update(wf, mf, cf, alpha=alpha)
        self.ell_b.update(wb, mb, cb, alpha=alpha)

    def seconds(self, beat):
        return 1.5 if beat == 3 else self.beat_seconds

    def draw(self, beat, t):
        e = T.smooth(t)
        self._tt = 0.0
        self.head.set_alpha(1.0 if beat >= 1 else e)
        self.pcap.set_alpha(0.0 if beat == 0 else (e if beat == 1 else 1.0))
        for i, (p, nt) in enumerate(zip(self.parts, self.notes)):
            on = beat - 3 >= i
            cur = (beat - 3 == i)
            p.set_alpha(1.0 if (on and not cur) else (e if cur else 0.0))
            nt.set_alpha(0.95 if (on and not cur) else (e * 0.95 if cur else 0.0))
        self.dlhs.set_alpha(1.0 if beat >= 3 else 0.0)
        self.usum.set_alpha(e if beat == 5 else (1.0 if beat > 5 else 0.0))

        if beat == 0:
            for a in (self.sc_f, self.sc_b):
                a.set_alpha(0)
            self._ellipses(0, 0, 0.0)
            for d, t_ in self.leg:
                d.set_alpha(0)
                t_.set_alpha(0)
            self.ellcap.set_alpha(0)
        elif beat == 1:
            self.sc_b.set_alpha(0.55 * e)
            self.sc_f.set_alpha(0.75 * e)
            for d, t_ in self.leg:
                d.set_alpha(e)
                t_.set_alpha(e)
            self._ellipses(0, 0, 0.0)
            self.ellcap.set_alpha(0)
        elif beat == 2:
            self.sc_b.set_alpha(0.55)
            self.sc_f.set_alpha(0.75)
            for d, t_ in self.leg:
                d.set_alpha(1)
                t_.set_alpha(1)
            self._ellipses(0, 0, e)
            self.ellcap.set_alpha(e)
        else:
            self.sc_b.set_alpha(0.55)
            self.sc_f.set_alpha(0.75)
            for d, t_ in self.leg:
                d.set_alpha(1)
                t_.set_alpha(1)
            self.ellcap.set_alpha(1)
            if beat == 3:
                self._tt = t            # refit animates while the equation builds
                self._ellipses(0, 3, 1.0)
            else:
                self._tt = 1.0
                self._ellipses(0, 3, 1.0)

    def export_beats(self):
        return {5: "04_gmm_colour_space"}


# ===========================================================================
class SmoothScene(Scene):
    title = "The smoothness term: what does it cost to cut here?"
    subtitle = ("Neighbours that look alike should get the same label - "
                "so disagreeing between them is charged")
    n_beats = 5

    # the flattest and the steepest neighbouring pair in the toy image
    FLAT = ((7, 4), (7, 5))
    EDGE = ((4, 1), (4, 2))

    def build(self):
        r = toy_run()
        self.r = r
        img = r["img"]
        n = img.shape[0]
        pairs, w, beta = TOY.nlink_weights(img)
        self.pairs, self.w, self.beta = pairs, np.asarray(w), beta

        def weight_of(pa, pb):
            ia, ib = pa[0] * n + pa[1], pb[0] * n + pb[1]
            return self.w[pairs.index((min(ia, ib), max(ia, ib)))] / TOY.GAMMA

        # --- left: prose, then V, then beta -------------------------------
        ax = self.add_axes([L, 0.08, 0.50, TOPY - 0.08])
        T.blank(ax)
        self.ax = ax
        self.prose = ax.text(0.0, 0.905,
                             "Two pixels side by side that are almost the same "
                             "colour are\nprobably part of the same thing. Splitting "
                             "them is suspicious,\nso the energy charges you for it - "
                             "and charges you most\nexactly where the image is "
                             "flattest.",
                             transform=ax.transAxes, fontsize=16, color=T.TEXT,
                             va="top", alpha=0, linespacing=1.72)
        self.veq = ax.text(0.0, 0.60,
                           r"$V(\alpha,z) \;=\; \gamma \sum_{(m,n)\,\in\, C}\;"
                           r"[\,\alpha_n \neq \alpha_m\,]\;"
                           r"\exp\!\left(-\beta\,\|z_m - z_n\|^2\right)$",
                           transform=ax.transAxes, fontsize=23, color=T.TEXT,
                           va="center", alpha=0)
        self.vbr1 = D.brace(ax, 0.354, 0.516, 0.505, color=T.GOLD, lw=1.4)
        self.vbr2 = D.brace(ax, 0.525, 0.873, 0.505, color=T.BLUE, lw=1.4)
        self.vn1 = ax.text(0.435, 0.428, "1 only where\nthe labels differ",
                           transform=ax.transAxes, ha="center", va="center",
                           fontsize=12.5, color=T.GOLD, alpha=0, linespacing=1.6)
        self.vn2 = ax.text(0.699, 0.428,
                           "near 1 for similar colours,\nnear 0 across an edge",
                           transform=ax.transAxes, ha="center", va="center",
                           fontsize=12.5, color=T.BLUE, alpha=0, linespacing=1.6)
        self.beq = ax.text(0.0, 0.225,
                           r"$\beta \;=\; \left(\,2\,\langle\,\|z_m-z_n\|^2\,"
                           r"\rangle\,\right)^{-1} \;=\; %.1f$" % beta,
                           transform=ax.transAxes, fontsize=22, color=T.TEXT,
                           va="center", alpha=0)
        self.bnote = ax.text(0.0, 0.112,
                             "β is not a knob - it is read off this image's own "
                             "average contrast,\nso one γ behaves the same way on a "
                             "flat image and on a busy one.",
                             transform=ax.transAxes, fontsize=13.5, color=T.MUTED,
                             va="top", alpha=0, linespacing=1.68)

        # --- right: the toy image, with the two pairs marked ---------------
        gax = self.add_axes([0.655, 0.435, 0.255, 0.345])
        self.grid = D.ImageGrid(gax, img, gap=0.09)
        self.gax = gax
        self.marks = []
        for (pa, pb), col in ((self.FLAT, T.BLUE), (self.EDGE, T.GOLD)):
            xy = [(c + 0.5, n - 1 - r_ + 0.5) for r_, c in (pa, pb)]
            ln, = gax.plot([xy[0][0], xy[1][0]], [xy[0][1], xy[1][1]], color=col,
                           lw=3.0, solid_capstyle="round", zorder=10, alpha=0)
            rings = [gax.add_patch(Circle(p, 0.46, facecolor="none", edgecolor=col,
                                          lw=2.2, zorder=11, alpha=0)) for p in xy]
            self.marks.append((ln, rings))

        # --- right: two comparison cards -----------------------------------
        self.cards = []
        for i, ((pa, pb), col, title, verdict) in enumerate((
                (self.FLAT, T.BLUE, "FLAT REGION", "expensive to cut"),
                (self.EDGE, T.GOLD, "ACROSS AN EDGE", "cheap to cut"))):
            cax = self.add_axes([0.607 + i * 0.187, 0.115, 0.163, 0.245])
            T.panel(cax, title)
            cax.title.set_color(col)
            cax.set_xlim(0, 1)
            cax.set_ylim(0, 1)
            wv = weight_of(pa, pb)
            sw = [cax.add_patch(Rectangle((x, 0.60), 0.26, 0.26,
                                          facecolor=img[p[0], p[1]],
                                          edgecolor=T.PANEL_EDGE, lw=1.0, zorder=4))
                  for x, p in ((0.13, pa), (0.61, pb))]
            cax.plot([0.39, 0.61], [0.73, 0.73], color=col,
                     lw=1.0 + 9.0 * wv, solid_capstyle="round", zorder=3)
            cax.text(0.5, 0.40, rf"$\exp(-\beta\|z_m - z_n\|^2) = {wv:.2f}$",
                     transform=cax.transAxes, ha="center", va="center",
                     fontsize=13.5, color=T.TEXT)
            cax.text(0.5, 0.18, verdict, transform=cax.transAxes, ha="center",
                     va="center", fontsize=13.5, color=col)
            for art in cax.texts + cax.lines + sw + [cax.title]:
                art.set_alpha(0)
            for sp in cax.spines.values():
                sp.set_alpha(0)
            cax.patch.set_alpha(0)
            self.cards.append(cax)

    def _card_alpha(self, cax, a):
        for art in cax.texts + cax.lines + cax.patches + [cax.title]:
            art.set_alpha(a)
        for sp in cax.spines.values():
            sp.set_alpha(a)
        cax.patch.set_alpha(a)

    def draw(self, beat, t):
        e = T.smooth(t)
        self.prose.set_alpha(1.0 if beat >= 1 else e)
        self.grid.set_image_alpha(1.0)
        self.veq.set_alpha(1.0 if beat >= 2 else (e if beat == 1 else 0.0))
        for a in (self.vbr1, self.vbr2, self.vn1, self.vn2):
            a.set_alpha(1.0 if beat >= 3 else (e if beat == 2 else 0.0))
        for a in (self.beq, self.bnote):
            a.set_alpha(e if beat == 4 else (1.0 if beat > 4 else 0.0))

        for i, (ln, rings) in enumerate(self.marks):
            want = beat >= 3 + i
            a = e if beat == 3 + i else (1.0 if beat > 3 + i else 0.0)
            # both pairs are shown from beat 3 so the cards can be compared
            if beat >= 3:
                a = 1.0 if beat > 3 else e
            ln.set_alpha(a)
            for rg in rings:
                rg.set_alpha(a)
            self._card_alpha(self.cards[i], a)

    def export_beats(self):
        return {4: "05_smoothness"}


# ===========================================================================
class Notes:
    """
    One note at a time, with step dots underneath.

    An accumulating list looks tidy in a mock-up and then collides with itself
    the moment a body runs to three lines, so only the current note is shown.

    Two slots swap so the outgoing note fades out *before* the incoming one
    fades in. Cross-fading them would overlap two different strings in the same
    place, and simply swapping the text leaves a blank frame.
    """

    OUT_END = 0.40          # old note is gone by here, new one starts after

    def __init__(self, ax, items, y=0.86, x=0.0, size=17):
        self.ax, self.items = ax, items
        self.x, self._y = x, y
        self.slots = []
        for _ in range(2):
            head = ax.text(x, y, "", transform=ax.transAxes, va="center",
                           fontsize=12.5, color=T.GOLD, alpha=0)
            body = ax.text(x, y - 0.075, "", transform=ax.transAxes, va="top",
                           fontsize=size, color=T.TEXT, alpha=0, linespacing=1.72)
            self.slots.append((head, body))
        self.cur = 0
        self.shown = None
        self.dots = []
        for i in range(len(items)):
            d = Circle((x + 0.022 + i * 0.048, y - 0.40), 0.0075,
                       transform=ax.transAxes, facecolor=T.DIM, edgecolor="none",
                       clip_on=False, zorder=5)
            ax.add_patch(d)
            self.dots.append(d)

    def _place(self, slot, alpha, rise):
        head, body = slot
        head.set_alpha(alpha * 0.9)
        body.set_alpha(alpha)
        head.set_position((self.x, self._y + rise))
        body.set_position((self.x, self._y - 0.075 + rise))

    def reveal(self, index, t=1.0):
        index = max(0, min(index, len(self.items) - 1))
        if index != self.shown:
            if self.shown is not None:
                self.cur = 1 - self.cur              # incoming takes the other slot
            head, body = self.slots[self.cur]
            head.set_text(self.items[index][0])
            body.set_text(self.items[index][1])
            self.shown = index

        incoming = self.slots[self.cur]
        outgoing = self.slots[1 - self.cur]
        out_a = 1.0 - T.smooth(min(t / self.OUT_END, 1.0))
        in_t = max(0.0, (t - self.OUT_END) / (1.0 - self.OUT_END))
        in_a = T.smooth(in_t)
        self._place(outgoing, out_a, (1 - out_a) * 0.018)
        self._place(incoming, in_a, (1 - in_a) * -0.018)

        for i, d in enumerate(self.dots):
            d.set_facecolor(T.GOLD if i == index else T.PANEL_EDGE)
            d.set_radius(0.0085 if i == index else 0.0062)


class GraphScene(Scene):
    title = "The image, rebuilt as a graph"
    subtitle = ("One node per pixel, two terminals, and an edge for every "
                "term in the energy")
    n_beats = 6
    beat_seconds = 1.0

    def build(self):
        r = toy_run()
        self.r = r
        h0 = r["history"][0]
        n = r["img"].shape[0]
        self.n = n

        gax = self.add_axes([0.475, 0.055, 0.50, 0.785])
        self.gv = GraphView(gax, r["img"], h0["pairs"])
        self.gv.set_tlink_weights(h0["d_bg"], h0["d_fg"])
        self.gv.set_nlink_weights(h0["nweights"])

        self.pick = 3 * n + 3                      # the one pixel we follow
        self.pick_nlinks = [i for i, (a, b) in enumerate(h0["pairs"])
                            if self.pick in (a, b)]

        ax = self.add_axes([L, 0.07, 0.40, TOPY - 0.07])
        T.blank(ax)
        self.notes = Notes(ax, [
            ("NODES",
             "Every pixel becomes a node, laid out\nin the same grid as the image."),
            ("TERMINALS",
             "Two extra nodes: a source S standing for\n"
             "foreground and a sink T for background."),
            ("t-LINKS  ·  THE DATA TERM",
             "Each pixel gets one edge to each terminal.\n"
             "Their weights are the two data costs\n"
             "D(α=0, …) and D(α=1, …) for that colour."),
            ("t-LINKS EVERYWHERE",
             "You pay for an edge when you cut it - and\n"
             "cutting a pixel's S-edge is exactly what\n"
             "assigns it to the background."),
            ("n-LINKS  ·  THE SMOOTHNESS TERM",
             "Each pixel is also joined to its four\n"
             "neighbours, weighted by the V term."),
            ("THE WHOLE GRAPH",
             "Thick n-links are similar neighbours.\n"
             "The thin ones trace the object's edge -\n"
             "which is exactly where cutting is cheap."),
        ], y=0.83, size=17)

    def draw(self, beat, t):
        e = T.smooth(t)
        gv = self.gv
        self.notes.reveal(beat, t)

        if beat == 0:
            gv.set_node_alpha(e)
            gv.set_terminal_alpha(0)
            gv.set_tlink_alpha(0)
            gv.set_nlink_alpha(0)
            gv.highlight_nodes([], 0)
        elif beat == 1:
            gv.set_node_alpha(1)
            gv.set_terminal_alpha(e)
            gv.set_tlink_alpha(0)
            gv.set_nlink_alpha(0)
        elif beat == 2:
            gv.set_node_alpha(1)
            gv.set_terminal_alpha(1)
            gv.set_tlink_alpha(e, subset=[self.pick])
            gv.set_nlink_alpha(0)
            gv.highlight_nodes([self.pick], e)
        elif beat == 3:
            gv.set_node_alpha(1)
            gv.set_terminal_alpha(1)
            gv.set_tlink_alpha(T.lerp(0.95, 0.42, e))
            gv.set_nlink_alpha(0)
            gv.highlight_nodes([self.pick], 1 - e)
        elif beat == 4:
            gv.set_node_alpha(1)
            gv.set_terminal_alpha(1)
            gv.set_tlink_alpha(0.42 * (1 - 0.55 * e))
            gv.set_nlink_alpha(e, subset=self.pick_nlinks)
            gv.highlight_nodes([self.pick], e)
        else:
            gv.set_node_alpha(1)
            gv.set_terminal_alpha(1)
            gv.set_tlink_alpha(0.19)
            gv.set_nlink_alpha(1.0)
            gv.highlight_nodes([self.pick], 1 - e)

    def export_beats(self):
        return {5: "06_graph"}


# ===========================================================================
class CutScene(Scene):
    title = "One min cut is one complete segmentation"
    subtitle = ("The cheapest way to separate S from T - found exactly, "
                "in one shot, by max flow")
    n_beats = 5
    beat_seconds = 1.0

    def build(self):
        r = toy_run()
        self.r = r
        h = r["history"][0]
        self.h = h
        n = r["img"].shape[0]

        gax = self.add_axes([0.455, 0.055, 0.50, 0.785])
        self.gv = GraphView(gax, r["img"], h["pairs"])
        self.gv.set_tlink_weights(h["d_bg"], h["d_fg"])
        self.gv.set_nlink_weights(h["nweights"])
        self.gv.set_terminal_alpha(1)
        self.gv.set_node_alpha(1)

        ax = self.add_axes([L, 0.07, 0.38, TOPY - 0.07])
        T.blank(ax)
        self.notes = Notes(ax, [
            ("THE QUESTION",
             "Delete a set of edges so that no path\n"
             "from S to T survives. Which set is\n"
             "cheapest?"),
            ("THE CUT",
             "Max flow answers it exactly, in\n"
             "polynomial time. No local minima, no\n"
             "initialisation to get wrong."),
            ("WHERE IT GOES",
             "Notice where the cut runs: through the\n"
             "thin n-links. Those are the neighbours\n"
             "whose colours already disagree."),
            ("THE PAYOFF",
             "Nodes still reachable from S are the\n"
             "foreground. The cut is not a step\n"
             "towards the answer - it is the answer."),
        ], y=0.83, size=17)

        self.eq = ax.text(0.0, 0.115,
                          r"$\min_{\alpha} E(\alpha,\cdot) \;\equiv\; $"
                          r"$\mathrm{min\ cut}(S,T)$",
                          transform=ax.transAxes, fontsize=19, color=T.GOLD,
                          va="center", alpha=0)
        self.cutcap = ax.text(0.0, 0.215, "", transform=ax.transAxes,
                              ha="left", va="center", fontsize=14,
                              color=T.GOLD, alpha=0)

    def seconds(self, beat):
        return 2.1 if beat == 1 else 1.0

    def draw(self, beat, t):
        e = T.smooth(t)
        gv, h = self.gv, self.h
        self.notes.reveal(min(beat, 3), t if beat < 4 else 1.0)

        gv.set_tlink_alpha(0.19)
        gv.set_nlink_alpha(1.0)
        if beat == 0:
            gv.clear_labels()
            gv.show_cut([], 0)
            self.cutcap.set_alpha(0)
            self.eq.set_alpha(0)
        elif beat == 1:
            gv.clear_labels()
            gv.show_cut(h["cut_edges"], t)       # sweeps around the boundary
            self.cutcap.set_alpha(e)
            self.cutcap.set_text(f"{len(h['cut_edges'])} n-links cut  ·  "
                                 f"max flow = {h['flow']:.1f}")
            self.eq.set_alpha(0)
        elif beat == 2:
            gv.clear_labels()
            gv.show_cut(h["cut_edges"], 1.0)
            self.cutcap.set_alpha(1)
            self.eq.set_alpha(0)
            gv.set_nlink_alpha(1.0)
        elif beat == 3:
            gv.show_cut(h["cut_edges"], 1.0)
            gv.show_labels(h["alpha"], strength=e)
            self.cutcap.set_alpha(1)
            self.eq.set_alpha(0)
        else:
            gv.show_cut(h["cut_edges"], 1.0)
            gv.show_labels(h["alpha"], strength=1.0)
            self.cutcap.set_alpha(1)
            self.eq.set_alpha(e)

    def export_beats(self):
        return {4: "07_mincut"}


# ===========================================================================
class LoopScene(Scene):
    """
    The whole algorithm, on one screen, running for real.

    Four panels that update together: the colour model, the graph and its cut,
    the mask, and the energy. The point of the screen is the last panel - E goes
    down at every half-step and never comes back up, because each half-step is
    an exact minimisation of the same E over a different variable.
    """

    title = "The loop: alternate two things you can each solve exactly"
    subtitle = ("Refit the colour models to the current mask, then re-cut the "
                "graph - and repeat")
    beat_seconds = 1.15

    def __init__(self):
        super().__init__()
        self.r = toy_run()
        self.H = self.r["history"]
        self.n_beats = 2 + 2 * len(self.H)

    # -- beat bookkeeping ---------------------------------------------------
    def phase(self, beat):
        """-> ('intro'|'refit'|'cut'|'done', iteration index)."""
        if beat == 0:
            return "intro", 0
        if beat >= 1 + 2 * len(self.H):
            return "done", len(self.H) - 1
        i = (beat - 1) // 2
        return ("refit" if (beat - 1) % 2 == 0 else "cut"), i

    def seconds(self, beat):
        kind, _ = self.phase(beat)
        return {"intro": 0.9, "refit": 1.35, "cut": 1.6, "done": 1.2}[kind]

    def build(self):
        from . import realimg as RI
        r, H = self.r, self.H
        img = r["img"]
        n = img.shape[0]
        self.z = img.reshape(-1, 3)
        i, j = RI.PROJ

        # --- left: which half-step we are in ------------------------------
        ax = self.add_axes([L, 0.07, 0.205, TOPY - 0.07])
        T.blank(ax)
        self.ax = ax
        self.itlab = ax.text(0.0, 0.93, "", transform=ax.transAxes, va="center",
                             fontsize=13, color=T.MUTED)
        self.steps = []
        for k, (name, body) in enumerate((
                ("STEP 1   refit  θ",
                 "Hold the mask still.\nRe-estimate the two\nGMMs from the pixels\n"
                 "each one currently owns."),
                ("STEP 2   min cut",
                 "Hold the models still.\nRebuild the t-links and\ncut the graph again,\n"
                 "exactly."))):
            y = 0.76 - k * 0.315
            head = ax.text(0.0, y, name, transform=ax.transAxes, va="center",
                           fontsize=14, color=T.DIM)
            bar = ax.plot([-0.045, -0.045], [y - 0.20, y + 0.035],
                          transform=ax.transAxes, color=T.PANEL_EDGE, lw=2.6,
                          solid_capstyle="round")[0]
            txt = ax.text(0.0, y - 0.052, body, transform=ax.transAxes, va="top",
                          fontsize=13.5, color=T.DIM, linespacing=1.7)
            self.steps.append((head, bar, txt))
        self.evalue = ax.text(0.0, 0.085, "", transform=ax.transAxes, va="center",
                              fontsize=15, color=T.GOLD)

        # --- centre: the graph --------------------------------------------
        gax = self.add_axes([0.268, 0.055, 0.315, 0.775])
        self.gv = GraphView(gax, img, H[0]["pairs"], node_s=132, grid_h=0.60)
        self.gv.set_nlink_weights(H[0]["nweights"])
        self.gv.set_tlink_weights(H[0]["d_bg"], H[0]["d_fg"])
        self.gv.set_terminal_alpha(1)
        self.gv.set_node_alpha(1)
        self.gv.set_tlink_alpha(0.17)
        self.gv.set_nlink_alpha(1.0)
        self.gv.s_lab.set_text("S")
        self.gv.t_lab.set_text("T")
        self.gv.s_lab.set_alpha(0)
        self.gv.t_lab.set_alpha(0)
        gax.set_title("THE GRAPH AND ITS CUT", color=T.MUTED, fontsize=10.5,
                      loc="center", pad=4)

        # --- right top: colour space --------------------------------------
        sax = self.add_axes([0.622, 0.475, 0.163, 0.315])
        T.panel(sax, "colour model")
        lo = self.z[:, [i, j]].min(0)
        hi = self.z[:, [i, j]].max(0)
        pad = 0.22 * (hi - lo) + 0.02
        sax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        sax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        sax.text(0.965, 0.045, f"{RI.PROJ_NAMES[0]} →", transform=sax.transAxes,
                 ha="right", va="bottom", fontsize=10.5, color=T.DIM,
                 family="DejaVu Sans")
        sax.text(0.075, 0.045, f"{RI.PROJ_NAMES[1]} →", transform=sax.transAxes,
                 ha="left", va="bottom", rotation=90, fontsize=10.5, color=T.DIM,
                 family="DejaVu Sans")
        self.sc = sax.scatter(self.z[:, i], self.z[:, j], s=26,
                              c=[T.SLATE] * len(self.z), zorder=5,
                              edgecolors=T.BG, linewidths=0.7)
        self.ell_f = D.EllipseSet(sax, TOY.K_TOY, T.BLUE, proj=(i, j), zorder=8)
        self.ell_b = D.EllipseSet(sax, TOY.K_TOY, T.SLATE, proj=(i, j), zorder=6)
        self.sax = sax

        # --- right middle: the mask ---------------------------------------
        max_ = self.add_axes([0.815, 0.475, 0.143, 0.315])
        self.grid = D.ImageGrid(max_, img, gap=0.10)
        self.outline = D.mask_outline(max_, H[0]["alpha"], lw=2.4)
        max_.set_title("MASK", color=T.MUTED, fontsize=10.5, loc="center", pad=4)
        self.max_ = max_

        # --- right bottom: energy -----------------------------------------
        eax = self.add_axes([0.622, 0.115, 0.336, 0.245])
        self.chart = D.EnergyChart(eax, len(H), label="")
        eax.set_title("TOTAL ENERGY  E  PER ITERATION", color=T.MUTED,
                      fontsize=10.5, loc="left", pad=10)
        eax.set_xlabel("")
        self.eax = eax
        self.enote = self.fig_text(0.958, 0.068, "", ha="right", va="center",
                                   fontsize=12, color=T.GOLD, alpha=0)

    # -- helpers ------------------------------------------------------------
    def _set_scatter(self, alpha_mask):
        a = np.asarray(alpha_mask).reshape(-1).astype(bool)
        self.sc.set_facecolor([T.BLUE if v else T.SLATE for v in a])

    def _set_gmm(self, pa, pb, t):
        from .gmm import blend_params
        wf, mf, cf, _ = blend_params(pa[0], pb[0], t)
        wb, mb, cb, _ = blend_params(pa[1], pb[1], t)
        self.ell_f.update(wf, mf, cf, alpha=1.0, nsig=1.8)
        self.ell_b.update(wb, mb, cb, alpha=1.0, nsig=1.8)

    def _highlight_step(self, which):
        for k, (head, bar, txt) in enumerate(self.steps):
            on = (k == which)
            head.set_color(T.GOLD if on else T.DIM)
            txt.set_color(T.TEXT if on else D.mix(T.TEXT, T.BG, 0.6))
            bar.set_color(T.GOLD if on else T.PANEL_EDGE)
            bar.set_linewidth(3.0 if on else 2.0)

    def _energy_upto(self, k, frac=1.0):
        xs = [h["it"] for h in self.H[:k]]
        ys = [h["E"] for h in self.H[:k]]
        self.chart.set_data(xs, ys, frac)

    # -- draw ---------------------------------------------------------------
    def draw(self, beat, t):
        e = T.smooth(t)
        H = self.H
        kind, i = self.phase(beat)
        h = H[i]

        if kind == "intro":
            self.itlab.set_text("BEFORE THE FIRST ITERATION")
            self._highlight_step(-1)
            self._set_scatter(h["alpha_before"])
            self._set_gmm(h["gmm_before"], h["gmm_before"], 0.0)
            self.grid.set_mask(h["alpha_before"], strength=e)
            D.set_outline(self.outline, h["alpha_before"])
            self.outline.set_alpha(e)
            self.gv.show_labels(h["alpha_before"], strength=e)
            self.gv.show_cut([], 0)
            self.chart.set_data([], [])
            self.evalue.set_text("")
            self.enote.set_alpha(0)
            return

        self.itlab.set_text(f"ITERATION {h['it']}  of  {len(H)}")

        if kind == "refit":
            self._highlight_step(0)
            self._set_scatter(h["alpha_before"])
            self._set_gmm(h["gmm_before"], h["gmm_after"], e)
            self.grid.set_mask(h["alpha_before"])
            D.set_outline(self.outline, h["alpha_before"])
            self.outline.set_alpha(1)
            self.gv.show_labels(h["alpha_before"], strength=1.0)
            self.gv.show_cut([], 0)
            self._energy_upto(i)
            self.evalue.set_text("")
            self.enote.set_alpha(0)
        elif kind == "cut":
            self._highlight_step(1)
            self._set_gmm(h["gmm_after"], h["gmm_after"], 1.0)
            self.gv.set_tlink_weights(h["d_bg"], h["d_fg"])
            self.gv.show_cut(h["cut_edges"], min(t / 0.62, 1.0))
            late = max(0.0, (t - 0.55) / 0.45)
            lab = h["alpha"] if late > 0.5 else h["alpha_before"]
            self.gv.show_labels(lab, strength=1.0)
            self._set_scatter(lab)
            self.grid.set_mask(lab)
            D.set_outline(self.outline, lab)
            self._energy_upto(i + 1, frac=late)
            self.evalue.set_text(f"E = {h['E']:,.1f}" if late > 0.2 else "")
            self.enote.set_alpha(0)
        else:                                   # done
            self._highlight_step(-1)
            self.itlab.set_text("CONVERGED")
            self._set_gmm(h["gmm_after"], h["gmm_after"], 1.0)
            self._set_scatter(h["alpha"])
            self.grid.set_mask(h["alpha"])
            D.set_outline(self.outline, h["alpha"])
            self.gv.show_labels(h["alpha"], strength=1.0)
            self.gv.show_cut(h["cut_edges"], 1.0)
            self._energy_upto(len(H))
            self.evalue.set_text(f"E = {h['E']:,.1f}")
            self.enote.set_alpha(e)
            self.enote.set_text("E fell at every step, and never rose - "
                                "each half-step exactly minimises the same E.")

    def export_beats(self):
        return {self.n_beats - 1: "08_loop_energy",
                4: "08b_loop_midstep"}


# ===========================================================================
class BridgeScene(Scene):
    title = "From 64 nodes to a quarter of a million"
    subtitle = "Everything so far was the real algorithm - only the image was small"
    n_beats = 4
    beat_seconds = 0.95

    def build(self):
        from . import realimg as RI
        img = RI.load_image()
        h, w = img.shape[:2]
        n_px = h * w
        toy_n = TOY.N_SIDE ** 2

        ax = self.add_axes([L, 0.08, R_ - L, TOPY - 0.08])
        T.blank(ax)
        self.ax = ax

        self.cols = []
        specs = [("THE TOY", f"{toy_n}", "nodes", f"{2 * toy_n - 2 * TOY.N_SIDE}",
                  "n-links", "Edmonds–Karp, written out in\nfull in toy.py - about\n"
                  "forty lines, exact.", T.SLATE),
                 ("A REAL PHOTO", f"{n_px:,}", "nodes",
                  f"{4 * n_px - 3 * (h + w) + 2:,}", "n-links",
                  "Boykov–Kolmogorov max flow\ninside cv2.grabCut - the same\n"
                  "cut, found fast.", T.BLUE)]
        for i, (head, n1, l1, n2, l2, note, col) in enumerate(specs):
            x = 0.16 + i * 0.44
            self.cols.append([
                ax.text(x, 0.885, head, transform=ax.transAxes, ha="center",
                        va="center", fontsize=12.5, color=col, alpha=0),
                ax.text(x, 0.735, n1, transform=ax.transAxes, ha="center",
                        va="center", fontsize=54, color=T.TEXT, alpha=0),
                ax.text(x, 0.625, l1, transform=ax.transAxes, ha="center",
                        va="center", fontsize=14, color=T.MUTED, alpha=0),
                ax.text(x, 0.475, n2, transform=ax.transAxes, ha="center",
                        va="center", fontsize=34, color=T.TEXT, alpha=0),
                ax.text(x, 0.385, l2, transform=ax.transAxes, ha="center",
                        va="center", fontsize=14, color=T.MUTED, alpha=0),
                ax.text(x, 0.225, note, transform=ax.transAxes, ha="center",
                        va="center", fontsize=13.5, color=T.MUTED, alpha=0,
                        linespacing=1.7),
            ])
        self.arrow = FancyArrowPatch((0.300, 0.70), (0.462, 0.70),
                                     transform=ax.transAxes, color=T.GOLD,
                                     arrowstyle="-|>", mutation_scale=22,
                                     lw=2.0, alpha=0, zorder=5)
        ax.add_patch(self.arrow)
        self.arrlab = ax.text(0.381, 0.768, f"× {n_px / toy_n:,.0f}",
                              transform=ax.transAxes, ha="center", va="center",
                              fontsize=15, color=T.GOLD, alpha=0)
        self.kicker = ax.text(0.5, 0.075,
                              "Same energy. Same two alternating steps. Same "
                              "guarantee.\nNow draw a box on a real photograph.",
                              transform=ax.transAxes, ha="center", va="center",
                              fontsize=19, color=T.TEXT, alpha=0, linespacing=1.7)

    def draw(self, beat, t):
        e = T.smooth(t)
        for i, col in enumerate(self.cols):
            on = beat >= i
            cur = beat == i
            for k, art in enumerate(col):
                a = T.stagger(e, k, len(col)) if cur else (1.0 if on else 0.0)
                art.set_alpha(a)
        for a in (self.arrow, self.arrlab):
            a.set_alpha(e if beat == 2 else (1.0 if beat > 2 else 0.0))
        self.kicker.set_alpha(e if beat == 3 else (1.0 if beat > 3 else 0.0))

    def export_beats(self):
        return {3: "09_scale"}
