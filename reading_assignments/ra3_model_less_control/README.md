# When the Model Is Wrong

An interactive in-class demo for ENGR 6010 (AI in Robotics), built on

> M. C. Yip and D. B. Camarillo, "Model-Less Feedback Control of Continuum
> Manipulators in Constrained Environments," *IEEE Transactions on Robotics*
> 30(4), 880–889, 2014. https://doi.org/10.1109/TRO.2014.2309194

The one idea: **when your model of a system is wrong, estimate the model from
what you observe.** Make the smallest correction that explains the newest
measurement, and let a small optimization problem choose each step.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.10+
pip install -r requirements.txt
python main.py
```

| Flag | Effect |
|---|---|
| `--windowed` | 1600 × 900 resizable window (default) |
| `--fullscreen` | full screen at desktop resolution |
| `--act N` | start at act N (0–4) |
| `--screenshots OUTDIR` | render every beat headless, verify the layout, exit |

Everything is drawn on one 1920 × 1080 virtual canvas and scaled uniformly into
the window, with letterbox bars in the background colour. Windowed and full
screen show the same layout, just at a different size. Press `F` at any time.

## Presenter cheat sheet

Nothing advances on a timer. **Space / →** is the only thing that moves you on;
**←** goes back; **R** replays the current beat. About 6 to 8 minutes in total.

| Beat | On screen | Say / do |
|---|---|---|
| **0 · Title** | Robot draws itself, reaches a target | Introduce the paper. |
| **1.1 The bowl** | γ = 1: both methods hit the centre | "On a round bowl, following the slope works." |
| **1.2 The ravine** | Bowl stretches 1 → 25, then both rerun | Gradient descent zigzags (40 steps), Newton takes 1. Drag the **γ slider** (1 to 50) to rerun. **G** replays the stretch. |
| **1.3 The fix and the catch** | The two update rules; H⁻¹ lights up | "Newton needs H. What if you cannot compute it?" |
| **2.1 Secant** | Second point slides in; secant turns toward the tangent | Two observations give a slope estimate: Δg / Δx. |
| **2.2 BFGS** | Estimate ellipse (solid) morphs onto true curvature (dashed) | B s = y. Center in 2 steps vs 40. |
| **2.3 Same trick** | *Space* reveals row 2, then Broyden's update | Hessian ↔ Jacobian. Prediction error term; the predicted arrow rotates onto the measured one. |
| **3.1 Meet the robot** | Each actuator moves in turn; its column arrow appears | Insertion, left tendon, right tendon. Δx ≈ J Δy. |
| **3.2 Wiggle** | Each actuator twitches; estimated column grows | Initial estimate Ĵᵢ = Δx / Δyᵢ. |
| **3.3 The race** | Both robots, straight endoscope | **Click** anywhere in a panel to set a target (appears in both). **T**: square trajectory. Both work. |
| **3.4 Bend** | Square trajectory already running | Press **B** (or drag φ). Past 90° the model-based robot runs away: *positive feedback* warning. The model-less robot's arrows swing onto the dashed truth. **J** toggles the dashed truth arrows. |
| **3.5 Noise & smoothing** | Noise on (σ = 0.6 mm), sliders for σ, α, threshold, tendon friction | Raise **α** toward 1: fast but jittery. Lower it to 0.1 and press **B**: smooth but lags after the bend. **N** toggles noise. |
| **4 · Close** | *Space* reveals takeaways 2 and 3 | References on the card. |

Other keys: **1–4** jump to an act, **H** help overlay, **I** hide the beat
indicator, **Esc** quit.

## What is simulated

* **Acts 1–2** (`sim/optim.py`): gradient descent with exact line search,
  Newton, and BFGS (B₀ = I, exact line search) on f(x) = ½(x₁² + γx₂²).
* **Robot** (`sim/robot.py`): planar constant-curvature instrument leaving a
  bendable endoscope. Actuators are insertion and two antagonistic tendons with
  tendon friction g (the true g is 0.85; the model assumes 1). Tension
  τ = τ₀ + k·y (paper eq. 8), τ_min = 0.3 N. Actuators follow commands with a
  35 ms lag. Tip measurements carry optional Gaussian noise.
* **Controllers** (`sim/controller.py`): both run the same control step, paper
  eq. 9: minimise ‖τ + KΔy‖² subject to ĴWΔy = Δx_d, every tendon ≥ τ_min, and
  step and actuator limits. It is solved exactly in closed form in about 25 µs.
  The model-based controller uses the model's analytic Jacobian (φ = 0, g = 1).
  The model-less controller starts from the wiggle estimate and applies
  Broyden's update (paper eq. 10) with smoothing α and a movement threshold.

Design choices beyond the paper, all visible in the code:

* The square trajectory pauses while the endoscope is actually bending. The
  surgeon holds still, so targets stay fixed in the endoscope frame. The paper
  assumes a static environment, and a moving one corrupts any secant update.
* The Jacobian update is skipped when the tip moved far more than the actuator
  motion could explain (again, the endoscope moving underneath).
* If the estimate claims no step is feasible, the controller takes a bounded
  damped-least-squares step. That keeps the robot moving, so the estimate can
  repair itself. The on-screen tension bars show the result.

As the paper says, there is no stability proof. The α filter and the backward
differences add lag, and beat 3.5 shows exactly that.

## Tests and verification

```bash
python -m pytest -q                        # 69 tests, about 10 s
python main.py --screenshots out/          # every beat at six window sizes
```

* `tests/test_optim.py`: GD contracts by (γ−1)/(γ+1) per step; Newton converges
  in 1 step; BFGS in at most 3, and the secant condition holds after every update.
* `tests/test_controller.py`: Broyden meets Ĵ_new Δy = Δx exactly and is the
  smallest-norm correction; the tension solve meets the equality to 1e-6 and
  τ ≥ τ_min; closed-loop behaviour at φ = 0°, 60° and 120° (both converge; the
  model-based controller is slower; the model-based one fails while the
  model-less one converges).
* `tests/test_layout.py`: every beat keeps registered text and UI 40 px inside
  the canvas with no overlapping text blocks.
* `tests/test_interaction.py`: every key and mouse path on every beat, and a
  60 fps frame budget (update + draw + present).
* The screenshot harness presents each beat into 1280×720, 1600×900, 1920×1080,
  2560×1440, 1920×1200 (letterboxed) and 1366×768 through the real present
  step. It crops each to its content area, scales back to 1920×1080 and
  compares against the virtual canvas. A control shifted by 6 px scores 2 to 4
  times higher than the threshold, so the check would catch a real misalignment.

## Layout

```
main.py                  entry point, main loop, beat navigation, transitions
core/  canvas.py         virtual canvas, present step, letterbox, input mapping
       theme.py          palette, type scale, regions
       text.py           UI font, cached text, layout recorder
       mathtext.py       TeX -> cached surfaces with per-term colour
       anim.py draw.py widgets.py
acts/  act0_title.py … act4_close.py, robotdraw.py
sim/   optim.py robot.py controller.py
tools/ screenshot_all.py
tests/
```

Colour means the same thing in every act: amber = gradient / first order,
blue = curvature / second order, green = estimate, warm white dashed = truth,
pink = assumed model, red with a ⚠ = warning, violet = what you changed (s, Δy).
