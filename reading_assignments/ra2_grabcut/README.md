# GrabCut — a live demo

A single Python program that first explains how GrabCut works and then runs it,
for real, on a photograph the presenter draws a box on.

> Rother, Kolmogorov and Blake, *"GrabCut: Interactive Foreground Extraction
> using Iterated Graph Cuts"*, ACM SIGGRAPH 2004.

Both halves are the same application and share one visual language, so moving
from the maths to the live tool is one continuous demo rather than two programs.

## Run it

```bash
pip install -r requirements.txt
python grabcut.py
```

No other setup. The photograph for Part B is a bundled `scikit-image` sample, so
the demo works offline on a machine it has never seen before.

```bash
python grabcut.py                      # the whole demo: maths, then live tool
python grabcut.py --image cup.jpg      # your own photo for Part B
python grabcut.py --sample coffee      # a different scikit-image sample
python grabcut.py --live               # skip Part A, go straight to the tool
python grabcut.py --export             # render the report figures and exit
```

## Controls

Everything is driven by the keyboard, and nothing is on a timer — a step holds
until you advance it, so the demo can be paused mid-sentence to take a question.

| key | in the walkthrough |
|---|---|
| `→` or `space` | next step (press again mid-animation to snap it to the end) |
| `←` | previous step |
| `e` | save the current frame to `out/` |
| `q` | quit |

| key | in the live tool |
|---|---|
| drag with the mouse | draw the bounding box; releasing starts iteration 1 |
| `→` | run one more iteration |
| `a` | run to convergence |
| `f` | final foreground extraction |
| `m` | toggle the morphological cleanup (after `f`) |
| `r` | reset and draw a new box |
| `←` | back to the maths |

## What the two parts show

**Part A — the maths.** An 8×8 toy image, small enough to draw every node and
every edge of its graph.

1. **Setup** — pixels `z`, labels `α`, and the trimap a bounding box induces.
2. **Energy** — `E(α,k,θ,z) = U(α,k,θ,z) + V(α,z)` built up term by term.
3. **Data term** — the full `D(·)` expression, beside a real photograph's pixels
   scattered in RGB with the five Gaussian components per class drawn as
   covariance ellipses.
4. **Smoothness term** — `V` in plain words first, then the equation, with two
   real neighbouring pairs from the toy priced side by side (0.99 inside a flat
   region, 0.04 across the object's edge) and `β` read off the image's own
   average contrast.
5. **Graph construction** — nodes coloured by their own pixel, terminals `S` and
   `T`, t-links weighted by the data term and n-links by the smoothness term,
   with edge width and opacity carrying the weight.
6. **Min cut** — the cheapest S–T separation sweeps around the boundary, and it
   runs through the *thin* n-links, which is the whole point.
7. **The loop** — the colour model, the graph, the mask and the energy on one
   screen, advancing a half-step at a time. `E` falls at every step and never
   rises, because each half-step exactly minimises the same `E` over a different
   variable.
8. **Bridge** — 64 nodes versus 262,144, and why Part B calls OpenCV.

**Part B — the live tool.** `cv2.grabCut` with `GC_INIT_WITH_RECT`, called with
`iterCount=1` inside a loop so each iteration can be shown as it happens: the
mask overlaid on the photo, the colour model re-fitting, the pixels-changed
count falling, and the energy curve extending.

## Honest notes on the numbers

- The toy's max flow is an exact Edmonds–Karp implementation in `toy.py`, not a
  heuristic, so the monotonically decreasing energy plot is a result rather than
  a drawing.
- OpenCV does not expose its internal energy, so Part B's `E` is **recomputed**
  from each mask using the same GMM and smoothness term as Part A. The panel
  says so on screen. The primary convergence signal there is the number of
  pixels that changed label.
- The morphological open/close pass is *not* part of GrabCut. It is a separate,
  toggleable step, included because the ImageJ visualiser motivates it.
- The GMMs live in RGB, as in the paper and in `cv2.grabCut`. RGB ties
  brightness to hue, so a shadowed part of an object can sit a whole mixture
  component away from its lit part; Lab or HSV separate those more cleanly.

## Figures

`python grabcut.py --export` writes these to `out/`, for reuse in the report:

| file | what it is |
|---|---|
| `01_title.png` | title, with the toy image and the cut GrabCut finds |
| `02_trimap.png` | the bounding box and the trimap it induces |
| `03_energy.png` | the energy function, with both terms annotated |
| `04_gmm_colour_space.png` | the data term beside the RGB scatter and its ellipses |
| `05_smoothness.png` | the smoothness term, with two neighbouring pairs priced |
| `06_graph.png` | the toy image as an explicit s–t graph |
| `07_mincut.png` | the min cut, and the labelling it *is* |
| `08_loop_energy.png` | the full loop, with energy decreasing monotonically |
| `08b_loop_midstep.png` | one half-step of the loop, mid-iteration |
| `09_scale.png` | toy versus real photograph, by node count |
| `10_live_converged.png` | the live tool at convergence |
| `11_final_extraction.png` | the foreground pulled out |
| `12_final_extraction_morphology.png` | the same, with the cleanup on |
| `13_morphology_compare.png` | raw mask, cleaned mask, and what moved |

`--export-all` additionally renders every beat of every screen, which is useful
for checking layouts after an edit.

## Layout

```
grabcut.py              entry point: argument parsing, scene list, export mode
grabcut_demo/
  theme.py              palette, typography, axis styling, easing
  scene.py              Scene/Runner: beats, the clock, the presenter HUD
  draw.py               shared primitives: equations, ellipses, pixel grids, charts
  graphview.py          the toy image drawn as an s-t graph
  gmm.py                the paper's hard-assignment GMM
  toy.py                the toy instance: image, graph weights, exact max flow
  realimg.py            photo loading, energy re-evaluation, the stepped session
  scenes_a.py           Part A: every screen of the walkthrough
  part_b.py             Part B: the live interactive tool
  export.py             headless figure rendering
out/                    exported figures
```

Scenes build their artists once and then only mutate them — alpha, position,
colour — which is what keeps the animation smooth with a few hundred edges on
screen.
