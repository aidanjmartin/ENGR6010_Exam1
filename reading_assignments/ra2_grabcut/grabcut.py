#!/usr/bin/env python3
"""
GrabCut - a live class demo.

Part A walks through the maths (energy, GMMs, the graph, the min cut and the
alternating loop) on an 8x8 toy image small enough to draw every node and edge.
Part B hands the mouse to the presenter and runs the real thing on a photograph.
Both parts are the same program and share one visual language.

    python grabcut.py                     the whole demo, maths then live tool
    python grabcut.py --image cup.jpg     use your own photo for Part B
    python grabcut.py --live              skip straight to the live tool
    python grabcut.py --export            render the report figures and exit

Controls
    →  /  space      next step            ←   back
    e                save this frame      q   quit
  in the live tool
    drag             draw the box         →   one more iteration
    a                run to convergence   f   final extraction
    m                morphology on/off    r   reset

Reference: Rother, Kolmogorov and Blake, "GrabCut: Interactive Foreground
Extraction using Iterated Graph Cuts", ACM SIGGRAPH 2004.
"""

import argparse
import os
import sys

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def build_scenes(args):
    from grabcut_demo import scenes_a as A
    from grabcut_demo.part_b import LiveScene

    live = LiveScene(image_path=args.image, sample=args.sample)
    if args.live:
        return [live]
    return [
        A.TitleScene(),
        A.SetupScene(),
        A.EnergyScene(),
        A.DataTermScene(),
        A.SmoothScene(),
        A.GraphScene(),
        A.CutScene(),
        A.LoopScene(),
        A.BridgeScene(),
        live,
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="GrabCut live demo (ENGR 6010).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Controls")[1] if "Controls" in __doc__ else None)
    ap.add_argument("--image", metavar="PATH", default=None,
                    help="photo for Part B (default: a bundled scikit-image sample)")
    ap.add_argument("--sample", default=None,
                    help="scikit-image sample name, e.g. astronaut or coffee")
    ap.add_argument("--live", action="store_true",
                    help="skip Part A and go straight to the live tool")
    ap.add_argument("--export", action="store_true",
                    help="render the report figures to out/ and exit")
    ap.add_argument("--export-all", action="store_true",
                    help="render every beat of every scene to out/ (for review)")
    ap.add_argument("--no-gif", action="store_true",
                    help="skip the animated GIFs during --export")
    ap.add_argument("--backend", default=None,
                    help="force a matplotlib backend (macosx, TkAgg, Agg)")
    args = ap.parse_args(argv)

    import matplotlib
    headless = args.export or args.export_all
    matplotlib.use(args.backend or ("Agg" if headless else "macosx"
                                    if sys.platform == "darwin" else "TkAgg"))

    # fail on a bad --image now, not twenty screens into the walkthrough
    if args.image and not os.path.isfile(args.image):
        raise SystemExit(f"no such image: {args.image}")

    from grabcut_demo.scene import Runner

    scenes = build_scenes(args)
    runner = Runner(scenes, OUT_DIR, title="GrabCut - ENGR 6010")

    if headless:
        os.makedirs(OUT_DIR, exist_ok=True)
        print(f"rendering figures to {os.path.relpath(OUT_DIR)}/")
        runner.render_all(every_beat=args.export_all)
        from grabcut_demo.export import export_gifs, export_live_figures
        export_live_figures(runner, args, OUT_DIR)
        if not args.no_gif:
            export_gifs(runner, OUT_DIR)
        print("done.")
        return 0

    print(__doc__.split("Controls")[0].strip())
    print("\nControls" + __doc__.split("Controls")[1])
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
