"""
Run the five Q4 synthetic-data scripts and collect their tables.

Each script runs in its own process with the same Python interpreter. Its
printed output, which is Markdown, goes into q4_data_sim/results/SUMMARY.md
under one heading per script, with the wall-clock time of the run. The run
stops at the first script that fails.

Run:  python run_all.py
Out:  results/SUMMARY.md, plus every figure the scripts save
"""

import os
import platform
import subprocess
import sys
import time

import common as C

SCRIPTS = [
    ("color_space.py", "Color spaces", "Ibraheem et al. [1]"),
    ("gaussian_pyramid.py", "Gaussian pyramids", "Szeliski Sec. 9.1.1 [2]"),
    ("optical_flow.py", "Optical flow", "Szeliski Ch. 9 [2]"),
    ("grabcut.py", "GrabCut", "Rother et al. [5]; Peng et al. [6]"),
    ("supervised_learning.py", "Supervised learning",
     "Kotsiantis [7]; Bottou et al. [8]"),
]


def main():
    sections = []
    total = 0.0
    for script, title, refs in SCRIPTS:
        print(f"running {script} ...", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, script], cwd=C.SYNTHETIC_DIR,
                              capture_output=True, text=True)
        dt = time.perf_counter() - t0
        total += dt
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout + proc.stderr)
            raise SystemExit(f"{script} failed with exit code {proc.returncode}")
        print(f"  done in {dt:.1f} s")
        sections.append(f"## {title}\n\n"
                        f"Script: `synthetic/{script}`. Reading: {refs}. "
                        f"Run time: {dt:.1f} s.\n"
                        f"{proc.stdout.rstrip()}\n")

    header = (
        "# Q4 synthetic-data results\n\n"
        "`synthetic/run_all.py` generates this file. Every number below comes "
        "from synthetic data with exact ground truth and a fixed random seed "
        f"({C.SEED}), so a rerun reproduces it. Figures are in this folder.\n\n"
        f"Environment: Python {platform.python_version()} on "
        f"{platform.system()} {platform.machine()}. Total run time: "
        f"{total:.1f} s.\n\n"
        "References follow the numbering of the Q1 review paper.\n"
    )
    path = os.path.join(C.RESULTS_DIR, "SUMMARY.md")
    with open(path, "w") as f:
        f.write(header + "\n" + "\n".join(sections))
    print(f"\nwrote {path} (total {total:.1f} s)")


if __name__ == "__main__":
    main()
