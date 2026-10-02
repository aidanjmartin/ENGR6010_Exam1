"""Plot the training curves of a unitree_rl_gym run from its TensorBoard log.

Reads the event file that rsl_rl writes during training and draws three panels,
each with its own axis: mean episode reward, mean episode length, and the
velocity-tracking reward term. Also writes the values to CSV.

Usage:
    python plot_training.py ~/go2_deps/unitree_rl_gym/logs/rough_go2/<run>
"""
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

PANELS = [
    ("Train/mean_reward", "Mean episode reward"),
    ("Train/mean_episode_length", "Mean episode length (policy steps)"),
    ("Episode/rew_tracking_lin_vel", "Linear-velocity tracking reward per episode"),
]
SERIES = "#2a78d6"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def main(run_dir):
    event_file = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))[-1]
    acc = EventAccumulator(event_file, size_guidance={"scalars": 0})
    acc.Reload()
    data = {tag: [(e.step, e.value) for e in acc.Scalars(tag)] for tag, _ in PANELS}

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "training_curve.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration"] + [tag for tag, _ in PANELS])
        steps = [s for s, _ in data[PANELS[0][0]]]
        lookup = {tag: dict(v) for tag, v in data.items()}
        for s in steps:
            w.writerow([s] + [round(lookup[tag].get(s, float("nan")), 5) for tag, _ in PANELS])

    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2})
    fig, axes = plt.subplots(1, len(PANELS), figsize=(12, 3.4), facecolor="#fcfcfb")
    for ax, (tag, title) in zip(axes, PANELS):
        s, v = zip(*data[tag])
        ax.set_facecolor("#fcfcfb")
        ax.plot(s, v, color=SERIES, linewidth=2)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("training iteration")
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.annotate(f"{v[-1]:.2f}", (s[-1], v[-1]), textcoords="offset points",
                    xytext=(-4, 6), ha="right", color=INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "training_curve.png"), dpi=150)
    for tag, title in PANELS:
        first, last = data[tag][0][1], data[tag][-1][1]
        print(f"{title}: iteration {data[tag][0][0]} {first:.3f} -> iteration {data[tag][-1][0]} {last:.3f}")


if __name__ == "__main__":
    main(os.path.expanduser(sys.argv[1]))
