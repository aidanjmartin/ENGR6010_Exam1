"""Record the two control loops of Go2Py's examples/02-MuJoCo-sim.ipynb offscreen.

The notebook shows its runs only in live windows. This script repeats its two loops
with the same commands and gains, renders them offscreen, and writes a GIF and a CSV
of body height, so the result stays in the repository:

1. Low level: standUpReset(), then 10 s of joint torque tau = 20 (q0 - q).
2. High level: Go2Sim(mode='highlevel'), standUpReset(), then 10 s of
   step(0, 0, 0, step_height=0, kp=[2, 0.5, 0.5], ki=[0.02, 0.01, 0.01]), the
   walk-these-ways policy tracking zero body velocity.

The notebook times its loops by wall clock. This script runs a fixed number of steps
(10 s / dt) instead, so every run produces the same frames. Seeds are fixed for NumPy
and PyTorch.

Usage:
    python go2py_record.py                      # writes output/go2py_*.gif and .csv
"""
import argparse
import csv
import os

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
import numpy as np
import torch
from PIL import Image

from Go2Py.sim.mujoco import Go2Sim

HERE = os.path.dirname(os.path.abspath(__file__))


def make_camera(model):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = model.body("base").id
    cam.distance = 1.6
    cam.azimuth = 135.0
    cam.elevation = -20.0
    return cam


def run(robot, step_fn, seconds, fps, size):
    renderer = mujoco.Renderer(robot.model, size[1], size[0])
    cam = make_camera(robot.model)
    steps = int(round(seconds / robot.dt))
    frame_every = max(1, int(round(1.0 / (fps * robot.dt))))
    log_every = max(1, int(round(0.02 / robot.dt)))
    frames, rows = [], []
    for k in range(steps):
        step_fn()
        if k % log_every == 0:
            rows.append((round(k * robot.dt, 4), round(float(robot.data.qpos[2]), 5)))
        if k % frame_every == 0:
            renderer.update_scene(robot.data, cam)
            frames.append(Image.fromarray(renderer.render()))
    renderer.close()
    return frames, rows


def save(frames, rows, stem, fps):
    frames[0].save(f"{stem}.gif", save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    with open(f"{stem}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "base_height_m"])
        w.writerows(rows)
    heights = [r[1] for r in rows]
    print(f"{os.path.basename(stem)}: {len(frames)} frames, body height "
          f"start {heights[0]:.3f} m, end {heights[-1]:.3f} m, "
          f"mean over last 5 s {np.mean(heights[len(heights) // 2:]):.3f} m")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--out", default=os.path.join(HERE, "output"))
    args = p.parse_args()
    np.random.seed(0)
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)
    size = (400, 300)

    # Loop 1: low level, as in notebook cell 6.
    robot = Go2Sim(mode="lowlevel", render=False)
    robot.standUpReset()

    def lowlevel_step():
        state = robot.getJointStates()
        tau = 20 * np.eye(12) @ (robot.q0 - state["q"]).reshape(12, 1)
        robot.setCommands(np.zeros(12), np.zeros(12), np.zeros(12), np.zeros(12), tau)
        robot.step()

    frames, rows = run(robot, lowlevel_step, args.seconds, args.fps, size)
    save(frames, rows, os.path.join(args.out, "go2py_lowlevel"), args.fps)

    # Loop 2: high level, as in notebook cells 9 and 10.
    robot = Go2Sim(mode="highlevel", render=False)
    robot.standUpReset()

    def highlevel_step():
        robot.step(0, 0, 0., step_height=0, kp=[2, 0.5, 0.5], ki=[0.02, 0.01, 0.01])

    frames, rows = run(robot, highlevel_step, args.seconds, args.fps, size)
    save(frames, rows, os.path.join(args.out, "go2py_highlevel"), args.fps)


if __name__ == "__main__":
    main()
