"""Run the unitree_mujoco Go2 simulator for the SDK examples, with optional recording.

This wraps unitree_mujoco/simulate_python and keeps its model and its SDK bridge
(the code that turns MuJoCo state into LowState and SportModeState messages and
applies LowCmd messages) unchanged. It differs from the stock unitree_mujoco.py in
four settings:

- DDS domain 0 on the loopback interface, the default of the SDK examples, so the
  examples run with the single argument `lo`. The stock simulator uses domain 1.
- No gamepad. The stock simulator exits its physics thread when it finds none.
- A fixed run time, after which the process exits.
- Optional offscreen recording to a GIF, a PNG of the last frame, and a CSV of the
  body height, so a run leaves files behind instead of only a window.

MuJoCo physics is deterministic and nothing here draws random numbers, but the
SDK example runs as a separate process, so the exact step at which each command
arrives varies from run to run by a few milliseconds.

Usage:
    python sim_go2.py --duration 12 --gif output/stand.gif --sheet output/stand.png --csv output/stand.csv
    python sim_go2.py --viewer            # interactive MuJoCo window instead
"""
import argparse
import csv
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "glfw")  # EGL fails on this machine; glfw works offscreen

import mujoco
import mujoco.viewer
from PIL import Image, ImageDraw

UNITREE_MUJOCO = os.environ.get("UNITREE_MUJOCO", os.path.expanduser("~/go2_deps/unitree_mujoco"))
sys.path.insert(0, os.path.join(UNITREE_MUJOCO, "simulate_python"))

import config  # unitree_mujoco's config module; the bridge reads ROBOT from it at import

config.ROBOT = "go2"
config.USE_JOYSTICK = 0

from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: E402
from unitree_sdk2py_bridge import UnitreeSdk2Bridge  # noqa: E402

SCENE = os.path.join(UNITREE_MUJOCO, "unitree_robots", "go2", "scene.xml")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--duration", type=float, default=20.0, help="seconds of simulated time")
    p.add_argument("--domain", type=int, default=0, help="DDS domain id")
    p.add_argument("--interface", default="lo", help="network interface for DDS")
    p.add_argument("--viewer", action="store_true", help="open the interactive MuJoCo viewer")
    p.add_argument("--gif", help="write an offscreen recording to this GIF")
    p.add_argument("--png", help="write the last recorded frame to this PNG")
    p.add_argument("--sheet", help="write frames at --sheet-times side by side to this PNG")
    p.add_argument("--sheet-times", default="1,3.5,5,7.5", help="seconds, comma separated")
    p.add_argument("--csv", help="write time and body height to this CSV")
    p.add_argument("--fps", type=int, default=20, help="GIF frame rate")
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=360)
    return p.parse_args()


def make_camera(model):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = model.body("base_link").id
    cam.distance = 1.6
    cam.azimuth = 135.0
    cam.elevation = -20.0
    return cam


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    model.opt.timestep = config.SIMULATE_DT

    ChannelFactoryInitialize(args.domain, args.interface)
    UnitreeSdk2Bridge(model, data)  # starts its own publisher threads
    print(f"sim_go2: Go2 on DDS domain {args.domain}, interface {args.interface}, "
          f"dt {model.opt.timestep} s, {args.duration} s", flush=True)

    recording = bool(args.gif or args.png or args.sheet)
    renderer = mujoco.Renderer(model, args.height, args.width) if recording else None
    camera = make_camera(model) if recording else None
    viewer = mujoco.viewer.launch_passive(model, data) if args.viewer else None

    frames, rows = [], []
    steps = int(round(args.duration / model.opt.timestep))
    frame_every = max(1, int(round(1.0 / (args.fps * model.opt.timestep))))
    log_every = max(1, int(round(0.02 / model.opt.timestep)))
    wall_start = time.perf_counter()

    for k in range(steps):
        step_start = time.perf_counter()
        mujoco.mj_step(model, data)
        if k % log_every == 0:
            rows.append((round(data.time, 4), round(float(data.qpos[2]), 5)))
        if recording and k % frame_every == 0:
            renderer.update_scene(data, camera)
            frames.append(Image.fromarray(renderer.render()))
        if viewer is not None:
            if not viewer.is_running():
                break
            if k % 4 == 0:
                viewer.sync()
        # Run in real time so the SDK example, a separate process, sees normal timing.
        remaining = model.opt.timestep - (time.perf_counter() - step_start)
        if remaining > 0:
            time.sleep(remaining)

    wall = time.perf_counter() - wall_start
    print(f"sim_go2: {data.time:.2f} s simulated in {wall:.2f} s wall time; "
          f"body height start {rows[0][1]:.3f} m, end {data.qpos[2]:.3f} m", flush=True)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "base_height_m"])
            w.writerows(rows)
    if args.gif and frames:
        frames[0].save(args.gif, save_all=True, append_images=frames[1:],
                       duration=int(1000 / args.fps), loop=0, optimize=True)
    if args.png and frames:
        frames[-1].save(args.png)
    if args.sheet and frames:
        times = [float(t) for t in args.sheet_times.split(",")]
        sheet = Image.new("RGB", (args.width * len(times), args.height))
        for i, t in enumerate(times):
            tile = frames[min(int(round(t * args.fps)), len(frames) - 1)].copy()
            draw = ImageDraw.Draw(tile)
            draw.rectangle([0, 0, 90, 22], fill="black")
            draw.text((6, 5), f"t = {t:.1f} s", fill="white")
            sheet.paste(tile, (i * args.width, 0))
        sheet.save(args.sheet)
    # The bridge's publisher threads do not stop on their own.
    os._exit(0)


if __name__ == "__main__":
    main()
