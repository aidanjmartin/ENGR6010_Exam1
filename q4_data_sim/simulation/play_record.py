"""Play back a trained unitree_rl_gym Go2 policy headless, record a GIF, measure tracking.

This follows unitree_rl_gym's legged_gym/scripts/play.py: at most 100 environments,
no observation noise, no friction randomization, no pushes. It differs in four ways:

- The command is fixed instead of random: walk forward at --vx m/s, no sideways
  velocity, heading 0. Every environment gets the same command.
- It runs headless but turns rendering on for one offscreen camera attached to
  the body of environment 0. Isaac Gym turns rendering off entirely when headless, so this
  script sets the graphics device back to the simulation device before the
  simulation is created.
- It runs a fixed number of policy steps and exits.
- It writes a GIF of environment 0 and a CSV of commanded and measured forward
  velocity, and prints tracking error and the number of falls.

Usage (from the unitree_rl_gym folder, with this environment active):
    python <repo>/q4_data_sim/simulation/play_record.py --task=go2 --load_run=<run> --seed=1
"""
import os
import sys

import isaacgym  # noqa: F401  (must be imported before torch)
from isaacgym import gymapi
from legged_gym.envs import *  # noqa: F401,F403  (registers the tasks)
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils import get_args, task_registry

import csv
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

# Settings that play.py does not take as arguments.
VX = float(os.environ.get("PLAY_VX", 1.0))       # commanded forward velocity, m/s
SECONDS = float(os.environ.get("PLAY_SECONDS", 10.0))
OUT = os.environ.get("PLAY_OUT", os.path.join(HERE, "results"))
FPS = 25
WIDTH, HEIGHT = 480, 320


_create_sim = LeggedRobot.create_sim


def create_sim_with_rendering(self):
    self.graphics_device_id = self.sim_device_id  # headless, but keep offscreen cameras
    _create_sim(self)


LeggedRobot.create_sim = create_sim_with_rendering


def main(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = True
    env_cfg.commands.ranges.lin_vel_x = [VX, VX]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0.0, 0.0]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    train_cfg.runner.resume = True
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)

    gym, sim = env.gym, env.sim
    props = gymapi.CameraProperties()
    props.width, props.height = WIDTH, HEIGHT
    cam = gym.create_camera_sensor(env.envs[0], props)
    # Follow the body's position (not its rotation) from 2 m behind-right, 0.5 m up,
    # looking at the body: yaw turns the camera's +x axis toward it, pitch tilts it down.
    offset = np.array([1.2, -1.6, 0.5])
    yaw = np.arctan2(-offset[1], -offset[0])
    pitch = np.arctan2(offset[2], np.hypot(offset[0], offset[1]))
    mount = gymapi.Transform(gymapi.Vec3(*offset), gymapi.Quat.from_euler_zyx(0.0, pitch, yaw))
    base = gym.get_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], 0)
    gym.attach_camera_to_body(cam, env.envs[0], base, mount, gymapi.FOLLOW_POSITION)

    steps = int(round(SECONDS / env.dt))
    frame_every = max(1, int(round(1.0 / (FPS * env.dt))))
    frames, rows, falls = [], [], 0
    for k in range(steps):
        with torch.no_grad():
            actions = policy(obs.detach())
        obs, _, _, dones, _ = env.step(actions.detach())
        falls += int((dones & ~env.time_out_buf).sum())
        vx = env.base_lin_vel[:, 0].cpu().numpy()
        rows.append((round((k + 1) * env.dt, 3), VX, round(float(vx[0]), 4), round(float(vx.mean()), 4)))
        if k % frame_every == 0:
            gym.fetch_results(sim, True)  # legged_gym skips this on the GPU pipeline
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
            img = gym.get_camera_image(sim, env.envs[0], cam, gymapi.IMAGE_COLOR)
            frames.append(Image.fromarray(img.reshape(HEIGHT, WIDTH, 4)[:, :, :3]))

    os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, f"play_go2_vx{VX:g}")
    frames[0].save(f"{stem}.gif", save_all=True, append_images=frames[1:],
                   duration=int(1000 / FPS), loop=0, optimize=True)
    with open(f"{stem}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "vx_command", "vx_env0", "vx_mean_all_envs"])
        w.writerows(rows)

    settled = np.array([r[3] for r in rows if r[0] >= 2.0])  # skip the first 2 s
    env0 = np.array([r[2] for r in rows if r[0] >= 2.0])
    print(f"play: {env.num_envs} environments, {SECONDS:g} s, command vx = {VX:g} m/s")
    print(f"play: measured vx after 2 s, mean over environments: {settled.mean():.3f} m/s; "
          f"environment 0: {env0.mean():.3f} +/- {env0.std():.3f} m/s")
    print(f"play: falls (episode ends before time-out): {falls}")
    print(f"play: wrote {stem}.gif ({len(frames)} frames) and {stem}.csv")


if __name__ == "__main__":
    args = get_args()
    args.headless = True
    main(args)
