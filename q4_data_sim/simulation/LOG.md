# Q4 simulation log

Machine: lab desktop, Intel Core Ultra 9 285K, 60 GB RAM, NVIDIA RTX 4000 Ada Generation (20 GB, compute capability 8.9), driver 595.91.07 (CUDA 13.2), Ubuntu 26.04.1 LTS, kernel 7.0.0-34. Machine setup (system packages, Miniconda, upstream clones) is in [`../../q2_sdk/LOG.md`](../../q2_sdk/LOG.md).
Date: 2026-10-02.

## Compatibility of Isaac Gym with this computer

Isaac Gym Preview 4 (2022) predates NVIDIA's Blackwell GPUs, such as the RTX 5060 Ti in the home computer this phase was first planned on. This lab computer has an RTX 4000 Ada instead. Ada GPUs (compute capability 8.9) shipped in late 2022, and PyTorch's CUDA 12.1 builds support them. NVIDIA lists Ubuntu 18.04 and 20.04 for Isaac Gym, and this computer runs 26.04.

Result: **Isaac Gym runs here.** Its Python 3.8 binary (`gym_38.so`) loads, and GPU PhysX simulates on the RTX 4000 Ada:

| version | value |
| :--- | :--- |
| NVIDIA driver | 595.91.07 (supports CUDA up to 13.2) |
| CUDA runtime in PyTorch | 12.1 |
| PyTorch | 2.3.1+cu121 |
| Python | 3.8 (conda-forge) |
| Isaac Gym | Preview 4 |

## Environment

The steps follow `unitree_rl_gym/doc/setup_en.md`.

| step | command | outcome |
| :--- | :--- | :--- |
| create environment | `conda create -p q4_data_sim/simulation/.venv --override-channels -c conda-forge python=3.8 pip` | success: Python 3.8 |
| install PyTorch | `pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121` | success: torch 2.3.1+cu121 |
| check PyTorch on the GPU | `python -c "torch.cuda.is_available(); x @ x on cuda"` | success: device "NVIDIA RTX 4000 Ada Generation", matrix product on the GPU |
| install rsl_rl | `git clone -b v1.0.2 https://github.com/leggedrobotics/rsl_rl.git && pip install -e rsl_rl` | success: commit `2ad79cf`. Git warns that tag `v1.0.2` is an annotated tag object; the checkout is correct. |
| install Isaac Gym | `pip install -e ~/Downloads/IsaacGym_Preview_4_Package/isaacgym/python` | success: also installs imageio 2.35.1, ninja 1.13.2, scipy 1.10.1 |
| install unitree_rl_gym | `pip install -e ~/go2_deps/unitree_rl_gym` | success: numpy 1.20.0, mujoco 3.2.3, matplotlib 3.7.5, tensorboard 2.14.0; `pip check` finds no conflicts |
| first Isaac Gym test | import `isaacgym`, `gymtorch`; GPU PhysX sim with one falling box | fails: `RuntimeError: Ninja is required to load C++ extensions`; see Problems |
| second Isaac Gym test | same, with `.venv/bin` on `PATH` and `.venv/lib` on `LD_LIBRARY_PATH` | success: "Using GPU PhysX", "GPU Pipeline: enabled", the 0.2 m box comes to rest at height 0.102 m, state tensor on `cuda:0` |
| pin Pillow | `pip install pillow==9.5.0` | success; see Problems |

`requirements.txt` lists the final versions. [`run_go2.sh`](run_go2.sh) sets both search paths and runs every step below.

## No Go2 path to MuJoCo in unitree_rl_gym

The plan named a fallback for the case that Isaac Gym fails: the repository's MuJoCo sim2sim script, `deploy/deploy_mujoco/deploy_mujoco.py`, with its Go2 config and pretrained policy. At commit `276801e` (2025-07-25), neither exists for the Go2:

| file | G1 | H1 | H1_2 | Go2 |
| :--- | :---: | :---: | :---: | :---: |
| `deploy/deploy_mujoco/configs/<robot>.yaml` | yes | yes | yes | no |
| `deploy/pre_train/<robot>/motion.pt` | yes | yes | yes | no |
| `legged_gym/envs/<robot>/` (Isaac Gym training task) | yes | yes | yes | yes |

The README's Sim2Sim table also shows only G1, H1 and H1_2. For the Go2, `unitree_rl_gym` offers only training and playback in Isaac Gym. Isaac Gym works on this computer, so the fallback is not needed.

## Training

The `go2` task trains a velocity-tracking walking policy with PPO (proximal policy optimization, from rsl_rl) on flat ground. Its defaults: 4096 parallel environments, 20 s episodes, 1500 iterations, seed 1. A short run of 500 iterations:

```bash
cd ~/go2_deps/unitree_rl_gym
python legged_gym/scripts/train.py --task=go2 --headless --max_iterations=500 --seed=1 --run_name=seed1_500it
```

| quantity | value |
| :--- | :--- |
| run time | 337 s, 0.68 s per iteration, about 144,000 simulation steps per second; GPU at 90 %, 3.9 GB |
| simulation steps | 49.2 million |
| mean episode reward | 0.00 at iteration 0, 2.68 at 100, 13.72 at 200, 22.53 at 499 |
| mean episode length | 13.7 policy steps at iteration 0, 997.3 of 1000 at 499 |
| linear-velocity tracking reward per episode | 0.002 at iteration 0, 0.940 at 499 |

[`plot_training.py`](plot_training.py) reads the TensorBoard log and draws [`results/training_curve.png`](results/training_curve.png), with the values in `results/training_curve.csv`. The curves show two stages. The robots first learn not to fall: episode length reaches about 900 of 1000 steps by iteration 60. Velocity tracking then rises from 0.2 to 0.9 between iterations 100 and 220 and levels off near 0.94 after iteration 300.

**Repeatability.** A second run with the same command and seed took 346 s. All 18 logged training quantities (rewards, episode length, losses), 9000 values in all, matched the first run exactly. Only the timing values differed. On this computer, GPU PhysX and PyTorch with seed 1 reproduce the training run exactly.

Checkpoints (`model_<iteration>.pt`) stay in `unitree_rl_gym/logs/rough_go2/`, outside this repository.

## Playback

`unitree_rl_gym`'s `play.py` opens a viewer, samples random commands, runs 10 episodes, and records nothing: its `RECORD_FRAMES` flag is set but never used. [`play_record.py`](play_record.py) keeps play.py's test settings (100 environments, no observation noise, no friction randomization, no pushes), and changes four things. It fixes the command to walk straight ahead at 1.0 m/s. It runs headless with one offscreen camera following robot 0. It stops after 10 s. It writes a GIF, a velocity CSV and a summary.

```bash
python <repo>/q4_data_sim/simulation/play_record.py --task=go2 --load_run=<run> --seed=1
```

| quantity | value |
| :--- | :--- |
| commanded forward velocity | 1.0 m/s |
| measured forward velocity after the first 2 s, mean over 100 robots | 0.947 m/s (per-step range 0.889 to 0.991) |
| robot 0 | 0.947 m/s, standard deviation 0.052 m/s |
| falls in 10 s | 0 of 100 |

A second playback gave the same numbers. [`results/play_go2_vx1.gif`](results/play_go2_vx1.gif) (2.1 MB) shows robot 0 trotting, and [`results/play_go2_frames.png`](results/play_go2_frames.png) shows four frames.

## Problems found

**Ninja not found.** On first import, Isaac Gym compiles its PyTorch bridge (`gymtorch`) with PyTorch's extension builder, which calls the `ninja` build tool. pip installs `ninja` into `.venv/bin`, which is not on `PATH` unless the environment is activated, so the import failed with `RuntimeError: Ninja is required to load C++ extensions`. With `.venv/bin` on `PATH`, gcc 15.2 compiled the bridge. For systems with no matching system `libpython` package, Isaac Gym's install guide (`docs/install.html`) sets `LD_LIBRARY_PATH` to the conda environment's `lib` folder, which holds `libpython3.8.so`; `run_go2.sh` does the same.

**Pillow and numpy 1.20.** `unitree_rl_gym` pins numpy 1.20.0, because Isaac Gym uses the `np.float` alias that numpy 1.24 removed. PyTorch had pulled in Pillow 10.4.0, which uses `numpy.typing.NDArray`, added in numpy 1.21. Any import of matplotlib or Pillow then failed with `AttributeError: module 'numpy.typing' has no attribute 'NDArray'`. `pip check` does not catch this, because Pillow declares no numpy version. Pillow 9.5.0 predates that use and works.

**Camera did not follow the robot.** The first recording placed the camera with `set_camera_location` each frame. The camera's reported pose followed robot 0, but the images showed the robot walking away from a fixed viewpoint. On the GPU pipeline, `legged_gym` skips `fetch_results`, so the renderer did not receive the new poses. The script now attaches the camera to robot 0's body (`attach_camera_to_body`, following position only) and calls `fetch_results` before rendering each frame.

**Rendering off when headless.** Isaac Gym sets the graphics device to -1 in headless mode, which disables camera sensors as well as the viewer. `play_record.py` sets the graphics device back to the simulation device before the simulation is created, so it renders offscreen without opening a window.
