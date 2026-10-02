# Q3 log

Machine: lab desktop, Intel Core Ultra 9 285K, 60 GB RAM, NVIDIA RTX 4000 Ada Generation (20 GB), driver 595.91.07 (CUDA 13.2), Ubuntu 26.04.1 LTS. Machine setup (system packages, Miniconda, upstream clones) is in [`../q2_sdk/LOG.md`](../q2_sdk/LOG.md).
Date: 2026-10-02.

[`REPO_SURVEY.md`](REPO_SURVEY.md) chose two repositories: Go2Py in simulation, and a WebRTC client on the real Go2 X. This log covers both.

## Environment

One environment, `q3_apps/.venv`, holds both Go2Py and the WebRTC library.

| step | command | outcome |
| :--- | :--- | :--- |
| create environment | `conda create -p q3_apps/.venv --override-channels -c conda-forge python=3.10 pip` | success: Python 3.10 |
| install PyTorch, CPU build | `pip install torch --index-url https://download.pytorch.org/whl/cpu` | success: torch 2.14.1+cpu; replaced below |
| install notebook packages | `pip install mujoco numpy scipy opencv-python matplotlib nbconvert ipykernel` | success: mujoco 3.14.0, numpy 2.2.6; MuJoCo downgraded below |
| install Go2Py | `pip install -e ~/go2_deps/Go2Py` | success: Go2Py 1.0.0, commit `eff240b` |
| pin MuJoCo | `pip install mujoco==3.4.0` | success; see Problems |
| install PyTorch, CUDA build | `pip uninstall torch && pip install torch --index-url https://download.pytorch.org/whl/cu128` | success: torch 2.11.0+cu128, `torch.cuda.is_available()` is True on the RTX 4000 Ada |
| install WebRTC library | `pip install -e ~/go2_deps/unitree_webrtc_connect` | success: unitree_webrtc_connect 2.2.0, commit `e0abc57`, aiortc 1.15.0; `pip check` finds no conflicts |

`requirements.txt` lists the final versions.

## Simulation: Go2Py notebook `examples/02-MuJoCo-sim.ipynb`

The notebook runs unchanged and unattended with:

```bash
cd ~/go2_deps/Go2Py/examples
jupyter nbconvert --to notebook --execute --allow-errors \
  --output-dir <repo>/q3_apps/output --output 02-MuJoCo-sim.executed.ipynb 02-MuJoCo-sim.ipynb
```

It opens a MuJoCo viewer window and an OpenCV window during its two 10 s loops.

| run | environment | outcome |
| :--- | :--- | :--- |
| 1 | torch 2.14.1+cpu, mujoco 3.14.0 | low-level half succeeds (cells 1 to 6); high-level half fails in three cells; 18 s |
| 2 | torch 2.11.0+cu128, mujoco 3.4.0 | every cell succeeds; 28 s |

What each cell produced in run 2:

| cells | what it does | result |
| :--- | :--- | :--- |
| 1, 2 | create `Go2Sim(mode='lowlevel')` | viewer window opens |
| 3 | front camera RGB image | [`output/go2py_front_camera.png`](output/go2py_front_camera.png), 640 x 480 |
| 4 | front camera depth as a point cloud | [`output/go2py_pointcloud.png`](output/go2py_pointcloud.png) |
| 5 | `getJointStates()` | 12 joint angles, velocities and torques |
| 6 | 10 s of joint torque `tau = 20 (q0 - q)` with the camera shown | runs to completion |
| 8 to 10 | create `Go2Sim(mode='highlevel')`, which loads the walk-these-ways policy, then 10 s of `step(0, 0, 0, step_height=0)` | runs to completion; prints the policy's joint gains `p_gains: [20 ... 20]` |
| 11, 12 | `getLaserScan()` and a plot of hits within 3 m | 1024 rays cast; no hit within 3 m, because the scene is an empty floor, so the plot is empty |

[`output/02-MuJoCo-sim.executed.ipynb`](output/02-MuJoCo-sim.executed.ipynb) holds every output.

### Recording the two control loops

The notebook shows motion only in live windows. [`go2py_record.py`](go2py_record.py) repeats its two loops with the same commands, renders them offscreen, and saves a GIF and a CSV of body height for each. It runs a fixed number of steps (10 s at `dt = 0.002` s) where the notebook uses a wall-clock timer, and seeds NumPy and PyTorch.

| step | command | outcome |
| :--- | :--- | :--- |
| record | `.venv/bin/python go2py_record.py` | success in 11 s |
| low-level loop | | body height 0.330 m at start, 0.249 m at 10 s. The PD target `q0` is Go2Py's sitting pose, so the hind legs fold. |
| high-level loop | | body height 0.330 m at start, 0.250 m at 10 s. The policy holds a standing crouch at zero commanded velocity. |
| repeat | same command | both CSV files identical byte for byte |
| compress | `ffmpeg ... fps=10, scale=360, 96-color palette` | `go2py_lowlevel.gif` 1.3 MB, `go2py_highlevel.gif` 2.6 MB |

[`output/go2py_frames.png`](output/go2py_frames.png) shows both loops at 0, 2 and 9.9 s.

## Real robot: WebRTC

The robot is a Go2 X. This computer has no Wi-Fi adapter, so it reaches the robot through a USB-C Ethernet adapter on the robot's wired network; [`../q2_sdk/LOG.md`](../q2_sdk/LOG.md) records the network setup. The phone app was not used.

Two scripts:

- [`probe_signaling.py`](probe_signaling.py) asks the robot which WebRTC handshake it uses. It sends only the first signaling request (`con_notify` on port 9991) and prints the `data2` field: 2 means firmware 1.1.14 or earlier, 3 means firmware 1.1.15 or later, which needs the per-device AES-128 key. It falls back to checking port 8081 for firmware older than 1.1.11.
- [`webrtc_read_state.py`](webrtc_read_state.py) connects with `unitree_webrtc_connect`, saves front camera frames, and records `rt/lf/lowstate` and `rt/lf/sportmodestate` with arrival times. It sends no motion commands. It reads the robot address and key from the untracked `.env` file and writes neither to its output.

| step | command | outcome |
| :--- | :--- | :--- |
| handshake type | `python probe_signaling.py <GO2_IP>` | port 9991 open, port 8081 closed, `data2 = 2`: firmware 1.1.11 to 1.1.14. No AES-128 key or Unitree account is needed, and both go2-webrtc and unitree_webrtc_connect apply. |
| camera and state, 20 s | `python webrtc_read_state.py --seconds 20` | **success**: connected in 0.5 s over the wired network |

What the 20 s run received:

| stream | messages | rate | contents |
| :--- | ---: | ---: | :--- |
| front camera video | 274 frames | 13.7 per second | 1280 x 720 color |
| `rt/lf/lowstate` | 20 | 1.0 per second | IMU roll, pitch and yaw; battery 38 %, 28.45 V; four foot forces between 78 and 94 |
| `rt/lf/sportmodestate` | 400 | 20.0 per second | mode 0, gait 0, body height 0.321 m, body velocity below 0.04 m/s |

[`output/webrtc/camera_frames.jpg`](output/webrtc/camera_frames.jpg) shows four frames spread over the run, and the two CSV files beside it hold the state. [`output/webrtc/terminal.txt`](output/webrtc/terminal.txt) is the terminal output, with the robot address replaced by `<GO2_IP>`.

The low-level state arrives at 1 per second over WebRTC, which matches the rate that go2_ros2_sdk's README reports for firmware 1.1.7. The robot stood still throughout, and the IMU yaw drifted from 0.1200 to 0.1271 rad in 20 s.

## Problems found

**CUDA checkpoint on a CPU-only PyTorch.** With the CPU build of torch, `Go2Sim(mode='highlevel')` failed with `RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False`. Go2Py loads the walk-these-ways policy with `torch.load` and no `map_location`, and the checkpoint was saved from a GPU. The CUDA build of torch loads it unchanged. The next cell then failed with `TypeError: Go2Sim.stepLowlevel() got an unexpected keyword argument 'step_height'`, a consequence of the first error: `robot` was still the low-level simulator.

**MuJoCo API change.** With MuJoCo 3.14.0, `getLaserScan()` failed with `TypeError: mj_multiRay(): incompatible function arguments`. MuJoCo 3.5.0 added a required `normal` output argument to `mj_multiRay`, and Go2Py's call predates it. Checking the binding's signature in each release showed that 3.3.0, 3.3.7 and 3.4.0 lack `normal` and 3.5.0 and 3.6.0 have it. MuJoCo 3.4.0 runs every cell.

**Video decoder warnings at connect.** The WebRTC run printed `H264Decoder() failed to decode, skipping package` 9 times in its first second. The decoder drops packets until the first complete frame (a keyframe) arrives, and then decodes every frame. At disconnect, the video callback logs one empty `Error in callback` when the video track closes. Neither affects the recorded data.

**Display messages.** On this GNOME Wayland desktop, glfw prints `Failed to load plugin 'libdecor-gtk.so'` and `Wayland: The platform does not provide the window position`, and OpenCV's Qt window prints `QFontDatabase: Cannot find font directory`. None of these stops a run.
