# Q2 log

Machine: lab desktop, Intel Core Ultra 9 285K (24 cores), 60 GB RAM, NVIDIA RTX 4000 Ada Generation (20 GB), driver 595.91.07 (CUDA 13.2), Ubuntu 26.04.1 LTS, kernel 7.0.0-34, system Python 3.14.4.
Date: 2026-10-02.

## Machine setup

This setup serves Q2, Q3 and Q4.

| step | command | outcome |
| :--- | :--- | :--- |
| system packages | `sudo apt install build-essential cmake python3-venv ffmpeg portaudio19-dev libglfw3` | success: gcc 15.2.0, cmake 4.2.3, ffmpeg, portaudio19-dev 19.7.0, libglfw3 3.4 |
| Miniconda | `bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3` | success: conda 26.7.1, shell startup files left unchanged |
| upstream repositories | `git clone --depth 1 <url> ~/go2_deps/<name>` | success; commits below |

Ubuntu 26.04 ships only Python 3.14. Isaac Gym needs Python 3.8, and CycloneDDS 0.10.2 and MuJoCo are better tested on older releases, so each part gets its own conda environment at `<part>/.venv`, created as a conda prefix. Every environment comes from conda-forge only (`--override-channels -c conda-forge`), because Anaconda's default channel requires accepting its terms of service.

The upstream repositories live outside this repository, in `~/go2_deps/`:

| repository | branch or tag | commit | commit date |
| :--- | :--- | :--- | :--- |
| eclipse-cyclonedds/cyclonedds | `releases/0.10.x` | `5041f35` | 2026-03-23 |
| unitreerobotics/unitree_sdk2_python | default | `814556d` | 2026-09-21 |
| unitreerobotics/unitree_mujoco | default | `1eb6642` | 2026-09-07 |
| unitreerobotics/unitree_rl_gym | default | `276801e` | 2025-07-25 |
| leggedrobotics/rsl_rl | `v1.0.2` | `2ad79cf` | |
| machines-in-motion/Go2Py | default | `eff240b` | 2025-12-03 |
| legion1581/unitree_webrtc_connect | default | `e0abc57` | 2026-08-16 |
| tfoldi/go2-webrtc | default | `9239041` | 2026-01-19 |

## Environment

| step | command | outcome |
| :--- | :--- | :--- |
| create environment | `conda create -p q2_sdk/.venv --override-channels -c conda-forge python=3.11 pip` | success: Python 3.11.16 |
| build CycloneDDS | `cd ~/go2_deps/cyclonedds/build && cmake .. -DCMAKE_INSTALL_PREFIX=../install -DBUILD_EXAMPLES=OFF && cmake --build . --target install` | success |
| install the SDK | `CYCLONEDDS_HOME=~/go2_deps/cyclonedds/install pip install -e ~/go2_deps/unitree_sdk2_python` | success: unitree_sdk2py 1.0.1, cyclonedds 0.10.2 built against the library above, numpy 2.4.6, opencv-python 5.0.0.93 |
| install simulator packages | `pip install mujoco pygame pillow` | success: mujoco 3.14.0, pygame 2.6.1, pillow 12.3.0 |
| check offscreen rendering | `MUJOCO_GL=<backend> python -c "mujoco.Renderer(...).render()"` | `egl`: fails with `EGLError`; `osmesa`: fails, library not installed; `glfw`: works |

`requirements.txt` lists the final versions.

## Real robot over Ethernet

The SDK talks DDS over the robot's wired network (192.168.123.0/24), which a Go2 EDU exposes and a Go2 X is not expected to. The robot connects to this computer through a USB-C Ethernet adapter.

| step | command | outcome |
| :--- | :--- | :--- |
| link state | `ip -brief link`, `nmcli dev` | adapter up with a link; NetworkManager stuck on "getting IP configuration", because the robot's wired port runs no DHCP server |
| find the device without configuring an address | `ping -6 ff02::1%<adapter>` (IPv6 link-local, all nodes) | one device answers, 1.6 ms round trip |
| check its ports over IPv6 link-local | TCP connect to 9991, 8081, 8082, 22, 80 | all refused: the device runs IPv6 but offers none of these services on it |
| static IPv4 address on the adapter | `nmcli con add type ethernet ifname <adapter> con-name go2-eth ipv4.method manual ipv4.addresses 192.168.123.99/24 ipv4.never-default yes ipv6.method link-local && nmcli con up go2-eth` | success; the internet connection on the other port is unchanged |
| find the robot over IPv4 | `ping` to the addresses Unitree documents | the robot's main board answers (address in `.env`); the address of a Go2 EDU's onboard Jetson computer does not |
| one read-only SDK example | `python example/go2/front_camera/capture_image.py <adapter>`, unchanged | **success**: the robot's video service answered over DDS and returned a 1920 x 1080 JPEG of 144 KB ([`output/real_front_camera.jpg`](output/real_front_camera.jpg), reduced to 960 x 540) |
| a second read-only example | `python examples/wireless_controller_go2.py <adapter>`: upstream with the Go2 message type, keeping the upstream topic `rt/lf/lowstate` | **success**: 78 `LowState` messages parsed in a 4 s run that includes start-up, about 26 per second; every remote-control field reads 0 because the remote was not touched ([`output/real_wireless_controller.txt`](output/real_wireless_controller.txt)) |

**The Go2 X answers DDS requests.** The plan expected a Go2 X to keep its DDS interface closed to user code, as [`../q3_apps/REPO_SURVEY.md`](../q3_apps/REPO_SURVEY.md) assumed. On this robot, with firmware between 1.1.11 and 1.1.14 (see [`../q3_apps/LOG.md`](../q3_apps/LOG.md)), an unchanged SDK example reached the video service over the wired network on the first try. Only the onboard Jetson computer of the EDU is missing.

## Simulator

The SDK examples run against [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)'s Python simulator over the loopback interface (`lo`). That simulator supports only low-level messages: it publishes `rt/lowstate` and `rt/sportmodestate`, and it applies `rt/lowcmd`. It has no sport-mode, motion-switcher or video services.

| step | command | outcome |
| :--- | :--- | :--- |
| stock simulator | `cd simulate_python && python unitree_mujoco.py` | window opens, robot stays frozen; see Problems |
| runner | `python sim_go2.py --duration 3 --png t.png` | success: 3.0 s simulated in 3.5 s; with no commands the body falls from 0.446 m to 0.077 m |

[`sim_go2.py`](sim_go2.py) wraps the simulator. It keeps the simulator's model and SDK bridge unchanged, uses DDS domain 0 on `lo` with no gamepad, runs for a fixed time, and can record a GIF, a frame contact sheet and a body-height CSV offscreen.

## Runs

[`run_sim_examples.sh`](run_sim_examples.sh) reproduces every run below and writes to `output/`.

| example | command | outcome |
| :--- | :--- | :--- |
| 1. `helloworld` publisher and subscriber | `python publisher.py` and `python subscriber.py`, unchanged | success: the subscriber received all 12 messages sent in 12 s |
| 2a. `go2/low_level/go2_stand_example.py`, as shipped | `python go2_stand_example.py lo` | fails: `[ClientStub] send request error`, then `TypeError: 'NoneType' object is not subscriptable` at `while result['name']:` |
| 2. the same, with the service calls removed | `python examples/go2_stand_example_sim.py lo` | success: the robot stands from lying, holds, and crouches; prints `Done!` |
| 3. `wireless_controller/wireless_controller.py`, Go2 message type | `python examples/wireless_controller_sim.py lo` | success: 574 and 577 `LowState` messages parsed in two 3 s runs, about 200 per second; every field reads 0 with no gamepad |
| not supported: `go2/high_level/go2_sport_client.py` | `python go2_sport_client.py lo`, option 1 (stand up) | fails: `[ClientStub] send request error`, because the simulator has no sport-mode service |

[`examples/`](examples/) holds the changed copies of examples 2 and 3. Each starts with a docstring that lists its changes from upstream.

Example 2 in numbers, from `output/ex2_stand_height.csv`, over two runs:

| quantity | run 1 | run 2 |
| :--- | ---: | ---: |
| body height first above 0.25 m | 2.94 s | 3.02 s |
| peak body height | 0.336 m at 3.75 s | 0.336 m at 3.88 s |
| mean body height, 4 to 5 s | 0.298 m | 0.320 m |

MuJoCo's physics is deterministic, but the example runs as a separate process and starts about 0.1 s later or earlier from run to run. The peak height repeats exactly, and the timing shifts by that offset. The 4 to 5 s window falls at a different point in the hold phase in each run, so the means differ.

[`output/ex2_stand_frames.png`](output/ex2_stand_frames.png) shows the robot at 1.0, 3.5, 5.0 and 7.5 s, and [`output/ex2_stand.gif`](output/ex2_stand.gif) (4.1 MB) shows the whole run.

## Problems found

**The stock simulator stops with no gamepad.** `config.py` sets `USE_JOYSTICK = 1`. With no gamepad attached, `SetupJoystick` prints `No gamepad detected.` and calls `sys.exit()`, which ends only the physics thread. The viewer window stays open with a frozen robot and ignores the stop signal from `timeout`, so the process needed `kill -9`. `sim_go2.py` sets `USE_JOYSTICK = 0`.

**DDS domain mismatch.** Every Go2 SDK example calls `ChannelFactoryInitialize(0, <interface>)`, and the simulator's `config.py` sets `DOMAIN_ID = 1` to keep simulated traffic apart from a real robot on domain 0. `sim_go2.py` uses domain 0 so the examples run unchanged. Traffic on `lo` never leaves this computer, so it cannot reach a robot.

**Loopback multicast.** CycloneDDS prints `selected interface "lo" is not multicast-capable: disabling multicast` at every start. Discovery still works over unicast on `lo`.

**No motion-switcher service.** `go2_stand_example.py` first asks the motion-switcher service to release the robot from sport mode, a step a real Go2 EDU needs before it accepts low-level commands. In the simulator nothing answers, `CheckMode()` returns `None`, and the example stops with a `TypeError`. The simulator copy removes that block and nothing else.

**Command before state.** In example 2 the first one or two control-loop ticks raised `AttributeError: 'NoneType' object has no attribute 'motor_state'`, because the loop starts before the first `LowState` message arrives. The upstream example hides this race with its service calls, which take time. The loop recovers on the next tick.

**The simulator holds the last torque, not the last target.** The bridge turns each `LowCmd` into a joint torque, `tau + kp (q_target - q) + kd (dq_target - dq)`, once, when the message arrives. When the example exits, that torque stays applied while the joints keep moving, so the legs drift (frame at 7.5 s). A real robot's motor controllers keep tracking the last target.

**Sag while holding.** During the hold phase the body sags from 0.336 m to about 0.30 m. The example's PD gains (`Kp = 60`, `Kd = 5`) include no gravity compensation, so each joint settles where the spring torque balances the load.

**Recording slows the simulator.** With offscreen rendering at 20 frames per second, 12 s of simulated time took 13.8 to 14.2 s.

**Wrong message type for the Go2.** `wireless_controller.py` ships with the G1 and H1-2 message type (`unitree_hg`) active, and the Go2 lines commented out. It also subscribes to `rt/lf/lowstate`, which the simulator does not publish. The simulator copy switches both.
