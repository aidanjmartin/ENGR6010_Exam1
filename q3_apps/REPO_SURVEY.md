# Q3 repository survey

This survey covers the five Go2 demo repositories listed in Q3. Each entry states what the repository's README and source code say, read on 2026-09-30, with links. Where a README is silent on a question, the entry says so.

## Terms

- **Go2 X**: the Go2 model used in this exam. It accepts body velocity commands over WebRTC and does not expose its onboard computer to user code.
- **Go2 EDU**: the research model. It adds an onboard NVIDIA Jetson computer and lets a computer on the robot's Ethernet network send joint-level commands.
- **DDS** (Data Distribution Service): the publish-subscribe middleware that Unitree's own SDKs use, through the CycloneDDS implementation, over the robot's Ethernet network.
- **WebRTC** (Web Real-Time Communication): the browser protocol that the Unitree phone app uses to reach the robot over Wi-Fi. It carries video, audio and data channels.
- **ROS 2** (Robot Operating System 2): a robotics middleware built on DDS. "Humble" is the 2022 long-term release for Ubuntu 22.04.
- **SLAM** (simultaneous localization and mapping): building a map while tracking the robot's position in it.
- **Nav2**: the ROS 2 navigation stack, which plans paths and drives a robot to a goal.

## Repositories

### Unitree-Go2-Robot/go2_robot

[github.com/Unitree-Go2-Robot/go2_robot](https://github.com/Unitree-Go2-Robot/go2_robot)

A ROS 2 integration from the Intelligent Robotics Lab at Universidad Rey Juan Carlos. Its checklist marks as done the robot description, odometry, point cloud, joint states, RViz visualization, velocity commands (`cmd_vel`), mode changes, SLAM and Nav2. It marks the hardware interface and Gazebo simulation as not done. **Models:** the README does not name a Go2 model. Its driver ([go2_driver](https://github.com/Unitree-Go2-Robot/go2_driver)) subscribes to Unitree's DDS topics (`lowstate`, `api/sport/request`, `/utlidar/cloud`), which is the EDU interface. The README adds that optional sensor packages "will only work if you throw everything inside the robot", which requires the EDU's onboard computer. **ROS 2:** required, Humble on Ubuntu 22.04. **Simulation:** none; Gazebo support is an open checklist item.

### tfoldi/go2-webrtc

[github.com/tfoldi/go2-webrtc](https://github.com/tfoldi/go2-webrtc)

A WebRTC API for the Go2 with two parts: a JavaScript web page that connects to the robot, sends movement and action commands and shows the camera video, and a Python package for console programs. **Models:** "Go2's WebRTC API supports all models, including Go2 Air, Pro, and Edu versions." The README does not mention the Go2 X. **ROS 2:** not used. **Simulation:** none; the README requires a powered-on robot on the same network. **Firmware:** the latest commit (January 2026) adds the handshake for firmware 1.1.8 and later (`data2=2`). The source code does not handle the per-device key (`data2=3`) that, according to the successor project below, firmware 1.1.15 and later requires. The README points to that successor, [legion1581/unitree_webrtc_connect](https://github.com/legion1581/unitree_webrtc_connect) (formerly `go2_webrtc_connect`), as "more featureful", with video and audio support.

### h-naderi/unitree-go2-slam-nav2

[github.com/h-naderi/unitree-go2-slam-nav2](https://github.com/h-naderi/unitree-go2-slam-nav2)

SLAM with RTAB-Map and autonomous navigation with Nav2, plus frontier exploration and face recognition, demonstrated in two videos. It fuses an Intel RealSense depth camera with a RoboSense LiDAR. **Models:** the README says only "Unitree-Go2" and does not name a model. It lists [unitree_ros2](https://github.com/unitreerobotics/unitree_ros2) as its Unitree dependency, which uses DDS over Ethernet. The README describes launching "everything on the robot". **ROS 2:** required; the README does not name a release. **Simulation:** the README does not mention any.

### YasiruDEX/Go2-Dynamic-Inspection

[github.com/YasiruDEX/Go2-Dynamic-Inspection](https://github.com/YasiruDEX/Go2-Dynamic-Inspection)

The README presents this repository as "Go2 Planner Suite": LiDAR odometry (MOLA), terrain analysis, a CUDA-accelerated visibility-graph planner and a reactive local planner, with a web mission-planning interface. Its FAQ names the hardware as a Go2 "with a Livox Mid-360 LiDAR and built-in IMU", so it expects a LiDAR added to the robot. **Models:** the repository description reads "Unofficial ROS2 SDK with 3D lidar support for Unitree GO2 AIR/PRO/EDU". The source tree includes a `go2_webrtc_bridge` ROS 2 node that forwards `/cmd_vel` velocity commands to the robot over WebRTC through the `go2_webrtc_driver` library. **ROS 2:** required, Humble, plus CUDA 11.0 or later (with slower CPU fallbacks). **Simulation:** yes, a Gazebo world (`workspaces/autonomous_exploration/go2_simulator`) launched by `scripts/sim.sh`. That script's default world file is an absolute path on the author's machine (`/home/yasiru/factory.world`), so it needs a `--world` argument elsewhere. The repository is about 1.3 GB.

### machines-in-motion/Go2Py

[github.com/machines-in-motion/Go2Py](https://github.com/machines-in-motion/Go2Py)

A Python interface for low-level (joint) and high-level control of the Go2, with a Docker-based C++ bridge built on unitree_ros2, a safety state machine, a Pinocchio kinematics and dynamics model, and a MuJoCo simulation "with a Python interface identical to the real robot". **Models:** the README does not name a model. The [setup guide](https://github.com/machines-in-motion/Go2Py/blob/main/docs/setup.md) states that "the GO2 EDU comes with an onboard Jetson Orin NX" and targets that onboard computer. **ROS 2:** optional; the robot interface talks to the bridge "through either DDS (ROS independent) or ROS2 interfaces". **Simulation:** yes. The `Go2Sim` class runs MuJoCo with no robot, demonstrated in the notebook `examples/02-MuJoCo-sim.ipynb`.

## Summary

| repository | robot link | models named | ROS 2 | runs without the robot |
| :--- | :--- | :--- | :--- | :--- |
| go2_robot | DDS (unitree_ros2) | none; EDU interface | Humble, required | no |
| go2-webrtc | WebRTC | Air, Pro, EDU | no | no |
| unitree-go2-slam-nav2 | DDS (unitree_ros2) | none | required | not stated |
| Go2-Dynamic-Inspection | WebRTC bridge for velocity | Air, Pro, EDU (description) | Humble, required | yes, Gazebo |
| Go2Py | DDS through a bridge on the EDU's Jetson | EDU (setup guide) | optional | yes, MuJoCo |

No README names the Go2 X.

## Recommendation

**On the Go2 X: go2-webrtc, with its successor as the fallback.** WebRTC is the only link a Go2 X offers to user code. Of the five repositories, only go2-webrtc is a WebRTC client in its own right, and it needs neither ROS 2 nor added sensors. The firmware version, shown in the Unitree app, decides which library to run. Up to firmware 1.1.14, go2-webrtc's own handshake applies. From firmware 1.1.15, use `unitree_webrtc_connect`, the successor that go2-webrtc's README links to, with the robot's per-device key. Both libraries provide the camera video stream and movement commands. The successor also provides robot-state subscriptions. Go2-Dynamic-Inspection's WebRTC bridge is a later option for driving the robot from ROS 2, but its navigation stack needs a Livox LiDAR that a stock Go2 X does not have.

**In simulation: Go2Py with MuJoCo.** It is plain Python, does not need ROS 2, and its simulator uses the same interface as the real-robot class. Go2-Dynamic-Inspection's Gazebo simulation is the alternative, at the cost of a ROS 2 Humble workspace, CUDA and a 1.3 GB clone. go2_robot and unitree-go2-slam-nav2 offer no simulation.

## Correction after testing on the robot (2026-10-02)

The recommendation above calls WebRTC the only link a Go2 X offers to user code. Testing on the exam's Go2 X, with firmware between 1.1.11 and 1.1.14, showed otherwise. Over the robot's wired network, two examples from `unitree_sdk2_python` reached it over DDS. `capture_image.py`, unchanged, returned a front camera frame. `wireless_controller.py`, with only its message type switched to the Go2's, read `LowState` at about 26 messages per second. The robot has no onboard Jetson computer, so the parts of these repositories that run on the EDU's Jetson still do not apply. [`../q2_sdk/LOG.md`](../q2_sdk/LOG.md) and [`LOG.md`](LOG.md) record the tests. The WebRTC client still works as recommended, and the firmware check did not need the Unitree app: the robot's first signaling reply names its handshake ([`probe_signaling.py`](probe_signaling.py)).
