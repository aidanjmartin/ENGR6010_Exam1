# Q4 data collection plan

This plan covers how to collect data from the Go2 X for the five methods in Q4, and how the synthetic data in `synthetic/` stands in until real data exists. Statements about the repositories reflect their READMEs and source code as read on 2026-09-30, with links. Terms (Go2 X, Go2 EDU, DDS, WebRTC, ROS 2) follow the definitions in [`q3_apps/REPO_SURVEY.md`](../q3_apps/REPO_SURVEY.md).

## 1. The suggested starting point: unitree_go2_create_dataset

[github.com/ShijieZhao36/unitree_go2_create_dataset](https://github.com/ShijieZhao36/unitree_go2_create_dataset)

**What it records.** One ROS 2 node, `record_data`, subscribes to the robot's `lf/lowstate` topic (or `lowstate`) and writes each message to a rosbag, the ROS 2 log file format, under `dataset/`. The README lists the recorded fields:

| field | meaning |
| :--- | :--- |
| `foot_force` (4) | force on each foot |
| `motor_q` (12) | joint angle |
| `motor_dq` (12) | joint velocity |
| `motor_ddq` (12) | joint acceleration |
| `motor_tau_est` (12) | estimated joint torque |

A second node, `control_speed`, drives the robot by publishing velocity and gait requests to `/api/sport/request` while it reads `sportmodestate`. A third node, `read_data`, reads a recorded bag back. The repository records no camera images and no LiDAR.

**What it requires.** A ROS 2 workspace built with `colcon` and Unitree's [unitree_ros2](https://github.com/unitreerobotics/unitree_ros2) package, which the README asks to install first. unitree_ros2 talks to the robot over DDS, through CycloneDDS: its README connects the computer to the robot by Ethernet cable and gives that interface a static address on the robot's wired subnet.

**Can it work on a Go2 X?** Not as written. Every topic it uses (`lowstate`, `sportmodestate`, `/api/sport/request`) arrives over DDS on the Ethernet network, and a Go2 X does not expose that interface to user code. Its data content is still reachable another way: the WebRTC library in Section 2 defines the same topics (`rt/lf/lowstate`, `rt/sportmodestate`, `rt/api/sport/request`) on its data channel. A port would keep the recorder's message handling and replace the DDS subscription with a WebRTC one. One README detail needs care in any port: it gives the allowed range for `vx` and `vy` as "[−2.5,−5] m/s", which cannot be right as printed.

## 2. What a Go2 X exposes over WebRTC

The WebRTC repositories list the Go2 Air, Pro and EDU. None lists the Go2 X, so each stream below needs checking on the robot in the Linux phase.

| stream | WebRTC source | contents | source |
| :--- | :--- | :--- | :--- |
| front camera video | video track | color video | go2-webrtc README; unitree_webrtc_connect example `video/camera_stream` |
| audio | audio track | microphone audio (send and receive) | unitree_webrtc_connect feature table |
| low-level state | `rt/lf/lowstate` | IMU, 20 motor states, battery, foot force, a `tick` counter | unitree_webrtc_connect `constants.py`; message fields from unitree_ros2 `LowState.msg` |
| sport-mode state | `rt/sportmodestate`, `rt/lf/sportmodestate` | time stamp, mode, gait, body position and velocity, yaw rate, body height, foot positions and speeds | unitree_webrtc_connect `constants.py`; unitree_ros2 `SportModeState.msg` |
| LiDAR | `rt/utlidar/voxel_map_compressed` | voxel map of the built-in LiDAR, decoded to points | unitree_webrtc_connect example `data_channel/lidar` |
| robot pose | `rt/utlidar/robot_pose` | pose estimate from the LiDAR odometry | unitree_webrtc_connect `constants.py` |

Sources: [legion1581/unitree_webrtc_connect](https://github.com/legion1581/unitree_webrtc_connect), [tfoldi/go2-webrtc](https://github.com/tfoldi/go2-webrtc), [abizovnuralem/go2_ros2_sdk](https://github.com/abizovnuralem/go2_ros2_sdk).

Two limits from the go2_ros2_sdk README apply to any logger. First, on firmware 1.1.7 over WebRTC, joint states arrive at about 1 Hz and LiDAR at about 7 Hz. Second, the phone app must disconnect before a WebRTC client connects. Firmware 1.1.15 and later also requires the robot's per-device key, which `unitree_webrtc_connect` fetches with the owner's Unitree account (`unitree-fetch-aes-key`).

## 3. Logging on a Raspberry Pi 4 mounted on the robot

None of the repositories documents a Raspberry Pi. The plan below uses only what the WebRTC library documents: it installs with `pip install unitree_webrtc_connect` plus `portaudio19-dev` on Linux, connects in local-network mode (STA-L) by IP address or serial number, and delivers each topic through a callback.

**Hardware.** A Raspberry Pi 4 (4 GB or more) running 64-bit Raspberry Pi OS, strapped to the robot's back, powered by a USB power bank, with a USB 3 SSD for storage. The Pi joins the same Wi-Fi network as the robot.

**Software.** One Python process with three parts:

1. A WebRTC connection that subscribes to `rt/lf/lowstate`, `rt/lf/sportmodestate`, `rt/utlidar/voxel_map_compressed` and `rt/utlidar/robot_pose`, and receives the video track.
2. A receive callback per stream that stamps every message on arrival with `time.monotonic_ns()` (for ordering and intervals) and `time.time_ns()` (for alignment with other machines), and keeps any time stamp the robot sends (`SportModeState.stamp`, `LowState.tick`).
3. A writer per stream:

| stream | file format | one record holds |
| :--- | :--- | :--- |
| low state, sport state, pose | JSON Lines (one JSON object per line) | both Pi time stamps, the robot time stamp, the message fields |
| LiDAR | one compressed NumPy file (`.npz`) per message, plus an index in JSON Lines | both Pi time stamps, the points |
| video | MP4 at the received frame rate, plus a CSV of per-frame time stamps | frame number, both Pi time stamps |

**Clock.** `chrony` keeps the Pi's wall clock synchronized to a network time server, so `time.time_ns()` lines up with logs from other machines. Each session also writes a header file with the start time, firmware version, library version and topic list.

**Checks before a long recording.** Measure the received rate of each stream against Section 2, the dropped video frames and the Pi's CPU load. The library receives video through aiortc (named in its acknowledgments), which decodes video on the CPU, and a Pi 4 may not keep up at full resolution. If it falls behind, record raw state and LiDAR on the Pi and record video separately.

**Privacy.** `LowState` carries the robot's serial number (`sn`). The logger drops that field before writing. Robot IP addresses, serial numbers, account credentials and the per-device key stay in an untracked `.env` file and never enter the repository. Recorded datasets stay out of the repository too.

## 4. Synthetic data as a stand-in

Until the robot data exists, each script in `synthetic/` generates its own data with exact ground truth. Each one maps to a real stream and to a way of getting ground truth for it later:

| script | synthetic data | real stream that replaces it | ground truth on real data |
| :--- | :--- | :--- | :--- |
| `color_space.py` | orange disc under an illumination gradient and a shadow | front camera frames of a colored target | hand-drawn masks on a few frames |
| `gaussian_pyramid.py` | striped texture | front camera frames | none needed: reconstruction error and aliasing are measured against the frame itself |
| `optical_flow.py` | textured frame with a known shift, rotation and zoom | consecutive camera frames while the robot walks | camera motion from `robot_pose`, checked against flow on distant static scenery |
| `grabcut.py` | blob with a known mask and controlled color overlap | camera frames of an object, with a box drawn around it | hand-drawn masks |
| `supervised_learning.py` | 32 x 32 patches with and without a bright blob | patches cut from camera frames around a target | labels from the hand-drawn masks |

The swap is small in every script: replace the generator function with a loader that reads frames from the video file and its time stamp CSV, and replace the exact ground truth with the hand labels or pose data above. The distribution-shift test in `supervised_learning.py` also suggests what to expect first: classifiers trained on synthetic patches will likely lose accuracy on robot camera patches, as they do on the shifted synthetic set.
