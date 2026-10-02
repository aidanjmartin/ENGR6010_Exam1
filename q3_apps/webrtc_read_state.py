"""Read camera video and robot state from a Go2 over WebRTC. Sends no motion commands.

Built on legion1581/unitree_webrtc_connect, following its examples
go2/video/camera_stream and go2/data_channel/lowstate. For a set time it:

- receives the front camera video track and saves a few frames as PNG,
- subscribes to rt/lf/lowstate and rt/lf/sportmodestate and writes selected
  fields with arrival times to CSV,
- prints the received rate of each stream.

The robot address, the per-device AES-128 key (firmware 1.1.15 and later) and the
serial number come from the untracked ../.env file and are never written to output.

Usage:
    python webrtc_read_state.py                  # LocalSTA mode, GO2_IP from .env
    python webrtc_read_state.py --ap             # robot's own Wi-Fi hotspot (LocalAP)
    python webrtc_read_state.py --seconds 30 --out output/webrtc
"""
import argparse
import asyncio
import csv
import logging
import os
import time

import cv2
from unitree_webrtc_connect.constants import RTC_TOPIC
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod

HERE = os.path.dirname(os.path.abspath(__file__))


def read_env():
    env = {}
    try:
        with open(os.path.join(HERE, "..", ".env")) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    env[key.strip()] = value.strip()
    except OSError:
        pass
    return env


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seconds", type=float, default=20.0, help="how long to record")
    p.add_argument("--ap", action="store_true", help="connect through the robot's hotspot")
    p.add_argument("--frames", type=int, default=4, help="camera frames to save")
    p.add_argument("--out", default=os.path.join(HERE, "output", "webrtc"))
    return p.parse_args()


async def main():
    args = parse_args()
    env = read_env()
    key = env.get("GO2_AES_KEY") or None
    os.makedirs(args.out, exist_ok=True)

    if args.ap:
        conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP, aes_128_key=key)
    else:
        if not env.get("GO2_IP"):
            raise SystemExit("Set GO2_IP in .env, or use --ap")
        conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=env["GO2_IP"],
                                       aes_128_key=key)

    t0 = time.monotonic()
    frames, frame_times = [], []
    low_rows, sport_rows = [], []

    async def on_video(track):
        while True:
            frame = await track.recv()
            frame_times.append(time.monotonic() - t0)
            frames.append(frame.to_ndarray(format="bgr24"))
            if len(frames) > 2 * args.frames:  # keep memory bounded; spread out below
                del frames[1:-1:2]

    def on_lowstate(message):
        d = message["data"]
        rpy = d["imu_state"]["rpy"]
        low_rows.append([round(time.monotonic() - t0, 4), *[round(v, 4) for v in rpy],
                         d["bms_state"]["soc"], d["power_v"], *d["foot_force"]])

    def on_sportstate(message):
        d = message["data"]
        sport_rows.append([round(time.monotonic() - t0, 4), d.get("mode"), d.get("gait_type"),
                           d.get("body_height"), *d.get("velocity", [None] * 3)])

    print("connecting ...", flush=True)
    await conn.connect()
    print(f"connected after {time.monotonic() - t0:.1f} s", flush=True)
    conn.video.switchVideoChannel(True)
    conn.video.add_track_callback(on_video)
    conn.datachannel.pub_sub.subscribe(RTC_TOPIC["LOW_STATE"], on_lowstate)
    conn.datachannel.pub_sub.subscribe(RTC_TOPIC["LF_SPORT_MOD_STATE"], on_sportstate)

    t_start = time.monotonic() - t0
    await asyncio.sleep(args.seconds)
    span = time.monotonic() - t0 - t_start
    await conn.disconnect()

    with open(os.path.join(args.out, "lowstate.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "roll", "pitch", "yaw", "battery_soc_pct", "power_v",
                    "foot_force_0", "foot_force_1", "foot_force_2", "foot_force_3"])
        w.writerows(low_rows)
    with open(os.path.join(args.out, "sportmodestate.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "mode", "gait_type", "body_height_m", "vx", "vy", "vz"])
        w.writerows(sport_rows)
    for i, img in enumerate(frames[:: max(1, len(frames) // args.frames)][: args.frames]):
        cv2.imwrite(os.path.join(args.out, f"camera_{i}.png"), img)

    print(f"recorded {span:.1f} s")
    print(f"camera: {len(frame_times)} frames, {len(frame_times) / span:.1f} per second"
          + (f", {frames[0].shape[1]} x {frames[0].shape[0]}" if frames else ""))
    print(f"rt/lf/lowstate: {len(low_rows)} messages, {len(low_rows) / span:.1f} per second")
    print(f"rt/lf/sportmodestate: {len(sport_rows)} messages, {len(sport_rows) / span:.1f} per second")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
