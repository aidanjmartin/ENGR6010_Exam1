"""Find which WebRTC handshake a Go2 speaks, without the phone app.

The robot answers WebRTC signaling on TCP port 9991 (firmware 1.1.11 and later) or
8081 (older firmware). On port 9991, the first request (con_notify) returns a JSON
object whose `data2` field names the handshake:

    data2 = 2   firmware up to 1.1.14: static key; tfoldi/go2-webrtc or unitree_webrtc_connect
    data2 = 3   firmware 1.1.15 and later: per-device AES-128 key; unitree_webrtc_connect only

This script sends only that first request, the same one unitree_webrtc_connect sends
(unitree_auth.send_sdp_to_local_peer_new_method), and prints the port and `data2`.
It starts no session and sends no commands. It uses only the Python standard library.

Usage:
    python probe_signaling.py                       # GO2_IP from ../.env, then the defaults
    python probe_signaling.py 192.168.123.161 192.168.12.1
"""
import base64
import json
import os
import socket
import sys
import urllib.request

# Addresses a Go2 uses: Ethernet (sport-mode board), its own Wi-Fi hotspot, the EDU's Jetson.
DEFAULT_IPS = ["192.168.123.161", "192.168.12.1", "192.168.123.18"]
MEANING = {
    1: "no encryption of data1 (very old firmware)",
    2: "firmware 1.1.14 or earlier: static key, no per-device key needed",
    3: "firmware 1.1.15 or later: per-device AES-128 key required",
}


def env_ip():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    try:
        with open(path) as f:
            for line in f:
                key, _, value = line.strip().partition("=")
                if key == "GO2_IP" and value:
                    return value
    except OSError:
        pass
    return None


def port_open(ip, port, timeout=1.5):
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def probe(ip):
    print(f"{ip}:")
    open_9991, open_8081 = port_open(ip, 9991), port_open(ip, 8081)
    print(f"  port 9991 (con_notify): {'open' if open_9991 else 'no answer'}")
    print(f"  port 8081 (legacy offer): {'open' if open_8081 else 'no answer'}")
    if not open_9991:
        if open_8081:
            print("  result: legacy signaling, firmware before 1.1.11; tfoldi/go2-webrtc works")
        return
    req = urllib.request.Request(f"http://{ip}:9991/con_notify", data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=5) as resp:
        body = json.loads(base64.b64decode(resp.read()).decode("utf-8"))
    data2 = body.get("data2")
    print(f"  data2 = {data2}: {MEANING.get(data2, 'unknown value')}")


def main():
    ips = sys.argv[1:] or ([env_ip()] if env_ip() else []) + DEFAULT_IPS
    for ip in dict.fromkeys(ips):
        try:
            probe(ip)
        except Exception as e:  # report and move on to the next address
            print(f"  error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
