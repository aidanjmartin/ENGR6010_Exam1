#!/usr/bin/env bash
# Run the Q2 SDK examples against the simulated Go2 and write their output to output/.
# Needs .venv (see LOG.md) and the upstream repositories in ~/go2_deps, or set
# UNITREE_SDK2_PYTHON and UNITREE_MUJOCO.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
SDK=${UNITREE_SDK2_PYTHON:-$HOME/go2_deps/unitree_sdk2_python}
OUT=output
mkdir -p "$OUT"

echo "Example 1: helloworld publisher and subscriber (DDS only)"
(cd "$SDK/example/helloworld" && timeout -s KILL 15 "$OLDPWD/$PY" -u subscriber.py) > "$OUT/ex1_helloworld_subscriber.txt" 2>&1 &
sleep 2
(cd "$SDK/example/helloworld" && timeout -s KILL 12 "$OLDPWD/$PY" -u publisher.py) > "$OUT/ex1_helloworld_publisher.txt" 2>&1
wait

echo "Example 2a: go2_stand_example.py as shipped (expected to fail in simulation)"
$PY -u sim_go2.py --duration 30 > "$OUT/ex2a_sim.txt" 2>&1 &
SIM=$!
sleep 3
(cd "$SDK/example/go2/low_level" && echo | timeout -s KILL 25 "$OLDPWD/$PY" -u go2_stand_example.py lo) > "$OUT/ex2a_stand_unmodified.txt" 2>&1
kill -9 $SIM 2>/dev/null; wait $SIM 2>/dev/null

echo "Example 2: go2_stand_example_sim.py, recorded"
$PY -u sim_go2.py --duration 12 --gif "$OUT/ex2_stand.gif" --sheet "$OUT/ex2_stand_frames.png" \
    --csv "$OUT/ex2_stand_height.csv" > "$OUT/ex2_sim.txt" 2>&1 &
SIM=$!
sleep 2.5
timeout -s KILL 25 $PY -u examples/go2_stand_example_sim.py lo > "$OUT/ex2_stand_example.txt" 2>&1
wait $SIM
# Shrink the GIF below 10 MB: 12 frames per second, 400 px wide, 96-color palette.
ffmpeg -loglevel error -y -i "$OUT/ex2_stand.gif" -vf "fps=12,scale=400:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" "$OUT/ex2_stand_small.gif" \
    && mv "$OUT/ex2_stand_small.gif" "$OUT/ex2_stand.gif"

echo "Example 3: wireless_controller_sim.py"
$PY -u sim_go2.py --duration 8 > "$OUT/ex3_sim.txt" 2>&1 &
SIM=$!
sleep 2.5
FULL=$(mktemp)
timeout -s KILL 3 $PY -u examples/wireless_controller_sim.py lo > "$FULL" 2>&1
wait $SIM
{ echo "# First 50 lines of $(wc -l < "$FULL"). The example parsed $(grep -c 'debug unitreeRemoteController' "$FULL") LowState messages in a 3 s run. Every field reads 0 because no gamepad is attached to the simulator."
  head -n 50 "$FULL"; } > "$OUT/ex3_wireless_controller.txt"
rm -f "$FULL"

echo "Unsupported: go2_sport_client.py (high-level service, absent in simulation)"
$PY -u sim_go2.py --duration 20 > "$OUT/ex4_sim.txt" 2>&1 &
SIM=$!
sleep 2.5
(cd "$SDK/example/go2/high_level" && (printf '\n'; sleep 3; printf '1\n'; sleep 14) | timeout -s KILL 20 "$OLDPWD/$PY" -u go2_sport_client.py lo) > "$OUT/ex4_sport_client_unsupported.txt" 2>&1
wait $SIM
echo "done; see $OUT/"
