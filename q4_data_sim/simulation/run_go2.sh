#!/usr/bin/env bash
# Train the unitree_rl_gym Go2 task headless, plot the training curves, and play
# the policy back to a GIF. Results go to results/; checkpoints stay in
# unitree_rl_gym/logs/, outside this repository.
# Needs .venv (see LOG.md) and unitree_rl_gym in ~/go2_deps, or set UNITREE_RL_GYM.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
RL_GYM=${UNITREE_RL_GYM:-$HOME/go2_deps/unitree_rl_gym}
ITERATIONS=${ITERATIONS:-500}
SEED=${SEED:-1}
RUN_NAME=seed${SEED}_${ITERATIONS}it

# Isaac Gym compiles gymtorch with ninja on first import and loads libpython3.8
# from the environment, so both folders must be on the search paths.
export PATH=$HERE/.venv/bin:$PATH
export LD_LIBRARY_PATH=$HERE/.venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

cd "$RL_GYM"
python legged_gym/scripts/train.py --task=go2 --headless \
    --max_iterations="$ITERATIONS" --seed="$SEED" --run_name="$RUN_NAME"
RUN_DIR=$(ls -d logs/rough_go2/*_"$RUN_NAME" | tail -n 1)

python "$HERE/plot_training.py" "$RUN_DIR"
python "$HERE/play_record.py" --task=go2 --load_run="$(basename "$RUN_DIR")" --seed="$SEED"

# Shrink the GIF: 15 frames per second, 400 px wide, 64-color palette.
GIF=$HERE/results/play_go2_vx1.gif
ffmpeg -loglevel error -y -i "$GIF" -vf "fps=15,scale=400:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" "$GIF.tmp.gif"
mv "$GIF.tmp.gif" "$GIF"
