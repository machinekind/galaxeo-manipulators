#!/usr/bin/env bash
# The wrist-image grasp student, DAgger rounds on one box: collect with every
# core, train on the GPU, evaluate, repeat. Everything lands under
# sim/runs/$RUN_NAME in the checkout (pull that); the datasets under
# sim/datasets/$RUN_NAME stay on the box.
#
# Round 0 is teacher-driven; round k>0 has the student driving all but a
# BETAS[k] share of ticks, the teacher labelling every tick. The student of
# round k trains on every round so far, warm from round k-1.
set -euo pipefail

: "${TEACHER:=sim/grasp/teacher_v2.pt}"   # privileged policy.pt from sim/grasp/train.py, shipped in the checkout
: "${JOB_ID:=local}"
: "${RUN_NAME:=grasp_student_${JOB_ID}}"
: "${ROUNDS:=5}"                          # DAgger rounds including round 0
: "${EPISODES:=4000}"                     # episodes collected per round, over all shards
: "${BETAS:=1.0 0.5 0.3 0.2 0.1 0.05 0.0}"  # teacher share per round; the last value repeats
: "${EPOCHS:=15}"                         # student epochs per round (round 0 gets 2x)
: "${BATCH:=256}"
: "${IMG_W:=128}"
: "${IMG_H:=96}"
: "${COMPOSITES:=0.5}"
: "${DISTURB:=0.3}"
: "${FORCE:=0.3}"
: "${EVAL_N:=100}"                        # closed-loop episodes per evaluation
: "${SHARDS:=}"                           # collector processes; empty: cores minus two
: "${SEED:=0}"
: "${GPUS:=1}"
: "${ON_FAILURE:=Stops at the failed round; the rounds already under sim/runs/RUN_NAME (student.pt, summary.txt) are complete and pullable.}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# CUDA 13 wheels on a pre-580 driver: node_boot.sh installs the forward-compat libraries here
[ -d /usr/local/cuda-13.0/compat ] && export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"   # one BLAS thread per collector process
export PYTHONUNBUFFERED=1

cd "$(dirname "${BASH_SOURCE[0]}")/../.."       # the checkout root; every path below is relative to it
PY=sim/grasp
OUT="sim/runs/${RUN_NAME}"
DATA="sim/datasets/${RUN_NAME}"
mkdir -p "$OUT" "$DATA"
SUMMARY="$OUT/summary.txt"

[ -f "$TEACHER" ] || { echo "teacher $TEACHER is not in the checkout" >&2; exit 1; }

echo "== node =="
python - <<'PY'
import torch, mujoco, numpy as np
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("mujoco", mujoco.__version__)
# EGL render must work or every collector fails: a forward pass first, or the frame is black
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 2"/><geom type="box" size="1 1 .1" rgba="1 0 0 1"/></worldbody></mujoco>')
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 64, 64); r.update_scene(d); f = r.render()
assert f.mean() > 1.0, "EGL render came back black"
print("render ok, mean", round(float(f.mean()), 1))
PY

CORES=$(nproc --all 2>/dev/null || sysctl -n hw.ncpu)
[ -n "$SHARDS" ] || SHARDS=$(( CORES > 3 ? CORES - 2 : 1 ))
PER_SHARD=$(( EPISODES / SHARDS ))
echo "cores $CORES, shards $SHARDS x $PER_SHARD episodes, rounds $ROUNDS, images ${IMG_W}x${IMG_H}"
DEVICE=cpu
python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' && DEVICE=cuda

beta_for() {                          # BETAS[k], the last value repeating
    local k=$1 v="" i=0
    for b in $BETAS; do v=$b; [ "$i" -eq "$k" ] && break; i=$((i + 1)); done
    echo "$v"
}

DATA_DIRS=""
PREV=""
for (( r = 0; r < ROUNDS; r++ )); do
    RD="$DATA/r$r"; RO="$OUT/r$r"
    mkdir -p "$RD" "$RO"
    BETA=$(beta_for "$r")
    echo "== round $r: collect $EPISODES episodes, teacher share $BETA =="
    T0=$(date +%s)
    for (( k = 0; k < SHARDS; k++ )); do
        SEED_K=$(( (SEED + 1) * 1000000 + r * 100000 + k * 1000 ))
        if [ -z "$PREV" ]; then
            python $PY/collect.py --teacher "$TEACHER" --out "$RD" --episodes "$PER_SHARD" --seed "$SEED_K" \
                --size "$IMG_W" "$IMG_H" --composites "$COMPOSITES" --disturb "$DISTURB" --force "$FORCE" --vis \
                > "$RO/collect_$k.log" 2>&1 &
        else
            python $PY/collect.py --teacher "$TEACHER" --student "$PREV" --beta "$BETA" --out "$RD" \
                --episodes "$PER_SHARD" --seed "$SEED_K" --size "$IMG_W" "$IMG_H" \
                --composites "$COMPOSITES" --disturb "$DISTURB" --force "$FORCE" --vis \
                > "$RO/collect_$k.log" 2>&1 &
        fi
    done
    wait
    N_EP=$(ls "$RD" | wc -l | tr -d ' ')
    DRIVE_SUCC=$(grep -h "^done" "$RO"/collect_*.log | awk '{s+=$5; n++} END {if (n) printf "%.2f", s/n; else print "nan"}')
    echo "round $r: $N_EP episodes in $(( $(date +%s) - T0 )) s, driving success $DRIVE_SUCC"
    DATA_DIRS="$DATA_DIRS $RD"

    EP=$EPOCHS; [ "$r" -eq 0 ] && EP=$(( EPOCHS * 2 ))
    echo "== round $r: train $EP epochs on$DATA_DIRS =="
    T0=$(date +%s)
    # shellcheck disable=SC2086
    python $PY/student.py train --data $DATA_DIRS --out "$RO" --epochs "$EP" --batch "$BATCH" --device "$DEVICE" \
        ${PREV:+--init "$PREV"} 2>&1 | tee "$RO/train.log" | grep -E "ticks from|^epoch +[0-9]*[05]:|best val"
    PREV="$RO/student.pt"
    echo "round $r: trained in $(( $(date +%s) - T0 )) s"

    echo "== round $r: evaluate =="
    python $PY/student.py eval "$PREV" --n "$EVAL_N" --seed 50000 --size "$IMG_W" "$IMG_H" \
        --composites "$COMPOSITES" --disturb "$DISTURB" --force "$FORCE" 2>&1 | tee "$RO/eval_mix.log" | tail -1
    python $PY/student.py eval "$PREV" --n "$EVAL_N" --seed 51000 --size "$IMG_W" "$IMG_H" \
        --composites 0 --disturb 0 --force 0 2>&1 | tee "$RO/eval_plain.log" | tail -1
    {
        echo "round $r  episodes $N_EP  driving_success $DRIVE_SUCC  teacher_share $BETA"
        echo "  mix:   $(tail -1 "$RO/eval_mix.log")"
        echo "  plain: $(tail -1 "$RO/eval_plain.log")"
    } >> "$SUMMARY"
    cp "$PREV" "$OUT/student.pt"
done

echo "== done =="
cat "$SUMMARY"
