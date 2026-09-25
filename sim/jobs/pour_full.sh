#!/usr/bin/env bash
# The image run on one machine: generate a pour dataset with the laptop camera,
# the top-down map and the wrist camera, train ACT on it, evaluate closed loop,
# and evaluate a control checkpoint on the same seeds. Everything lands under
# sim/runs/$RUN_NAME. Generation and evaluation are CPU work, one process per
# core; training is the GPU's (through train_pour.sh, so the recipe is the one
# every earlier run used).
set -euo pipefail

: "${JOB_ID:=local}"
: "${RUN_NAME:=pour_full_${JOB_ID}}"
: "${EPISODES:=3000}"                 # successful episodes wanted
: "${SEED0:=10000}"                   # first scene seed; 2000-2019 are the held-out evaluation
: "${OVER:=1.6}"                      # seeds drawn per wanted episode (noise lowers the planner's rate)
: "${VARY:=1}"                        # 1: the grasp is drawn among the feasible candidates
: "${SERVO_NOISE:=1.5}"               # std [deg] of the drift added to the executed joint commands
: "${WRIST:=left}"                    # hand carrying the wrist camera
: "${STEPS:=100000}"                 # the state floor kept improving to 100k; the loss had not plateaued
: "${BATCH:=32}"
: "${CHUNK:=50}"
: "${LR:=1e-4}"                       # the transformer's rate; the backbone keeps lerobot's 1e-5
: "${VAE:=false}"                     # off, as in the state floor check that passed
: "${WORKERS:=16}"                    # dataloader workers
: "${INIT_FROM:=}"                    # optional warm start (pretrained_model dir or Hub model id)
: "${DATASET_REPO:=}"                 # optional: a Hub dataset to use instead of generating
: "${PUSH_DATASET:=}"                 # optional: private Hub dataset repo to push the merged set to
: "${PUSH_MODEL:=}"                   # optional: private Hub model repo to push the final weights to
: "${CONTROL_CKPT:=}"                 # optional: Hub model id (or repo-relative dir) evaluated as the control
export HF_TOKEN="${HF_TOKEN:-}"       # for the private repos above
: "${EVAL_N:=20}"                     # episodes per evaluation range
: "${EVAL_SEEDS:=2000 1000}"          # first seed of each evaluation range
: "${GEN_WORKERS:=0}"                 # 0: the box's cores (cgroup quota, not the host's count) minus 2
: "${GPUS:=1}"
: "${ON_FAILURE:=Image run stopped; sim/runs/<run> holds whatever stage finished, sim/datasets/<run>/merged the data.}"
export TORCH_HOME="${TORCH_HOME:-sim/runs/torch_home}"

# shellcheck disable=SC1091
. "$(dirname "${BASH_SOURCE[0]}")/render.sh"
[ "$GEN_WORKERS" -gt 0 ] || GEN_WORKERS=$(( $(cores_available) - 2 ))

OUT="sim/runs/${RUN_NAME}"
DATA="sim/datasets/${RUN_NAME}"
LOGS="sim/runs/${RUN_NAME}_logs"
mkdir -p "$LOGS" sim/runs
t0=$(date +%s)
elapsed() { echo "$1 at $(( ($(date +%s) - t0) / 60 )) min"; }

# ------------------------------------------------------------------ dataset
if [ ! -f "$DATA/merged/meta/info.json" ] && [ -n "$DATASET_REPO" ]; then
    echo "== downloading $DATASET_REPO"
    mkdir -p "$DATA/merged"
    HF_HUB_OFFLINE=0 hf download "$DATASET_REPO" --repo-type dataset --local-dir "$DATA/merged" >/dev/null
fi
if [ ! -f "$DATA/merged/meta/info.json" ]; then
    GEN=(--wrist "$WRIST" --over "$OVER")
    [ "$VARY" = 1 ] && GEN+=(--vary)
    [ "$(echo "$SERVO_NOISE > 0" | bc -l 2>/dev/null || echo 1)" = 1 ] && GEN+=(--servo-noise "$SERVO_NOISE")
    echo "== generating $EPISODES episodes from seed $SEED0 on $GEN_WORKERS workers: ${GEN[*]}"
    python sim/gen_dataset.py --root "$DATA" --episodes "$EPISODES" --seed0 "$SEED0" \
        --workers "$GEN_WORKERS" --repo-id "galaxeo/${RUN_NAME}" "${GEN[@]}" 2>&1 \
        | grep -vE "^Svt|torchcodec|dlopen|OSError|Traceback|raise OSError|^ +File|libavutil" \
        | tee "$LOGS/gen.log" | grep -E "^\[w[0-9]+\] +[0-9]+/ *[0-9]+ ok|episode\(s\)|merging|wanted" | tail -60
fi
elapsed "generation done"
if [ -n "$PUSH_DATASET" ] && [ ! -f "$LOGS/pushed_dataset" ]; then
    echo "== pushing the dataset to $PUSH_DATASET"
    HF_HUB_OFFLINE=0 hf repo create "$PUSH_DATASET" --repo-type dataset --private >/dev/null 2>&1 || true
    HF_HUB_OFFLINE=0 hf upload "$PUSH_DATASET" "$DATA/merged" . --repo-type dataset >/dev/null \
        && touch "$LOGS/pushed_dataset" || echo "WARN: dataset push failed" >&2
    elapsed "dataset push done"
fi

# ----------------------------------------------------------------- training
if [ ! -f "$OUT/checkpoints/last/pretrained_model/config.json" ]; then
    echo "== training ACT, $STEPS steps (train_pour.sh)"
    DATASET_ROOT="$DATA/merged" REPO_ID="galaxeo/${RUN_NAME}" RUN_NAME="$RUN_NAME" STEPS="$STEPS" BATCH="$BATCH" \
        CHUNK="$CHUNK" LR="$LR" VAE="$VAE" WORKERS="$WORKERS" INIT_FROM="$INIT_FROM" PIN_CPUS="$(cores_available)" \
        OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
        bash sim/jobs/train_pour.sh 2>&1 | tr '\r' '\n' | tee "$LOGS/train.log" \
        | grep -E "step:|Error|Traceback|training|warm start|run " | tail -40
fi
elapsed "training done"
CKPT="$OUT/checkpoints/last/pretrained_model"
if [ -n "$PUSH_MODEL" ] && [ ! -f "$LOGS/pushed_model" ]; then
    echo "== pushing the weights to $PUSH_MODEL"
    HF_HUB_OFFLINE=0 hf repo create "$PUSH_MODEL" --repo-type model --private >/dev/null 2>&1 || true
    HF_HUB_OFFLINE=0 hf upload "$PUSH_MODEL" "$CKPT" . --repo-type model >/dev/null \
        && touch "$LOGS/pushed_model" || echo "WARN: model push failed" >&2
fi

# --------------------------------------------------------------- evaluation
evaluate() {          # evaluate CKPT_DIR EVAL_DIR
    local ckpt="$1" dir="$2" s0 k s n
    mkdir -p "$dir"
    local pids=()
    for s0 in $EVAL_SEEDS; do
        for k in $(seq 0 5 $(( EVAL_N - 1 ))); do
            s=$(( s0 + k )); n=$(( EVAL_N - k < 5 ? EVAL_N - k : 5 ))
            python sim/planner/eval_policy.py --ckpt "$ckpt" --n "$n" --seed "$s" --device cpu --wrist "$WRIST" \
                --video "$dir/video" --json "$dir/seeds_${s}.json" > "$dir/seeds_${s}.log" 2>&1 &
            pids+=($!)
        done
    done
    for p in "${pids[@]}"; do wait "$p" || true; done
    # shellcheck disable=SC2086
    python sim/planner/eval_summary.py "$dir" $EVAL_SEEDS --n "$EVAL_N"
}

echo "== evaluating $RUN_NAME: $EVAL_N episodes from each of: $EVAL_SEEDS"
evaluate "$CKPT" "$OUT/eval"
elapsed "evaluation done"

if [ -n "$CONTROL_CKPT" ]; then
    if [ ! -d "$CONTROL_CKPT" ]; then
        CDIR="sim/runs/control/${CONTROL_CKPT//\//__}"
        if [ ! -f "$CDIR/model.safetensors" ]; then
            echo "== fetching the control checkpoint $CONTROL_CKPT"
            HF_HUB_OFFLINE=0 hf download "$CONTROL_CKPT" --repo-type model --local-dir "$CDIR" >/dev/null
        fi
        CONTROL_CKPT="$CDIR"
    fi
    echo "== evaluating the control checkpoint on the same seeds"
    evaluate "$CONTROL_CKPT" "$OUT/eval_control"
    elapsed "control evaluation done"
fi
echo "all done at $(( ($(date +%s) - t0) / 60 )) min"
