#!/usr/bin/env bash
# One DAgger round on one machine: roll a policy out on fresh scenes, let the
# planner take over from the states it reaches (gen_dagger.py), merge the
# recoveries with the base dataset, warm-start training from the rolled-out
# weights, evaluate the result and the rolled-out policy on the same seeds.
# Everything lands under sim/runs/$RUN_NAME; the data under sim/datasets.
set -euo pipefail

: "${JOB_ID:=local}"
: "${RUN_NAME:=pour_dagger_${JOB_ID}}"
: "${POLICY:?Hub model id or repo-relative pretrained_model dir of the policy to roll out}"
: "${BASE_DATASET:?Hub dataset id (or repo-relative root) of the set the recoveries are merged with}"
: "${SEEDS:=1500}"                    # scenes to roll out
: "${SEED0:=5000}"                    # first scene seed; 2000-2019 are refused by the collector
: "${TAKEOVERS:=1}"                   # usable segments to harvest per scene
: "${REWIND:=1.0,2.5,5.0}"            # seconds before the trouble the scene is restored to, in order
: "${WRIST:=left}"                    # hand carrying the wrist camera
: "${STEPS:=30000}"                   # training steps of the warm-started run
: "${BATCH:=32}"
: "${WORKERS:=16}"                    # dataloader workers
: "${PUSH_DATASET:=}"                 # optional: private Hub dataset repo to push the merged set to
: "${PUSH_MODEL:=}"                   # optional: private Hub model repo to push the final weights to
: "${CONTROL_CKPT:=}"                 # optional second checkpoint evaluated on the same seeds; default: POLICY
export HF_TOKEN="${HF_TOKEN:-}"       # for the private repos above
: "${EVAL_N:=20}"                     # episodes per evaluation range
: "${EVAL_SEEDS:=2000 1000}"          # first seed of each evaluation range
: "${GEN_WORKERS:=0}"                 # 0: the box's cores (cgroup quota, not the host's count) minus 2
: "${GPUS:=1}"
: "${ON_FAILURE:=DAgger round stopped; sim/runs/<run> holds whatever stage finished, sim/datasets/<run> the data.}"
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

fetch_model() {       # fetch_model ID_OR_DIR -> prints a local pretrained_model dir
    local id="$1" dir
    if [ -d "$id" ]; then echo "$id"; return; fi
    dir="sim/runs/init/${id//\//__}"
    if [ ! -f "$dir/model.safetensors" ]; then
        echo "fetching $id" >&2
        HF_HUB_OFFLINE=0 hf download "$id" --repo-type model --local-dir "$dir" >/dev/null
        find "$dir" -name '._*' -delete 2>/dev/null || true
    fi
    echo "$dir"
}

# --------------------------------------------------------------- base data
if [ -d "$BASE_DATASET" ]; then
    BASE="$BASE_DATASET"
else
    BASE="sim/datasets/${BASE_DATASET//\//__}/merged"
    if [ ! -f "$BASE/meta/info.json" ]; then
        echo "== downloading the base dataset $BASE_DATASET"
        mkdir -p "$BASE"
        HF_HUB_OFFLINE=0 hf download "$BASE_DATASET" --repo-type dataset --local-dir "$BASE" >/dev/null
    fi
fi
POLICY_DIR="$(fetch_model "$POLICY")"
elapsed "inputs ready"

# --------------------------------------------------------------- collection
if [ ! -f "$DATA/merged/meta/info.json" ]; then
    echo "== rolling out $POLICY on $SEEDS scenes from seed $SEED0 on $GEN_WORKERS workers"
    python sim/gen_dagger.py --ckpt "$POLICY_DIR" --root "$DATA" --seeds "$SEEDS" --seed0 "$SEED0" \
        --workers "$GEN_WORKERS" --base "$BASE" --wrist "$WRIST" --takeovers "$TAKEOVERS" --rewind "$REWIND" \
        --repo-id "galaxeo/${RUN_NAME}" 2>&1 \
        | grep -vE "^Svt|torchcodec|dlopen|OSError|Traceback|raise OSError|^ +File|libavutil" \
        | tee "$LOGS/gen.log" | grep -E "^\[w[0-9]+\] +[0-9]+/ *[0-9]+ ok|episode\(s\)|correction|merging|worker" | tail -60
    cp "$DATA/dagger_log.jsonl" "$LOGS/" 2>/dev/null || true
fi
elapsed "collection done"
if [ -n "$PUSH_DATASET" ] && [ ! -f "$LOGS/pushed_dataset" ]; then
    echo "== pushing the merged dataset to $PUSH_DATASET"
    HF_HUB_OFFLINE=0 hf repo create "$PUSH_DATASET" --repo-type dataset --private >/dev/null 2>&1 || true
    HF_HUB_OFFLINE=0 hf upload "$PUSH_DATASET" "$DATA/merged" . --repo-type dataset >/dev/null \
        && touch "$LOGS/pushed_dataset" || echo "WARN: dataset push failed" >&2
    elapsed "dataset push done"
fi

# ----------------------------------------------------------------- training
if [ ! -f "$OUT/checkpoints/last/pretrained_model/config.json" ]; then
    echo "== training ACT from $POLICY, $STEPS steps (train_pour.sh)"
    DATASET_ROOT="$DATA/merged" REPO_ID="galaxeo/${RUN_NAME}" RUN_NAME="$RUN_NAME" STEPS="$STEPS" BATCH="$BATCH" \
        WORKERS="$WORKERS" INIT_FROM="$POLICY_DIR" PIN_CPUS="$(cores_available)" \
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
CONTROL_DIR="$(fetch_model "${CONTROL_CKPT:-$POLICY}")"
echo "== evaluating the rolled-out policy (control) on the same seeds"
evaluate "$CONTROL_DIR" "$OUT/eval_control"
echo "all done at $(( ($(date +%s) - t0) / 60 )) min"
