#!/usr/bin/env bash
# The state-based floor check on one machine: generate a dataset of the true
# scene state (no images), train ACT on it, evaluate it closed loop. Everything
# lands under sim/runs/$RUN_NAME. Generation and evaluation are CPU work, one
# process per core; training is the GPU's.
set -euo pipefail

: "${JOB_ID:=local}"
: "${RUN_NAME:=pour_state_${JOB_ID}}"
: "${EPISODES:=1000}"                 # successful episodes wanted
: "${SEED0:=10000}"                   # first scene seed; 2000-2019 are the held-out evaluation
: "${STEPS:=50000}"
: "${BATCH:=32}"
: "${CHUNK:=50}"
: "${LR:=1e-4}"                       # a state-only ACT fits faster than the image recipe's 1e-5
: "${VAE:=false}"                     # ACT's variational objective; off, the decoder has only the observation to go on
: "${SAVE_FREQ:=10000}"
: "${EVAL_N:=20}"                     # episodes per evaluation range
: "${EVAL_SEEDS:=2000 1000}"          # first seed of each evaluation range
: "${GEN_WORKERS:=0}"                 # 0: the box's cores (cgroup quota, not the host's count) minus 2
: "${GPUS:=1}"
: "${ON_FAILURE:=State floor check stopped; sim/runs/<run> holds whatever stage finished.}"
export TORCH_HOME="${TORCH_HOME:-sim/runs/torch_home}"
export HF_HUB_OFFLINE=1
# Headless rendering (the evaluation renders the cameras), one BLAS thread per
# worker, and the core count the box was sold with.
# shellcheck disable=SC1091
. "$(dirname "${BASH_SOURCE[0]}")/render.sh"
[ "$GEN_WORKERS" -gt 0 ] || GEN_WORKERS=$(( $(cores_available) - 2 ))

OUT="sim/runs/${RUN_NAME}"
DATA="sim/datasets/${RUN_NAME}"
LOGS="sim/runs/${RUN_NAME}_logs"          # lerobot-train wants its output directory not to exist yet
mkdir -p "$LOGS" sim/runs
t0=$(date +%s)

if [ ! -f "$DATA/merged/meta/info.json" ]; then
    echo "== generating $EPISODES state episodes from seed $SEED0 on $GEN_WORKERS workers"
    python sim/gen_dataset.py --root "$DATA" --episodes "$EPISODES" --seed0 "$SEED0" --state \
        --workers "$GEN_WORKERS" --repo-id "galaxeo/${RUN_NAME}" 2>&1 | grep -vE "^Svt|torchcodec|dlopen|OSError|Traceback|raise OSError|^ +File|libavutil" \
        | tee "$LOGS/gen.log" | grep -E "^\[w[0-9]+\] +[0-9]+/ *[0-9]+ ok|episode\(s\)|merging" | tail -40
fi
echo "generation done at $(( $(date +%s) - t0 )) s"

if [ ! -f "$OUT/checkpoints/last/pretrained_model/config.json" ]; then
    echo "== training ACT, $STEPS steps"
    rm -rf "$OUT"
    # lerobot's ACT maps no normalisation onto ENV features, so without the
    # explicit entry the scene state would go in raw (metres, in the
    # hundredths) beside standardised joints and actions, and be all but
    # ignored: that is what the first floor check did.
    lerobot-train --policy.type=act --policy.chunk_size="$CHUNK" --policy.n_action_steps="$CHUNK" \
        --policy.optimizer_lr="$LR" --policy.use_vae="$VAE" \
        --policy.normalization_mapping='{"VISUAL":"MEAN_STD","STATE":"MEAN_STD","ACTION":"MEAN_STD","ENV":"MEAN_STD"}' \
        --dataset.repo_id="galaxeo/${RUN_NAME}" --dataset.root="$DATA/merged" \
        --policy.device=cuda --policy.push_to_hub=false --output_dir="$OUT" --job_name="$RUN_NAME" \
        --steps="$STEPS" --batch_size="$BATCH" --num_workers=8 --save_freq="$SAVE_FREQ" --log_freq=500 \
        --seed=0 --resume=false --wandb.enable=false 2>&1 | tr '\r' '\n' | tee "$LOGS/train.log" | grep -E "step:|Error|Traceback|End of" | tail -30
fi
echo "training done at $(( $(date +%s) - t0 )) s"

echo "== evaluating $EVAL_N episodes from each of: $EVAL_SEEDS (one process per 5 seeds)"
mkdir -p "$OUT/eval"
CKPT="$OUT/checkpoints/last/pretrained_model"
pids=()
for s0 in $EVAL_SEEDS; do
    for k in $(seq 0 5 $(( EVAL_N - 1 ))); do
        s=$(( s0 + k )); n=$(( EVAL_N - k < 5 ? EVAL_N - k : 5 ))
        python sim/planner/eval_policy.py --ckpt "$CKPT" --n "$n" --seed "$s" --device cpu \
            --video "$OUT/eval/video" --json "$OUT/eval/seeds_${s}.json" > "$OUT/eval/seeds_${s}.log" 2>&1 &
        pids+=($!)
    done
done
for p in "${pids[@]}"; do wait "$p" || true; done
python - "$OUT/eval" $EVAL_SEEDS "$EVAL_N" <<'PY'
import glob, json, sys, collections
d, n = sys.argv[1], int(sys.argv[-1])
rows = []
for f in sorted(glob.glob(f"{d}/seeds_*.json")):
    rows += json.load(open(f))["episodes"]
for s0 in map(int, sys.argv[2:-1]):
    eps = [e for e in rows if s0 <= e["seed"] < s0 + n]
    ok = sum(e["success"] for e in eps)
    mix = collections.Counter("success" if e["success"] else (e.get("trouble") or ("timeout, carried" if e.get("carried") else "timeout, never touched")) for e in eps)
    print(f"seeds {s0}-{s0+n-1}: {ok}/{len(eps)} success  {dict(mix)}")
    for e in eps:
        print(f"  seed {e['seed']} {'SUCCESS' if e['success'] else 'FAIL'} poured {e['secs']:.1f}s trouble={e.get('trouble')} t={e.get('t_trouble')}")
json.dump(rows, open(f"{d}/all.json", "w"), indent=1)
PY
echo "all done at $(( $(date +%s) - t0 )) s"
