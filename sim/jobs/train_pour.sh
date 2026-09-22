#!/usr/bin/env bash
# One ACT training run on a pour dataset. Everything it writes lands under
# sim/runs/$RUN_NAME in the checkout; nothing else is touched.
set -euo pipefail

: "${DATASET_ROOT:?repository-relative path of the LeRobot dataset to train on}"
: "${DATASET_REPO:=}"             # optional Hub dataset repo, downloaded if the root is incomplete
export HF_TOKEN="${HF_TOKEN:-}"    # only needed when DATASET_REPO is private
: "${DATASET_ARCHIVE:=}"          # optional tar of the dataset, unpacked if the root is incomplete
: "${JOB_ID:=local}"
: "${RUN_NAME:=pour_act_${JOB_ID}}"
: "${REPO_ID:=galaxeo/a1x_pour_sim}"   # dataset id; the local root wins, see README
: "${STEPS:=100000}"
: "${BATCH:=32}"
: "${CHUNK:=50}"                       # chunk_size and n_action_steps
: "${LR:=1e-5}"
: "${VAE:=true}"                  # ACT's variational objective
: "${SEED:=0}"
: "${WORKERS:=4}"
: "${RESUME:=false}"
: "${RESUME_ARCHIVE:=}"           # optional tar of checkpoints/ to resume from when the run dir is absent
: "${INIT_FROM:=}"                # optional pretrained_model dir or Hub model id to warm-start a new run from
: "${PIN_CPUS:=0}"                # >0: run on that many CPUs only (taskset), see below
: "${RETRIES:=5}"                 # times a crashed run is picked up again from its last checkpoint
: "${WANDB:=false}"
: "${GPUS:=1}"
: "${ON_FAILURE:=ACT training stopped; the last checkpoint under sim/runs is still there and RESUME=true picks it up.}"

# At most ~10 checkpoints over the whole run.
: "${SAVE_FREQ:=$(( STEPS / 10 < 1000 ? 1000 : STEPS / 10 ))}"
: "${LOG_FREQ:=100}"

# Keep the ResNet18 backbone weights in the checkout, so the first job on a
# node downloads them and every later job on any node reuses them.
export TORCH_HOME="${TORCH_HOME:-sim/runs/torch_home}"

# shellcheck disable=SC1091
. "$(dirname "${BASH_SOURCE[0]}")/dataset.sh"

OUT="sim/runs/${RUN_NAME}"

# A Hub model id in INIT_FROM is fetched once into the checkout, before the
# loader goes offline for good, so the warm start below reads a local directory
# either way.
if [ -n "$INIT_FROM" ] && [ ! -d "$INIT_FROM" ]; then
    INIT_DIR="sim/runs/init/${INIT_FROM//\//__}"
    if [ ! -f "$INIT_DIR/model.safetensors" ]; then
        echo "fetching warm-start weights $INIT_FROM"
        HF_HUB_OFFLINE=0 hf download "$INIT_FROM" --repo-type model --local-dir "$INIT_DIR" >/dev/null
        find "$INIT_DIR" -name '._*' -delete 2>/dev/null || true
    fi
    INIT_FROM="$INIT_DIR"
fi
mkdir -p "$TORCH_HOME" sim/runs

# Photometric jitter only. `--dataset.image_transforms.tfs` REPLACES the whole
# default set, and the default set contains a RandomAffine that rotates and
# shifts the frame -- which would break the fixed relation between the image
# and the per-frame camera calibration the dataset carries. These five are the
# stock brightness/contrast/saturation/hue/sharpness entries, affine dropped.
TFS='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},
      "contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},
      "saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.5,1.5]}},
      "hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},
      "sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}}}'

if [ "$RESUME" = "true" ]; then
    # Resuming reads everything back from the checkpoint's own train_config.json
    # -- lerobot refuses --resume=true without it -- so only the knobs that may
    # legitimately change on a restart are passed again. Run directories do
    # not travel with a checkout sync, so a checkpoint to resume from arrives
    # the way the dataset does: as one archive, unpacked into the run dir.
    CKPT="$OUT/checkpoints/last/pretrained_model/train_config.json"
    if [ ! -f "$CKPT" ] && [ -n "$RESUME_ARCHIVE" ] && [ -f "$RESUME_ARCHIVE" ]; then
        echo "no checkpoint under $OUT; unpacking $RESUME_ARCHIVE"
        mkdir -p "$OUT"
        tar -C "$OUT" -xf "$RESUME_ARCHIVE"
        find "$OUT" -name '._*' -delete 2>/dev/null || true
    fi
    [ -f "$CKPT" ] || { echo "nothing to resume: no checkpoint under $OUT"; exit 1; }
    ARGS=(
        --config_path="$CKPT"
        --resume=true
        --steps="$STEPS"
        --save_freq="$SAVE_FREQ"
        --log_freq="$LOG_FREQ"
    )
    echo "resuming $RUN_NAME from its last checkpoint, up to $STEPS steps"
else
    # A warm start loads the policy config and weights from INIT_FROM (a
    # pretrained_model directory or a Hub model id) with a fresh optimizer;
    # the architecture and learning rate then come from that config, not
    # from CHUNK and LR. A fresh run builds an ACT from those instead.
    if [ -n "$INIT_FROM" ]; then
        POLICY=(--policy.path="$INIT_FROM")
        echo "warm start from $INIT_FROM"
    else
        POLICY=(--policy.type=act --policy.chunk_size="$CHUNK" --policy.n_action_steps="$CHUNK"
                --policy.optimizer_lr="$LR" --policy.use_vae="$VAE"
                --policy.normalization_mapping='{"VISUAL":"MEAN_STD","STATE":"MEAN_STD","ACTION":"MEAN_STD","ENV":"MEAN_STD"}')
    fi
    ARGS=(
        "${POLICY[@]}"
        --dataset.repo_id="$REPO_ID"
        --dataset.root="$DATASET_ROOT"
        --dataset.video_backend=pyav
        --dataset.image_transforms.enable=true
        --dataset.image_transforms.tfs="$TFS"
        --policy.device=cuda
        --policy.push_to_hub=false
        --output_dir="$OUT"
        --job_name="$RUN_NAME"
        --steps="$STEPS"
        --batch_size="$BATCH"
        --num_workers="$WORKERS"
        --save_freq="$SAVE_FREQ"
        --log_freq="$LOG_FREQ"
        --seed="$SEED"
        --resume=false
    )
    echo "run $RUN_NAME: $STEPS steps, batch $BATCH, chunk $CHUNK, lr $LR, seed $SEED, save every $SAVE_FREQ"
fi

if [ "$WANDB" = "true" ]; then
    # WANDB_MODE / WANDB_DIR / WANDB_API_KEY come from the caller's environment.
    ARGS+=(--wandb.enable=true)
else
    ARGS+=(--wandb.enable=false)
fi

# A run is hours on a rented machine, and a loader worker can die of something
# that has nothing to do with the run (a rented host once raised "'Tensor'
# object is not iterable" out of torchvision's `for i in torch.randperm(4)` at
# step 610). So a failed run is continued from its last checkpoint, or started
# again if it never wrote one, up to RETRIES times, instead of idling a GPU.
#
# PIN_CPUS: every PyAV decoder starts one thread per CPU it can see, and a
# rented container on a 128-core host sees all of them whatever share it was
# sold: 32 loader workers then want 4096 threads and avcodec_open2 fails with
# "Cannot allocate memory". An affinity mask is what the decoder counts, so
# pinning the run to the cores it actually has keeps that product sane.
RUN=()
if [ "$PIN_CPUS" -gt 0 ] && command -v taskset >/dev/null 2>&1; then
    RUN=(taskset -c "0-$(( PIN_CPUS - 1 ))")
    echo "pinned to $PIN_CPUS CPUs"
fi
try=0
until ${RUN[@]+"${RUN[@]}"} lerobot-train "${ARGS[@]}"; do
    try=$(( try + 1 ))
    [ "$try" -le "$RETRIES" ] || { echo "training failed $try times; giving up"; exit 1; }
    LAST="$OUT/checkpoints/last/pretrained_model/train_config.json"
    if [ -f "$LAST" ]; then
        echo "training failed (attempt $try of $RETRIES): resuming from the last checkpoint"
        ARGS=(--config_path="$LAST" --resume=true --steps="$STEPS" --save_freq="$SAVE_FREQ" --log_freq="$LOG_FREQ")
    else
        echo "training failed (attempt $try of $RETRIES) before its first checkpoint: starting again"
        rm -rf "$OUT"
    fi
done
echo "training done: checkpoints under $OUT/checkpoints"
