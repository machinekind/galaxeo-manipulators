#!/usr/bin/env bash
# One ACT training run on a pour dataset. Everything it writes lands under
# sim/runs/$RUN_NAME in the checkout; nothing else is touched.
set -euo pipefail

: "${DATASET_ROOT:?repository-relative path of the LeRobot dataset to train on}"
: "${DATASET_ARCHIVE:=}"          # optional tar of the dataset, unpacked if the root is incomplete
: "${JOB_ID:=local}"
: "${RUN_NAME:=pour_act_${JOB_ID}}"
: "${REPO_ID:=galaxeo/a1x_pour_sim}"   # dataset id; the local root wins, see README
: "${STEPS:=100000}"
: "${BATCH:=32}"
: "${CHUNK:=50}"                       # chunk_size and n_action_steps
: "${LR:=1e-5}"
: "${SEED:=0}"
: "${WORKERS:=4}"
: "${RESUME:=false}"
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
    # legitimately change on a restart are passed again.
    CKPT="$OUT/checkpoints/last/pretrained_model/train_config.json"
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
    ARGS=(
        --policy.type=act
        --dataset.repo_id="$REPO_ID"
        --dataset.root="$DATASET_ROOT"
        --dataset.video_backend=pyav
        --dataset.image_transforms.enable=true
        --dataset.image_transforms.tfs="$TFS"
        --policy.chunk_size="$CHUNK"
        --policy.n_action_steps="$CHUNK"
        --policy.optimizer_lr="$LR"
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

lerobot-train "${ARGS[@]}"
echo "training done: checkpoints under $OUT/checkpoints"
