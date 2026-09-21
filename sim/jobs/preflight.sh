#!/usr/bin/env bash
# Proves a GPU node can train on this repo: imports, video decoding, backbone
# weights, and a real five-step lerobot-train. Run it once on a new node
# before queueing a long training job.
set -euo pipefail

: "${DATASET_ROOT:?repository-relative path of the LeRobot dataset to check}"
: "${DATASET_REPO:=}"             # optional Hub dataset repo, downloaded if the root is incomplete
export HF_TOKEN="${HF_TOKEN:-}"    # only needed when DATASET_REPO is private
: "${DATASET_ARCHIVE:=}"          # optional tar of the dataset, unpacked if the root is incomplete
: "${STEPS:=5}"
: "${JOB_ID:=local}"
: "${REPO_ID:=galaxeo/a1x_pour_sim}"
: "${ALLOW_CPU:=0}"          # 1: report a missing GPU instead of failing (local runs)
: "${GPUS:=1}"
: "${ON_FAILURE:=The node cannot train: check the venv, the dataset path, or network access to the PyTorch CDN.}"

# Cache the ResNet18 backbone weights inside the checkout so a node without a
# warm home directory downloads them once and every later job reuses them.
export TORCH_HOME="${TORCH_HOME:-sim/runs/torch_home}"

# shellcheck disable=SC1091
. "$(dirname "${BASH_SOURCE[0]}")/dataset.sh"

OUT="sim/runs/preflight_${JOB_ID}"
mkdir -p "$TORCH_HOME" "$(dirname "$OUT")"
rm -rf "$OUT"          # throwaway smoke-test output; lerobot-train refuses a non-empty one

echo "== versions =="
python -c 'import sys; print("python", sys.version.split()[0])'

DEVICE=cuda
if ! python - "$ALLOW_CPU" <<'PY'
import sys
import torch
print("torch", torch.__version__)
ok = torch.cuda.is_available()
print("cuda available:", ok)
if ok:
    print("cuda devices:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
    print("cuda runtime:", torch.version.cuda)
sys.exit(0 if ok else 1)
PY
then
    if [ "$ALLOW_CPU" = "1" ]; then
        echo "no CUDA device; ALLOW_CPU=1, continuing on cpu"
        DEVICE=cpu
    else
        echo "no CUDA device visible to torch"
        exit 1
    fi
fi

python -c 'import lerobot; print("lerobot", lerobot.__version__)'
python -c 'import mujoco; print("mujoco", mujoco.__version__)'
python -c 'import av; print("av", av.__version__)'

echo "== dataset and pyav decoding =="
python - "$DATASET_ROOT" "$REPO_ID" <<'PY'
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root, repo_id = sys.argv[1], sys.argv[2]
ds = LeRobotDataset(repo_id, root=root, video_backend="pyav")
print("episodes:", ds.num_episodes, "frames:", ds.num_frames, "fps:", ds.fps)
print("features:", ", ".join(sorted(ds.meta.features)))
n = min(4, len(ds))
for i in range(n):
    item = ds[i]
    shapes = {k: tuple(v.shape) for k, v in item.items() if hasattr(v, "shape")}
    if i == 0:
        print("frame 0 shapes:", shapes)
img_keys = [k for k in ds.meta.features if k.startswith("observation.images.")]
for k in img_keys:
    t = ds[0][k]
    print(f"decoded {k}: {tuple(t.shape)} {t.dtype} range [{float(t.min()):.3f}, {float(t.max()):.3f}]")
print(f"decoded {n} frames with the pyav backend: ok")
PY

echo "== backbone weights =="
if ! python - <<'PY'
import sys
try:
    from torchvision.models import ResNet18_Weights, resnet18
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    print("resnet18 IMAGENET1K_V1 loaded,", sum(p.numel() for p in m.parameters()), "parameters")
except Exception as exc:                                   # noqa: BLE001
    print("weight download failed:", type(exc).__name__, exc)
    sys.exit(1)
PY
then
    echo "no network for backbone weights"
    exit 1
fi

echo "== lerobot-train smoke test ($STEPS steps on $DEVICE) =="
lerobot-train \
    --policy.type=act \
    --dataset.repo_id="$REPO_ID" \
    --dataset.root="$DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --policy.device="$DEVICE" \
    --policy.push_to_hub=false \
    --output_dir="$OUT" \
    --job_name="preflight_${JOB_ID}" \
    --steps="$STEPS" \
    --batch_size=2 \
    --num_workers=0 \
    --log_freq=1 \
    --save_freq="$STEPS" \
    --wandb.enable=false

echo "preflight ok: trained $STEPS steps on $DEVICE, checkpoint under $OUT"
