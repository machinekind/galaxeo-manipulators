# GPU payloads

Each file here is a self-contained bash script that does one job on a machine
with GPUs. They are written for an out-of-tree dispatcher, but there is nothing
in them that needs it: a payload is an ordinary script you can also run by hand.

**The contract.** A payload is `#!/usr/bin/env bash` with `set -euo pipefail`.
It runs from the root of the checkout, so every path it names is
repository-relative, and it writes its outputs only under `sim/runs/`. It takes
every parameter from the environment and declares them all at the top, in one
block, so that reading the first twenty lines tells you how to call it:

    : "${NAME:?what it is}"       # required; the script aborts if unset
    : "${NAME:=default}"          # optional, with a default
    export NAME="${NAME:-default}"  # optional, and passed on to child processes

Two of those names are read by the dispatcher rather than by the script:
`GPUS`, how many GPUs the job wants, and `ON_FAILURE`, one sentence saying what
a human should do if the job dies. Domain environment (`TORCH_HOME` and
friends) is declared by the payload itself. No hostnames, no absolute paths,
no scheduler flags: all of that belongs to the caller.

The environment is whatever the caller activated. The dispatcher syncs a venv
from `sim/pyproject.toml` + `sim/uv.lock` (`uv sync --frozen --no-dev`), puts
`sim/` on `PYTHONPATH`, and makes the GPUs visible; by hand, activate a venv
built the same way and run the script from the checkout root.

`TORCH_HOME` defaults to `sim/runs/torch_home` in both payloads, so the
ResNet18 backbone weights are downloaded once per checkout instead of once per
job, and survive a node with an empty home directory. Training does no
rendering, so no `MUJOCO_GL` is needed.

## The dataset travels as an archive

Both payloads source `dataset.sh` first. A checkout sync uses exclude
patterns that belong to whoever runs it, and a pattern such as `*.mp4` strips
a dataset's videos on the way without a word. So the dataset is also packed
as one tar of its root's last directory component, named by
`DATASET_ARCHIVE`; when the root is missing or incomplete and the archive is
present, it is unpacked next to the root. The loader is forced offline so a
dataset that is still missing fails right there instead of asking the Hub for
a repository of the same name.

    tar -C sim/datasets/pour_sim -cf sim/datasets/pour_sim.tar merged
    DATASET_ROOT=sim/datasets/pour_sim/merged DATASET_ARCHIVE=sim/datasets/pour_sim.tar bash sim/jobs/preflight.sh

## preflight.sh

Proves a node can actually train, in the order things usually break: python and
torch versions, the CUDA devices torch can see, `lerobot` and `mujoco` imports,
decoding a few dataset frames through the **pyav** backend, downloading the
ResNet18 IMAGENET1K_V1 weights (which is the test of outbound network access;
it prints `no network for backbone weights` and exits non-zero if that fails),
and finally a real short `lerobot-train`. Run it once on a new node.

| input | |
| --- | --- |
| `DATASET_ROOT` | **required**, repository-relative path of the dataset |
| `DATASET_REPO` | optional Hub dataset repository, downloaded when the root is missing or incomplete |
| `HF_TOKEN` | Hub token, only for a private `DATASET_REPO`; exported for the download |
| `DATASET_ARCHIVE` | optional tar of the dataset root, unpacked when the root is missing or incomplete |
| `STEPS` | training steps for the smoke test, default 5 |
| `JOB_ID` | names the output directory `sim/runs/preflight_<JOB_ID>`, default `local` |
| `REPO_ID` | dataset id, default `galaxeo/a1x_pour_sim` |
| `ALLOW_CPU` | `1`: report a missing GPU and train on the CPU instead of failing |
| `GPUS` | GPUs to ask for, default 1 |
| `ON_FAILURE` | one sentence for a human |

    DATASET_ROOT=data/pour_sim bash sim/jobs/preflight.sh

    ALLOW_CPU=1 DATASET_ROOT=data/pour_sim bash sim/jobs/preflight.sh   # laptop, no GPU

## train_pour.sh

One ACT training run. Writes to `sim/runs/$RUN_NAME`, checkpointing ten times
over the run; `RESUME=true` on the same `RUN_NAME` continues from the last one.
Image augmentation is on, but only the photometric half of it: passing
`--dataset.image_transforms.tfs` replaces the whole default transform set, and
the payload passes brightness, contrast, saturation, hue and sharpness while
leaving out the default `RandomAffine`, which would rotate and shift the frame
and so break the fixed relation between the image and the camera calibration
each frame carries.

Both training payloads pass an explicit normalisation mapping with an `ENV`
entry: lerobot's ACT maps none onto `observation.environment_state`, so
without it a state-based policy's scene input goes in raw (metres, in the
hundredths) beside standardised joints and actions, and is all but ignored.

| input | |
| --- | --- |
| `DATASET_ROOT` | **required**, repository-relative path of the dataset |
| `DATASET_REPO` | optional Hub dataset repository, downloaded when the root is missing or incomplete |
| `HF_TOKEN` | Hub token, only for a private `DATASET_REPO`; exported for the download |
| `DATASET_ARCHIVE` | optional tar of the dataset root, unpacked when the root is missing or incomplete |
| `RUN_NAME` | output name, default `pour_act_<JOB_ID>` |
| `REPO_ID` | dataset id, default `galaxeo/a1x_pour_sim` |
| `STEPS` | default 100000 |
| `BATCH` | default 32 |
| `CHUNK` | `chunk_size` and `n_action_steps`, default 50 |
| `LR` | default 1e-5 |
| `VAE` | ACT's variational objective, default true |
| `SEED` | default 0 |
| `WORKERS` | dataloader workers, default 4 |
| `RESUME` | default false |
| `RESUME_ARCHIVE` | optional tar of `checkpoints/` (with its `last` link), unpacked into the run dir when it has no checkpoint; run dirs do not travel with a sync |
| `INIT_FROM` | optional `pretrained_model` directory (repository-relative) or Hub model id; a new run starts from those weights and their config with a fresh optimizer, so `CHUNK` and `LR` are ignored. `HF_TOKEN` is exported for a private model repo |
| `PIN_CPUS` | default 0 (off). On a container that sees far more CPUs than it was given, pin the run to this many: each PyAV decoder starts a thread per visible CPU, and `WORKERS` times that can exceed the container's thread limit (`avcodec_open2 ... Cannot allocate memory`) |
| `RETRIES` | default 5: a `lerobot-train` that exits non-zero is continued from the run's last checkpoint (or started again if there is none yet) this many times before the payload gives up |
| `WANDB` | default false; `true` adds `--wandb.enable=true`. `WANDB_MODE`, `WANDB_DIR` and `WANDB_API_KEY` come from the caller |
| `SAVE_FREQ`, `LOG_FREQ` | default `STEPS/10` (min 1000) and 100 |
| `GPUS`, `ON_FAILURE` | for the dispatcher |

    DATASET_ROOT=data/pour_sim bash sim/jobs/train_pour.sh

    DATASET_ROOT=data/pour_sim RUN_NAME=pour_act_chunk100 CHUNK=100 STEPS=200000 \
        WANDB=true bash sim/jobs/train_pour.sh

### About `REPO_ID`

A LeRobot v3 dataset's `meta/info.json` does not record the repo id it was
created under, so nothing on disk can be checked against it. When
`--dataset.root` points at a local dataset, that root is what is read and the
repo id is only a label (used for the run's metadata and for anything pushed to
the Hub). `galaxeo/a1x_pour_sim` is the default `run_pour.py --lerobot` writes
with, so both payloads default to the same string; override `REPO_ID` if you
recorded the dataset under another name.

## render.sh

Sourced by the payloads that render or run many simulator processes
(`floor_state.sh`, `pour_full.sh`). On Linux it sets `MUJOCO_GL=egl`, test
renders one frame, and if that fails installs the GLVND front library
(`libegl1`, `libgl1`) and writes the NVIDIA EGL vendor file, which a CUDA
container image usually lacks even though it carries the driver's own
`libEGL_nvidia.so.0`. It also pins numpy's BLAS to one thread per process
(a container on a 88-core host shows all 88 to every worker, and 40 workers
times 88 threads is over the thread limit) and provides `cores_available`,
the cgroup CPU quota rather than `nproc`, which reports the host.

## floor_state.sh

The state-based floor check on one machine: generate a dataset of the true
scene state with `gen_dataset.py --state` (no images), train ACT on it, and
evaluate it closed loop on the same box, with a video of every evaluation
episode. If a policy that sees the bottle pose and the glass position cannot
pour, the action side is what is short, and no image run will fix it.

| input | |
| --- | --- |
| `RUN_NAME` | default `pour_state_<JOB_ID>`; the dataset lands in `sim/datasets/<RUN_NAME>` |
| `EPISODES` | successful episodes wanted, default 1000 |
| `SEED0` | first scene seed, default 10000 |
| `STEPS`, `BATCH`, `CHUNK`, `LR`, `SAVE_FREQ` | as in `train_pour.sh`; defaults 50000, 32, 50, 1e-4, 10000 |
| `VAE` | ACT's variational objective, default false here (true in `train_pour.sh`) |
| `EVAL_N`, `EVAL_SEEDS` | episodes per range and the first seed of each, default 20 and `2000 1000` |
| `GEN_WORKERS` | generator processes, default the box's cores minus 2 |

    RUN_NAME=pour_state_floor EPISODES=1000 STEPS=50000 bash sim/jobs/floor_state.sh

## pour_full.sh

The image run on one machine: generate a dataset with the laptop camera, the
top-down map and the wrist camera, train ACT on it through `train_pour.sh`
(so the recipe is the one the earlier runs used), evaluate the checkpoint,
and evaluate a control checkpoint on the same seeds. The generator runs with
planner variation (the grasp drawn among the feasible pitches and yaws of
the preferred place) and noise injection (a smooth drift on the executed
joint commands, the planner's clean command kept as the label), both from
`run_pour.py --vary --servo-noise`.

| input | |
| --- | --- |
| `RUN_NAME` | default `pour_full_<JOB_ID>` |
| `EPISODES`, `SEED0`, `OVER` | wanted successes, first seed, seeds per success; defaults 3000, 10000, 1.6 |
| `VARY`, `SERVO_NOISE`, `WRIST` | 1/0, degrees, hand; defaults 1, 1.5, left |
| `STEPS`, `BATCH`, `CHUNK`, `LR`, `WORKERS`, `INIT_FROM` | passed to `train_pour.sh` |
| `DATASET_REPO` | optional Hub dataset to train on instead of generating |
| `PUSH_DATASET`, `PUSH_MODEL` | optional private Hub repos the merged set and the final weights are pushed to (`HF_TOKEN`) |
| `CONTROL_CKPT` | optional Hub model id or repo-relative `pretrained_model` dir evaluated as the control |
| `EVAL_N`, `EVAL_SEEDS`, `GEN_WORKERS` | as in `floor_state.sh` |

Results: `sim/runs/<RUN_NAME>/eval` and `eval_control`, each with
`seeds_*.json`, `all.json` and `video/seed*_{ok,fail}.mp4` (laptop camera
with the wrist stream beside it). `planner/eval_summary.py` prints the
success counts and the failure-mode mix from those files.

## pour_dagger.sh

One DAgger round on one machine: roll a policy out on fresh scenes from seed
5000 with `gen_dagger.py`, let the planner take over from the states it
reaches, merge the recoveries with the base dataset, warm-start training from
the rolled-out weights, and evaluate the result and the rolled-out policy on
the same seeds (`eval` and `eval_control`).

| input | |
| --- | --- |
| `POLICY` | **required**, Hub model id or repo-relative `pretrained_model` dir to roll out and warm-start from |
| `BASE_DATASET` | **required**, Hub dataset id or repo-relative root the recoveries are merged with |
| `RUN_NAME` | default `pour_dagger_<JOB_ID>` |
| `SEEDS`, `SEED0`, `TAKEOVERS`, `REWIND`, `WRIST` | scenes to roll out and the collector's knobs; defaults 1500, 5000, 1, `1.0,2.5,5.0`, left |
| `STEPS`, `BATCH`, `WORKERS` | passed to `train_pour.sh`; default 30000 steps |
| `PUSH_DATASET`, `PUSH_MODEL`, `CONTROL_CKPT`, `EVAL_N`, `EVAL_SEEDS`, `GEN_WORKERS` | as in `pour_full.sh`; the control defaults to `POLICY` |

The next round takes this round's `PUSH_MODEL` as `POLICY` and its
`PUSH_DATASET` as `BASE_DATASET`.

## Evaluating a checkpoint

Training does not evaluate: the pour scene is not a gym environment. Closed-loop
evaluation is a separate script that runs on a machine with MuJoCo:

    sim/.venv-lerobot/bin/python sim/planner/eval_policy.py \
        --ckpt sim/runs/pour_act_0 --n 20 --gif pour_policy.gif
