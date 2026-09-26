#!/usr/bin/env bash
# First-boot setup of a bare Ubuntu GPU VM, run once as root before any
# payload: the OS packages the payloads need and nothing else. Python
# dependencies are NOT installed here; the job venv is built from
# sim/uv.lock by whoever runs the payload. Safe to run twice.
#
# rsync:            the checkout arrives by rsync, which minimal images lack.
# libegl1, libgl1,  MuJoCo renders through EGL on a headless box (the
# libglvnd0:        dataset payloads); the NVIDIA driver supplies the vendor
#                   library, the GLVND front-ends come from apt.
# ffmpeg:           lerobot's video encoder shells out to it when recording.
set -euo pipefail

SUDO=""
[ "$(id -u)" -eq 0 ] || SUDO="sudo -n"

export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq --no-install-recommends \
    rsync curl ca-certificates \
    libegl1 libgl1 libglvnd0 libglib2.0-0 \
    ffmpeg

# The NVIDIA EGL vendor file is missing on some images even with the driver
# present; without it EGL finds no device and MuJoCo renders black.
if [ ! -e /usr/share/glvnd/egl_vendor.d/10_nvidia.json ] && [ -e /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 ]; then
    $SUDO mkdir -p /usr/share/glvnd/egl_vendor.d
    printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}\n' \
        | $SUDO tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json >/dev/null
fi

# The lock pins torch with CUDA 13 wheels, which need driver 580 or newer.
# An older datacenter driver (570 on the first brev H100) works through
# NVIDIA's forward-compatibility package; payloads put its directory on
# LD_LIBRARY_PATH when it exists.
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
if [ -n "$DRIVER" ] && [ "$DRIVER" -lt 580 ] && [ ! -d /usr/local/cuda-13.0/compat ]; then
    . /etc/os-release
    REPO="ubuntu$(echo "$VERSION_ID" | tr -d .)"
    ( cd /tmp && curl -sSLO "https://developer.download.nvidia.com/compute/cuda/repos/$REPO/x86_64/cuda-keyring_1.1-1_all.deb" \
        && $SUDO dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null && $SUDO apt-get update -qq \
        && $SUDO apt-get install -y -qq --no-install-recommends cuda-compat-13-0 ) \
        && echo "cuda-compat-13-0 installed for driver $DRIVER" \
        || echo "cuda-compat-13-0 did not install; torch will not see the GPU"
fi

nvidia-smi -L || echo "no nvidia-smi yet; the driver may still be installing"
echo "node boot done"
