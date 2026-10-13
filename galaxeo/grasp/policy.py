"""The grasp student's checkpoint, and the policy that runs it on the CPU.

    pol = GraspStudentPolicy.load(path, sha256)   # sha256 first, then a weights-only load, strict weights
    a = pol.act(image, obs, uv)                   # uint8 (120, 160, 3), raw 19-vector, believed pixel -> (5,)

The caller gives the checkpoint's path and the sha256 it must have; nothing here downloads.
Every load checks the sha256 before anything is unpickled, and unpickles with
`weights_only=True`.

`act` is training's `Student._prep` + `Student.act` at batch 1: the image scaled to [0, 1],
the heat channel at the believed pixel appended, the 19-vector normalised with the
checkpoint's statistics in float32 without clipping. Nothing outside `act` normalises.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from .contract import (
    ACT_DIM,
    CHECKPOINT_BYTES,
    CHECKPOINT_SHA256,
    CHECKPOINT_URL,
    HEAT_SIGMA_FRAC,
    IMG_H,
    IMG_W,
    OBS_DIM,
)
from .net import StudentNet

log = logging.getLogger(__name__)

#: The inference budget at 20 Hz on the CPU, for `GraspStudentPolicy.time`.
TIME_MEDIAN_MAX_MS = 30.0
TIME_P95_MAX_MS = 45.0

_CHUNK = 1 << 20
_HEX64 = re.compile(r"[0-9a-f]{64}")


class CheckpointError(RuntimeError):
    """The checkpoint is missing, has another hash, or does not have the student's schema."""


@dataclass(frozen=True)
class InferenceTime:
    """Wall time of `act` [ms]: the first call, and the median and p95 of the calls after it."""

    first_ms: float
    median_ms: float
    p95_ms: float


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _hex(value: str, what: str) -> str:
    v = str(value).strip().lower()
    if not _HEX64.fullmatch(v):
        raise CheckpointError(f"{what}: {value!r} is not a sha256 (64 hex characters)")
    return v


def _safe_globals() -> list[Any]:
    import numpy

    return [numpy._core.multiarray._reconstruct, numpy.ndarray, numpy.dtype, type(numpy.dtype("float32"))]


def _check_schema(ckpt: Any, path: Path) -> None:
    if not isinstance(ckpt, Mapping):
        raise CheckpointError(f"{path}: not a checkpoint dict ({type(ckpt).__name__})")

    def need(key: str) -> Any:
        if key not in ckpt:
            raise CheckpointError(f"{path}: key {key} missing")
        return ckpt[key]

    if need("arch") != "resnet18":
        raise CheckpointError(f"{path}: arch {ckpt['arch']!r}, expected 'resnet18'")
    if need("heat") is not True:
        raise CheckpointError(f"{path}: heat {ckpt['heat']!r}, expected True")
    for key, want in (("obs_dim", OBS_DIM), ("act_dim", ACT_DIM)):
        if need(key) != want:
            raise CheckpointError(f"{path}: {key} {ckpt[key]!r}, expected {want}")
    if not isinstance(need("net"), Mapping):
        raise CheckpointError(f"{path}: net is {type(ckpt['net']).__name__}, not a state dict")

    for key in ("obs_mean", "obs_std"):
        try:
            arr = np.asarray(need(key), np.float64)
        except (TypeError, ValueError) as exc:
            raise CheckpointError(f"{path}: {key} is not numeric: {exc}") from exc
        if arr.shape != (OBS_DIM,):
            raise CheckpointError(f"{path}: {key} has shape {arr.shape}, expected ({OBS_DIM},)")
        if not np.all(np.isfinite(arr)):
            raise CheckpointError(f"{path}: {key} is not finite")
        if key == "obs_std" and not np.all(arr > 0):
            raise CheckpointError(f"{path}: obs_std has entries <= 0")


def load_checkpoint(path: str | Path, sha256: str) -> dict:
    """Verify the sha256, load weights-only, check the schema."""
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"{path}: no such file (the published checkpoint: {CHECKPOINT_URL})")

    want = _hex(sha256, "sha256")
    size = path.stat().st_size
    if want == CHECKPOINT_SHA256 and size != CHECKPOINT_BYTES:
        raise CheckpointError(f"{path}: {size} bytes, the published checkpoint has {CHECKPOINT_BYTES} "
                              f"(incomplete copy or another file)")
    got = sha256_of(path)
    if got != want:
        raise CheckpointError(f"{path}: sha256 {got}, expected {want} (another checkpoint needs its own sha256)")

    try:
        with torch.serialization.safe_globals(_safe_globals()):
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"{path}: weights-only load failed: {exc}") from exc
    _check_schema(ckpt, path)
    return ckpt


def heat_channel(uv: torch.Tensor, H: int = IMG_H, W: int = IMG_W) -> torch.Tensor:
    """(B, 1, H, W) Gaussian blob at each (u, v), training's function.

    Sigma comes from the width. Only NaN gives a zero channel: an off-frame point keeps
    the tail of its blob inside the frame. Not clipped, not normalised, peak 1.0.
    """
    uv = torch.as_tensor(np.array(uv, np.float32))
    ys = torch.arange(H, dtype=torch.float32).view(1, H, 1)
    xs = torch.arange(W, dtype=torch.float32).view(1, 1, W)
    u = uv[:, 0].view(-1, 1, 1)
    v = uv[:, 1].view(-1, 1, 1)
    s2 = 2 * (HEAT_SIGMA_FRAC * W) ** 2
    blob = torch.exp(-((xs - u) ** 2 + (ys - v) ** 2) / s2)
    return torch.nan_to_num(blob, nan=0.0).unsqueeze(1)


class GraspStudentPolicy:
    """The student network with its observation statistics, at batch 1 on the CPU."""

    obs_mean: np.ndarray
    obs_std: np.ndarray
    path: Path

    def __init__(self, net: StudentNet, obs_mean: np.ndarray, obs_std: np.ndarray, path: str | Path, *,
                 sha256: str | None = None, schema: Mapping[str, Any] | None = None) -> None:
        self.net = net.eval()
        self.obs_mean = np.array(obs_mean, np.float32)
        self.obs_std = np.array(obs_std, np.float32)
        if self.obs_mean.shape != (OBS_DIM,) or self.obs_std.shape != (OBS_DIM,):
            raise ValueError(f"obs_mean / obs_std must be ({OBS_DIM},)")
        self.path = Path(path)
        self.sha256 = sha256
        self.schema = dict(schema or {})

    @classmethod
    def load(cls, path: str | Path, sha256: str, *, threads: int = 2) -> GraspStudentPolicy:
        """Load the checkpoint at `path`, verified against `sha256` before anything is unpickled."""
        if not str(path).strip():
            raise CheckpointError(f"no checkpoint path given (the published checkpoint: {CHECKPOINT_URL})")
        found = Path(path).expanduser()
        want = _hex(sha256, "sha256")
        if want != CHECKPOINT_SHA256:
            log.warning("checkpoint %s is verified against sha256 %s, not the published %s",
                        found, want, CHECKPOINT_SHA256)

        ckpt = load_checkpoint(found, want)
        net = StudentNet(OBS_DIM, ACT_DIM)
        try:
            net.load_state_dict(ckpt["net"], strict=True)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise CheckpointError(f"{found}: net does not fit the student network: {exc}") from exc

        torch.set_num_threads(threads)
        schema = {k: v for k, v in ckpt.items() if k not in ("net", "obs_mean", "obs_std")}
        return cls(net, ckpt["obs_mean"], ckpt["obs_std"], found, sha256=want, schema=schema)

    @staticmethod
    def _image(image: np.ndarray) -> np.ndarray:
        img = np.asarray(image)
        if img.dtype != np.uint8 or img.shape != (IMG_H, IMG_W, 3):
            raise ValueError(f"image must be uint8 ({IMG_H}, {IMG_W}, 3), got {img.dtype} {img.shape}")
        # A private writable copy: torch warns on read-only buffers, and a camera may reuse its own.
        return np.array(img, order="C", copy=True)

    @staticmethod
    def _uv(uv: np.ndarray) -> np.ndarray:
        u = np.asarray(uv)
        if u.shape != (2,) or u.dtype.kind not in "fiu":
            raise ValueError(f"uv must be a numeric (2,), got {u.dtype} {u.shape}")
        return u.astype(np.float32)

    @staticmethod
    def _stack(image: np.ndarray, uv: np.ndarray) -> torch.Tensor:
        # Same tensor operations as training's `_prep`, so the convolutions see the same memory layout.
        x = torch.as_tensor(image[None]).permute(0, 3, 1, 2).float() / 255.0
        return torch.cat([x, heat_channel(uv[None], x.shape[2], x.shape[3])], 1)

    def inputs(self, image: np.ndarray, uv: np.ndarray) -> np.ndarray:
        """The (4, 120, 160) float32 image input `act` gives the network."""
        with torch.no_grad():
            return np.ascontiguousarray(self._stack(self._image(image), self._uv(uv))[0].numpy())

    def act(self, image: np.ndarray, obs: np.ndarray, uv: np.ndarray) -> np.ndarray:
        """(5,) float32 action for the raw 19-vector; a non-finite output is returned as it is."""
        img = self._image(image)
        o = np.asarray(obs)
        if o.shape != (OBS_DIM,) or o.dtype.kind not in "fiu":
            raise ValueError(f"obs must be a numeric ({OBS_DIM},), got {o.dtype} {o.shape}")
        with np.errstate(over="ignore"):             # an overflow becomes inf and is refused just below
            o = o.astype(np.float32)
        if not np.all(np.isfinite(o)):
            raise ValueError(f"obs is not finite at {np.flatnonzero(~np.isfinite(o)).tolist()}")
        u = self._uv(uv)

        with torch.no_grad():
            x = self._stack(img, u)
            on = torch.as_tensor(((o - self.obs_mean) / self.obs_std)[None])
            a = self.net(x, on)
        return np.array(a[0].numpy(), np.float32)

    def time(self, n: int = 200) -> InferenceTime:
        """Wall time of `act` on zeros: the first call, then median and p95 of the next `n`."""
        image = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        obs = np.zeros(OBS_DIM, np.float32)
        uv = np.zeros(2, np.float32)
        ms = []
        for _ in range(max(1, int(n)) + 1):
            t0 = perf_counter()
            self.act(image, obs, uv)
            ms.append((perf_counter() - t0) * 1e3)
        rest = np.asarray(ms[1:])
        return InferenceTime(first_ms=float(ms[0]), median_ms=float(np.median(rest)),
                             p95_ms=float(np.percentile(rest, 95)))
