"""Grasp student: the network rebuilt without torchvision, the heat channel, the checkpoint and `act`.

The recorded fixtures come from the training environment (tests/fixtures/grasp/README.md).
Tests that need the published weights take the checkpoint's path from GALAXEO_GRASP_STUDENT
and skip without it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
from grasp_recording import npz, state_keys

torch = pytest.importorskip("torch")

from galaxeo.grasp import contract as C  # noqa: E402
from galaxeo.grasp import policy as P  # noqa: E402
from galaxeo.grasp.net import StudentNet  # noqa: E402
from galaxeo.grasp.policy import (  # noqa: E402
    CheckpointError,
    GraspStudentPolicy,
    heat_channel,
    load_checkpoint,
    sha256_of,
)

CHECKPOINT_ENV = "GALAXEO_GRASP_STUDENT"
IMAGE = (C.IMG_H, C.IMG_W, 3)


def dtype_name(d: object) -> str:
    return str(d).removeprefix("torch.")


def flip_one_weight_byte(src: Path, dst: Path) -> None:
    """Copy `src` with one bit changed in the middle of its largest tensor, so the copy still loads."""
    shutil.copyfile(src, dst)
    with zipfile.ZipFile(dst) as z:
        info = max(z.infolist(), key=lambda i: i.file_size)
    assert info.compress_type == zipfile.ZIP_STORED
    with open(dst, "r+b") as f:
        f.seek(info.header_offset)
        head = f.read(30)
        name_len = int.from_bytes(head[26:28], "little")
        extra_len = int.from_bytes(head[28:30], "little")
        start = info.header_offset + 30 + name_len + extra_len
        at = start + info.file_size // 2
        f.seek(at)
        b = f.read(1)
        f.seek(at)
        f.write(bytes([b[0] ^ 0x01]))


def never_unpickle(*args: object, **kw: object) -> None:
    raise AssertionError("torch.load ran on an unverified file")


def student_schema(**change: object) -> dict:
    """A checkpoint dict with the student's schema and no weights: the schema check needs none."""
    ckpt = {"arch": "resnet18", "heat": True, "obs_dim": C.OBS_DIM, "act_dim": C.ACT_DIM, "net": {},
            "obs_mean": np.zeros(C.OBS_DIM, np.float32), "obs_std": np.ones(C.OBS_DIM, np.float32)}
    ckpt.update(change)
    return ckpt


def saved(ckpt: dict, path: Path) -> Path:
    torch.save(ckpt, path)
    return path


@pytest.fixture(scope="module")
def ckpt_path() -> Path:
    value = os.environ.get(CHECKPOINT_ENV, "").strip()
    if not value:
        pytest.skip(f"no checkpoint: {CHECKPOINT_ENV}=<path to the published student.pt>")
    path = Path(value).expanduser()
    if not path.is_file():
        pytest.fail(f"{CHECKPOINT_ENV}={value}: no such file")
    return path


@pytest.fixture(scope="module")
def policy(ckpt_path: Path) -> GraspStudentPolicy:
    return GraspStudentPolicy.load(ckpt_path, C.CHECKPOINT_SHA256)


@pytest.fixture(scope="module")
def changed_copy(ckpt_path: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    dst = tmp_path_factory.mktemp("changed") / "student.pt"
    flip_one_weight_byte(ckpt_path, dst)
    return dst


@pytest.fixture
def toy_policy() -> GraspStudentPolicy:
    torch.manual_seed(0)
    return GraspStudentPolicy(StudentNet(), np.zeros(C.OBS_DIM, np.float32),
                              np.full(C.OBS_DIM, 0.5, np.float32), "toy.pt")


@pytest.fixture
def junk(tmp_path: Path) -> Path:
    path = tmp_path / "junk.pt"
    path.write_bytes(b"not a checkpoint")
    return path


# ---------------------------------------------------------------- network, no checkpoint

def test_the_rebuilt_network_has_the_recorded_state_dict_names_shapes_and_dtypes() -> None:
    want = state_keys()
    sd = StudentNet().state_dict()
    assert len(sd) == 130
    assert [row[0] for row in want] == list(sd)
    assert [list(sd[name].shape) for name, _, _ in want] == [list(shape) for _, shape, _ in want]
    assert [dtype_name(sd[name].dtype) for name, _, _ in want] == [dtype_name(dtype) for _, _, dtype in want]


def test_the_network_uses_exact_gelu_default_batch_norm_and_imagenet_statistics() -> None:
    net = StudentNet()
    assert net.prop[1].approximate == "none"
    assert all(m.approximate == "none" for m in net.head if isinstance(m, torch.nn.GELU))
    bns = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bns
    assert all(m.eps == 1e-5 and m.momentum == 0.1 and m.affine for m in bns)
    assert net.mean.shape == (1, 3, 1, 1)
    assert net.std.shape == (1, 3, 1, 1)
    assert torch.equal(net.mean.flatten(), torch.tensor([0.485, 0.456, 0.406]))
    assert torch.equal(net.std.flatten(), torch.tensor([0.229, 0.224, 0.225]))


@pytest.mark.parametrize("slot", [2, 3, 8, 9])
def test_the_parameter_free_trunk_slots_stay_in_place(slot: int) -> None:
    assert not list(StudentNet().trunk[slot].parameters())


def test_the_network_maps_a_batch_to_five_actions_in_the_tanh_range() -> None:
    net = StudentNet().eval()
    torch.manual_seed(1)
    img = torch.rand(2, 4, C.IMG_H, C.IMG_W)
    obs = torch.randn(2, C.OBS_DIM)
    with torch.no_grad():
        a = net(img, obs)
    assert a.shape == (2, C.ACT_DIM)
    assert a.dtype == torch.float32
    assert bool(torch.all(a.abs() <= 1.0))


def test_the_heat_channel_equals_the_training_function() -> None:
    h = npz("heat.npz", "uv", "heat")
    got = heat_channel(h["uv"]).numpy()
    assert got.shape == h["heat"].shape
    assert got.dtype == np.float32
    assert float(np.abs(got - h["heat"]).max()) <= 1e-6


def test_only_a_nan_pixel_gives_an_empty_heat_channel_and_an_off_frame_one_keeps_its_tail() -> None:
    h = npz("heat.npz", "uv")
    got = heat_channel(h["uv"]).numpy()
    nan = np.isnan(h["uv"]).any(axis=1)
    off = ~nan & ~((h["uv"][:, 0] >= 0) & (h["uv"][:, 0] < C.IMG_W) & (h["uv"][:, 1] >= 0) & (h["uv"][:, 1] < C.IMG_H))
    assert nan.any()
    assert off.any()
    assert not got[nan].any()
    assert all(0.0 < float(blob.max()) < 1.0 for blob in got[off])


def test_the_heat_sigma_comes_from_the_width() -> None:
    blob = heat_channel(np.array([[80.0, 60.0]], np.float32))[0, 0].numpy()
    sigma = C.HEAT_SIGMA_FRAC * C.IMG_W
    assert blob[60, 80] == 1.0
    assert blob[60, 90] == pytest.approx(np.exp(-100.0 / (2 * sigma ** 2)), rel=1e-6)
    assert blob[70, 80] == pytest.approx(np.exp(-100.0 / (2 * sigma ** 2)), rel=1e-6)


# ---------------------------------------------------------------- act and inputs, no checkpoint

@pytest.mark.parametrize("bad", [np.zeros(IMAGE, np.float32), np.zeros((C.IMG_W, C.IMG_H, 3), np.uint8),
                                 np.zeros((C.IMG_H, C.IMG_W, 4), np.uint8), np.zeros((1, *IMAGE), np.uint8)],
                         ids=["float", "transposed", "rgba", "batched"])
def test_act_refuses_a_wrong_image(toy_policy: GraspStudentPolicy, bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="image"):
        toy_policy.act(bad, np.zeros(C.OBS_DIM, np.float32), np.array([40.0, 70.0], np.float32))


@pytest.mark.parametrize("bad", [np.zeros(18, np.float32), np.zeros((1, C.OBS_DIM), np.float32)],
                         ids=["short", "batched"])
def test_act_refuses_a_misshaped_observation(toy_policy: GraspStudentPolicy, bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="obs"):
        toy_policy.act(np.zeros(IMAGE, np.uint8), bad, np.array([40.0, 70.0], np.float32))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1e39])
def test_act_refuses_an_observation_that_is_not_finite_in_float32(toy_policy: GraspStudentPolicy,
                                                                  value: float) -> None:
    obs = np.zeros(C.OBS_DIM, np.float64)
    obs[4] = value
    with pytest.raises(ValueError, match="not finite"):
        toy_policy.act(np.zeros(IMAGE, np.uint8), obs, np.array([40.0, 70.0], np.float32))


def test_act_refuses_a_misshaped_pixel(toy_policy: GraspStudentPolicy) -> None:
    with pytest.raises(ValueError, match="uv"):
        toy_policy.act(np.zeros(IMAGE, np.uint8), np.zeros(C.OBS_DIM, np.float32), np.zeros(3, np.float32))


def test_a_nan_pixel_is_legal_and_gives_an_empty_heat_channel(toy_policy: GraspStudentPolicy) -> None:
    img = np.random.default_rng(0).integers(0, 256, IMAGE, dtype=np.uint8)
    a = toy_policy.act(img, np.zeros(C.OBS_DIM), np.array([np.nan, np.nan]))
    assert a.shape == (C.ACT_DIM,)
    assert a.dtype == np.float32
    assert np.all(np.isfinite(a))
    assert not toy_policy.inputs(img, np.array([np.nan, np.nan]))[3].any()


def test_inputs_are_the_scaled_image_and_the_heat_channel(toy_policy: GraspStudentPolicy) -> None:
    img = np.random.default_rng(1).integers(0, 256, IMAGE, dtype=np.uint8)
    img.setflags(write=False)
    uv = np.array([-12.0, 30.0])
    x = toy_policy.inputs(img, uv)
    assert x.shape == (4, C.IMG_H, C.IMG_W)
    assert x.dtype == np.float32
    np.testing.assert_array_equal(x[:3], img.transpose(2, 0, 1).astype(np.float32) / np.float32(255.0))
    np.testing.assert_array_equal(x[3], heat_channel(uv[None])[0, 0].numpy())
    assert x[3].max() > 0.0


def test_act_normalises_with_the_checkpoint_statistics_without_clipping(
        toy_policy: GraspStudentPolicy) -> None:
    img = np.random.default_rng(2).integers(0, 256, IMAGE, dtype=np.uint8)
    uv = np.array([44.6, 75.8], np.float32)
    a = toy_policy.act(img, np.full(C.OBS_DIM, 50.0), uv)
    x = torch.as_tensor(toy_policy.inputs(img, uv)[None])
    with torch.no_grad():
        want = toy_policy.net(x, torch.full((1, C.OBS_DIM), 100.0)).numpy()[0]
        clipped = toy_policy.net(x, torch.full((1, C.OBS_DIM), 5.0)).numpy()[0]
    np.testing.assert_allclose(a, want, rtol=0, atol=1e-6)
    assert np.abs(a - clipped).max() > 1e-3


def test_act_runs_without_gradients_and_changes_neither_the_network_nor_its_inputs(
        toy_policy: GraspStudentPolicy) -> None:
    grad = []
    toy_policy.net.register_forward_hook(lambda m, i, o: grad.append(torch.is_grad_enabled()))
    before = {k: v.clone() for k, v in toy_policy.net.state_dict().items()}
    img = np.random.default_rng(3).integers(0, 256, IMAGE, dtype=np.uint8)
    obs = np.random.default_rng(4).normal(size=C.OBS_DIM).astype(np.float32)
    uv = np.array([44.6, 75.8], np.float32)
    img_kept = img.copy()
    obs_kept = obs.copy()
    uv_kept = uv.copy()

    a1 = toy_policy.act(img, obs, uv)
    a2 = toy_policy.act(img, obs, uv)

    assert grad == [False, False]
    np.testing.assert_array_equal(a1, a2)
    # BatchNorm statistics stay: eval mode.
    assert [k for k, v in toy_policy.net.state_dict().items() if not torch.equal(v, before[k])] == []
    np.testing.assert_array_equal(img, img_kept)
    np.testing.assert_array_equal(obs, obs_kept)
    np.testing.assert_array_equal(uv, uv_kept)


def test_a_non_finite_network_output_is_returned_as_it_is(toy_policy: GraspStudentPolicy) -> None:
    with torch.no_grad():
        toy_policy.net.head[4].bias[:] = float("nan")
    a = toy_policy.act(np.zeros(IMAGE, np.uint8), np.zeros(C.OBS_DIM), np.zeros(2))
    assert a.shape == (C.ACT_DIM,)
    assert np.isnan(a).all()


# ---------------------------------------------------------------- refusals, no checkpoint

def test_a_missing_checkpoint_names_the_published_one(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="no such file.*huggingface"):
        GraspStudentPolicy.load(tmp_path / "nothing.pt", C.CHECKPOINT_SHA256)


@pytest.mark.parametrize("path", ["", "  "])
def test_an_empty_checkpoint_path_is_refused(path: str) -> None:
    with pytest.raises(CheckpointError, match="no checkpoint path"):
        GraspStudentPolicy.load(path, C.CHECKPOINT_SHA256)


@pytest.mark.parametrize("sha256", ["05e5aaf6", "g" * 64, ""])
def test_a_malformed_sha256_is_refused(junk: Path, sha256: str) -> None:
    with pytest.raises(CheckpointError, match="is not a sha256"):
        GraspStudentPolicy.load(junk, sha256)


def test_a_file_with_another_hash_is_refused_before_anything_is_unpickled(
        junk: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch, "load", never_unpickle)
    with pytest.raises(CheckpointError, match=f"expected {'0' * 64}"):
        load_checkpoint(junk, "0" * 64)


def test_a_file_of_another_size_than_the_published_one_is_refused_before_hashing(
        junk: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch, "load", never_unpickle)
    monkeypatch.setattr(P, "sha256_of", never_unpickle)
    with pytest.raises(CheckpointError, match=f"{C.CHECKPOINT_BYTES}"):
        load_checkpoint(junk, C.CHECKPOINT_SHA256)


def test_a_file_that_is_no_checkpoint_is_refused_by_the_weights_only_load(junk: Path) -> None:
    with pytest.raises(CheckpointError, match="weights-only load failed"):
        load_checkpoint(junk, hashlib.sha256(b"not a checkpoint").hexdigest())


def test_a_checkpoint_with_the_student_schema_passes_the_schema_check(tmp_path: Path) -> None:
    path = saved(student_schema(), tmp_path / "good.pt")
    assert load_checkpoint(path, sha256_of(path))["arch"] == "resnet18"


@pytest.mark.parametrize("key,value", [
    ("arch", "cnn"), ("heat", False), ("heat", 1), ("obs_dim", 18), ("act_dim", 4), ("net", None),
    ("obs_mean", np.zeros(18, np.float32)), ("obs_mean", np.full(19, np.nan, np.float32)),
    ("obs_std", np.zeros(19, np.float32)),
])
def test_a_checkpoint_with_another_schema_is_refused_naming_the_key(tmp_path: Path, key: str, value: object) -> None:
    path = saved(student_schema(**{key: value}), tmp_path / "bad.pt")
    with pytest.raises(CheckpointError, match=key):
        load_checkpoint(path, sha256_of(path))


def test_a_checkpoint_without_a_key_is_refused_naming_it(tmp_path: Path) -> None:
    ckpt = student_schema()
    del ckpt["obs_std"]
    path = saved(ckpt, tmp_path / "missing.pt")
    with pytest.raises(CheckpointError, match="obs_std missing"):
        load_checkpoint(path, sha256_of(path))


# ---------------------------------------------------------------- the published checkpoint

def test_the_checkpoint_has_the_published_sha256_and_size(ckpt_path: Path) -> None:
    assert ckpt_path.stat().st_size == C.CHECKPOINT_BYTES
    assert sha256_of(ckpt_path) == C.CHECKPOINT_SHA256


def test_the_checkpoint_loads_weights_only_on_the_cpu(ckpt_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    real = torch.load

    def spy(*args: object, **kw: object) -> object:
        calls.append(kw)
        return real(*args, **kw)

    monkeypatch.setattr(torch, "load", spy)
    load_checkpoint(ckpt_path, C.CHECKPOINT_SHA256)
    assert calls
    assert all(kw.get("weights_only") is True for kw in calls)
    assert all(kw.get("map_location") == "cpu" for kw in calls)


def test_the_checkpoint_has_the_student_schema(ckpt_path: Path) -> None:
    ckpt = load_checkpoint(ckpt_path, C.CHECKPOINT_SHA256)
    assert ckpt["arch"] == "resnet18"
    assert ckpt["heat"] is True
    assert ckpt["obs_dim"] == C.OBS_DIM
    assert ckpt["act_dim"] == C.ACT_DIM
    assert len(ckpt["net"]) == 130
    assert ckpt["obs_mean"].shape == (C.OBS_DIM,)
    assert ckpt["obs_mean"].dtype == np.float32
    assert ckpt["obs_std"].shape == (C.OBS_DIM,)
    assert ckpt["obs_std"].dtype == np.float32
    assert np.all(ckpt["obs_std"] > 0)


def test_the_checkpoint_weights_load_strictly_into_the_rebuilt_network(ckpt_path: Path) -> None:
    res = StudentNet().load_state_dict(load_checkpoint(ckpt_path, C.CHECKPOINT_SHA256)["net"], strict=True)
    assert not res.missing_keys
    assert not res.unexpected_keys


def test_load_gives_an_eval_network_on_the_cpu_with_the_asked_threads_and_the_file_statistics(
        ckpt_path: Path) -> None:
    threads = torch.get_num_threads()
    try:
        pol = GraspStudentPolicy.load(ckpt_path, C.CHECKPOINT_SHA256, threads=1)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(threads)
    ckpt = load_checkpoint(ckpt_path, C.CHECKPOINT_SHA256)

    assert pol.path == ckpt_path
    assert pol.sha256 == C.CHECKPOINT_SHA256
    assert not any(m.training for m in pol.net.modules())
    assert all(t.device.type == "cpu" for t in pol.net.state_dict().values())
    assert pol.obs_mean.dtype == np.float32
    assert pol.obs_std.dtype == np.float32
    np.testing.assert_array_equal(pol.obs_mean, ckpt["obs_mean"])
    np.testing.assert_array_equal(pol.obs_std, ckpt["obs_std"])
    assert [name for name, value in ckpt["net"].items() if not torch.equal(pol.net.state_dict()[name], value)] == []


def test_a_changed_copy_is_refused_against_the_published_hash_before_anything_is_unpickled(
        changed_copy: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch, "load", never_unpickle)
    with pytest.raises(CheckpointError, match=f"expected {C.CHECKPOINT_SHA256}"):
        GraspStudentPolicy.load(changed_copy, C.CHECKPOINT_SHA256)


def test_a_changed_copy_loads_against_its_own_hash_with_a_warning(
        changed_copy: Path, caplog: pytest.LogCaptureFixture) -> None:
    digest = sha256_of(changed_copy)
    assert digest != C.CHECKPOINT_SHA256
    with caplog.at_level(logging.WARNING, logger=P.__name__):
        pol = GraspStudentPolicy.load(changed_copy, f" {digest.upper()} ")
    assert pol.sha256 == digest
    assert pol.path == changed_copy
    assert any(digest in r.getMessage() and C.CHECKPOINT_SHA256 in r.getMessage() for r in caplog.records)


def test_the_published_checkpoint_is_refused_against_another_hash(ckpt_path: Path, changed_copy: Path) -> None:
    digest = sha256_of(changed_copy)
    with pytest.raises(CheckpointError, match=f"expected {digest}"):
        GraspStudentPolicy.load(ckpt_path, digest)


def test_the_policy_gives_the_recorded_actions_at_batch_1_from_the_recorded_inputs(
        policy: GraspStudentPolicy) -> None:
    s = npz("shard.npz", "img", "obs", "uv", "act")
    assert len(s["act"]) == 59
    assert np.isfinite(s["uv"]).all(axis=1).any()
    got = np.stack([policy.act(s["img"][i], s["obs"][i], s["uv"][i]) for i in range(len(s["act"]))])
    assert got.dtype == np.float32
    assert got.shape == s["act"].shape
    err = float(np.abs(got - s["act"]).max())
    print(f"max |act - recorded| over {len(got)} ticks at batch 1: {err:.3g}")
    assert err <= 1e-5


def test_the_policy_is_timed_on_zeros_input(policy: GraspStudentPolicy) -> None:
    r = policy.time(200)
    print(f"student on the CPU ({torch.get_num_threads()} threads): median {r.median_ms:.1f} ms, "
          f"p95 {r.p95_ms:.1f} ms, first {r.first_ms:.1f} ms")
    assert np.isfinite([r.first_ms, r.median_ms, r.p95_ms]).all()
    # The budget is 30 / 45 ms; the factor 2 absorbs other test runs on a shared machine.
    assert r.median_ms < 2 * P.TIME_MEDIAN_MAX_MS
    assert r.p95_ms < 2 * P.TIME_P95_MAX_MS
