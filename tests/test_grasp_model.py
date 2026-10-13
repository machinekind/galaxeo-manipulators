"""grasp_model.py without the network: a fake opener serves the bytes, a synthetic checkpoint has the
student's schema, a timed double stands in for the policy. The published weights are used only when
GALAXEO_GRASP_STUDENT names them."""

from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

import grasp_model as gm
from galaxeo.grasp.contract import ACT_DIM, CHECKPOINT_SHA256, OBS_DIM

CHECKPOINT_ENV = "GALAXEO_GRASP_STUDENT"
URL = "https://example.invalid/student.pt"
BLOB = b"weights" * 1000


@dataclass
class Response:
    """What urlopen returns: a byte stream that knows where it came from."""

    body: bytes
    url: str
    fail_after: int | None = None
    _stream: io.BytesIO = field(init=False)

    def __post_init__(self) -> None:
        self._stream = io.BytesIO(self.body)

    def read(self, n: int = -1) -> bytes:
        if self.fail_after is not None and self._stream.tell() >= self.fail_after:
            raise OSError("connection reset")
        return self._stream.read(min(n, 4096))

    def geturl(self) -> str:
        return self.url

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


@dataclass
class Opener:
    """urllib.request.urlopen as a double: serves `response` and counts the calls."""

    response: Response
    calls: list[str] = field(default_factory=list)

    def __call__(self, url: str, timeout: float) -> Response:
        self.calls.append(url)
        return self.response


def published(body: bytes = BLOB, url: str = URL) -> gm.Published:
    return gm.Published(url=url, sha256=hashlib.sha256(body).hexdigest(), size=len(body))


def fetch(dest: Path, opener: Opener) -> int:
    return gm.main(["fetch", str(dest)], gm.Deps(opener=opener))


@pytest.fixture
def blob_published(monkeypatch):
    monkeypatch.setattr(gm, "PUBLISHED", published())


# ====================================================================== fetch
def test_fetch_saves_the_file_when_size_and_hash_match(tmp_path, blob_published):
    dest = tmp_path / "models" / "student.pt"

    code = fetch(dest, Opener(Response(BLOB, URL)))

    assert code == 0
    assert dest.read_bytes() == BLOB


def test_fetch_leaves_no_part_file(tmp_path, blob_published):
    fetch(tmp_path / "student.pt", Opener(Response(BLOB, URL)))

    assert list(tmp_path.glob("*.part")) == []


def test_another_file_is_never_put_in_place(tmp_path, blob_published):
    dest = tmp_path / "student.pt"

    code = fetch(dest, Opener(Response(BLOB + b"!", URL)))

    assert code == 1
    assert list(tmp_path.iterdir()) == []


def test_a_cut_download_leaves_nothing(tmp_path, blob_published):
    code = fetch(tmp_path / "student.pt", Opener(Response(BLOB, URL, fail_after=4096)))

    assert code == 1
    assert list(tmp_path.iterdir()) == []


def test_a_redirect_to_plain_http_is_refused(tmp_path, blob_published):
    code = fetch(tmp_path / "student.pt", Opener(Response(BLOB, "http://example.invalid/student.pt")))

    assert code == 1
    assert list(tmp_path.iterdir()) == []


def test_a_plain_http_url_is_never_opened(tmp_path, monkeypatch):
    monkeypatch.setattr(gm, "PUBLISHED", published(url="http://example.invalid/student.pt"))
    opener = Opener(Response(BLOB, URL))

    code = fetch(tmp_path / "student.pt", opener)

    assert code == 1
    assert opener.calls == []


def test_a_file_already_in_place_is_not_downloaded_again(tmp_path, blob_published):
    dest = tmp_path / "student.pt"
    dest.write_bytes(BLOB)
    opener = Opener(Response(BLOB, URL))

    code = fetch(dest, opener)

    assert code == 0
    assert opener.calls == []


def test_another_file_at_the_destination_is_not_overwritten(tmp_path, blob_published):
    dest = tmp_path / "student.pt"
    dest.write_bytes(b"mine")

    code = fetch(dest, Opener(Response(BLOB, URL)))

    assert code == 1
    assert dest.read_bytes() == b"mine"


def test_fetch_has_no_default_destination():
    with pytest.raises(SystemExit) as exc:
        gm.main(["fetch"])

    assert exc.value.code == 2


def test_the_published_checkpoint_is_the_contracts():
    assert gm.PUBLISHED.sha256 == CHECKPOINT_SHA256


# ====================================================================== check and time (torch)
@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def state_entries() -> int:
    from galaxeo.grasp.net import StudentNet

    return len(StudentNet().state_dict())


@pytest.fixture(scope="module")
def synthetic(torch, tmp_path_factory) -> Path:
    """A checkpoint with the student's schema and random weights."""
    from galaxeo.grasp.net import StudentNet

    path = tmp_path_factory.mktemp("ckpt") / "student.pt"
    torch.manual_seed(0)
    ckpt = {"arch": "resnet18", "heat": True, "obs_dim": OBS_DIM, "act_dim": ACT_DIM,
            "net": StudentNet(OBS_DIM, ACT_DIM).state_dict(),
            "obs_mean": np.zeros(OBS_DIM, np.float32), "obs_std": np.ones(OBS_DIM, np.float32)}
    torch.save(ckpt, path)
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_check_loads_a_checkpoint_with_its_hash(synthetic, capsys):
    code = gm.main(["check", str(synthetic), "--sha256", sha(synthetic)])

    assert code == 0
    assert f"{state_entries()} wpisow wag" in capsys.readouterr().out


def test_check_names_a_file_that_is_not_the_published_one(synthetic, capsys):
    gm.main(["check", str(synthetic), "--sha256", sha(synthetic)])

    assert "INNY niz opublikowany" in capsys.readouterr().out


def test_check_refuses_a_checkpoint_with_another_hash(synthetic):
    code = gm.main(["check", str(synthetic)])

    assert code == 1


def test_check_of_a_missing_file_fails(torch, tmp_path):
    code = gm.main(["check", str(tmp_path / "student.pt")])

    assert code == 1


@dataclass(frozen=True)
class Timing:
    first_ms: float
    median_ms: float
    p95_ms: float


@dataclass
class TimedPolicy:
    timing: Timing

    def time(self, n: int = 200) -> Timing:
        return self.timing


def time_with(timing: Timing, path: Path) -> int:
    return gm.main(["time", str(path), "--n", "3"], gm.Deps(load_policy=lambda p, s: TimedPolicy(timing)))


def test_time_within_the_budget_passes(synthetic):
    assert time_with(Timing(first_ms=50.0, median_ms=12.0, p95_ms=20.0), synthetic) == 0


def test_time_over_the_median_budget_fails(synthetic):
    assert time_with(Timing(first_ms=50.0, median_ms=31.0, p95_ms=40.0), synthetic) == 1


def test_time_over_the_p95_budget_fails(synthetic):
    assert time_with(Timing(first_ms=50.0, median_ms=12.0, p95_ms=46.0), synthetic) == 1


@pytest.fixture(scope="module")
def published_path(torch) -> Path:
    value = os.environ.get(CHECKPOINT_ENV, "").strip()
    if not value:
        pytest.skip(f"no checkpoint: {CHECKPOINT_ENV}=<path to the published student.pt>")
    return Path(value).expanduser()


def test_the_published_checkpoint_checks(published_path):
    assert gm.main(["check", str(published_path)]) == 0
