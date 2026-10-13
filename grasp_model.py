#!/usr/bin/env python3
"""The grasp student's checkpoint: download it, check it, time it.

    python3 grasp_model.py fetch ~/models/student.pt     # HTTPS; size and sha256 checked before the file appears
    python3 grasp_model.py check ~/models/student.pt     # size, sha256, a weights-only strict load
    python3 grasp_model.py time ~/models/student.pt      # inference on this CPU: median < 30 ms, p95 < 45 ms

The published checkpoint is `contract.CHECKPOINT_URL`, sha256 `contract.CHECKPOINT_SHA256` (the same
file as sim/grasp/student_v4.pt on the branch grasp-rl). `fetch` writes only where it is told: the
download goes to `<file>.part` next to the destination and is renamed into place only when its size
and sha256 match, so a cut download or another file never sits where the strategy's `student_pt`
points. `check` and `time` load through `galaxeo.grasp.policy`: sha256 before anything is unpickled,
weights only. Another checkpoint needs its own `--sha256`.

`fetch` needs only the standard library; `check` and `time` need torch (pip install "galaxeo[grasp]").
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import os
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from galaxeo.grasp.contract import CHECKPOINT_BYTES, CHECKPOINT_SHA256, CHECKPOINT_URL

CHUNK = 1 << 20
TIMEOUT_S = 60.0


@dataclass(frozen=True)
class Published:
    """Where the checkpoint is published and what the downloaded file must be."""

    url: str
    sha256: str
    size: int


PUBLISHED = Published(url=CHECKPOINT_URL, sha256=CHECKPOINT_SHA256, size=CHECKPOINT_BYTES)


class FetchError(RuntimeError):
    """The download failed or delivered another file; the message is for the operator."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _https(url: str) -> bool:
    return str(url).lower().startswith("https://")


@dataclass(frozen=True)
class Streamed:
    size: int
    sha256: str


def _stream(published: Published, part: Path, opener: Callable[..., Any], timeout_s: float) -> Streamed:
    """The response body into `part`, stopped as soon as it outgrows the published size."""
    digest = hashlib.sha256()
    size = 0
    with opener(published.url, timeout=timeout_s) as response:
        final = response.geturl()
        # A redirect to plain HTTP would deliver bytes nobody vouches for until the hash.
        if not _https(final):
            raise FetchError(f"{published.url} przekierowuje na {final}, nie https")
        with open(part, "wb") as f:
            while chunk := response.read(CHUNK):
                size += len(chunk)
                # A server that keeps streaming would otherwise fill the disk before the hash is checked.
                if size > published.size:
                    raise FetchError(f"{published.url} wysyla wiecej niz {published.size} B - przerwano")
                digest.update(chunk)
                f.write(chunk)
    return Streamed(size=size, sha256=digest.hexdigest())


def download(published: Published, dest: Path, *, opener: Callable[..., Any] = urllib.request.urlopen,
             timeout_s: float = TIMEOUT_S) -> None:
    """Streams `published.url` into `<dest>.part`, then renames it to `dest` if size and sha256 match;
    whatever fails, the `.part` file is gone."""
    if not _https(published.url):
        raise FetchError(f"pobieram tylko przez https://, nie {published.url}")

    part = dest.with_name(dest.name + ".part")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        got = _stream(published, part, opener, timeout_s)
        if got.size != published.size or got.sha256 != published.sha256:
            raise FetchError(f"pobrano {got.size} B o sha256 {got.sha256}, oczekiwano {published.size} B o sha256 "
                             f"{published.sha256}; nic nie zapisano")
        os.replace(part, dest)
    except BaseException as exc:
        part.unlink(missing_ok=True)
        if isinstance(exc, (OSError, http.client.HTTPException)):
            raise FetchError(f"pobieranie {published.url} do {dest} nie powiodlo sie: {exc}") from exc
        raise


def load_policy(path: Path, sha256: str) -> Any:
    from galaxeo.grasp.policy import GraspStudentPolicy

    return GraspStudentPolicy.load(path, sha256)


@dataclass(frozen=True)
class Deps:
    """What the commands reach outside this file; tests replace both."""

    opener: Callable[..., Any] = urllib.request.urlopen
    load_policy: Callable[[Path, str], Any] = load_policy


def fetch(a: argparse.Namespace, deps: Deps) -> int:
    dest = Path(a.path).expanduser()
    # A directory or a symlink there would take the download and fail only at the rename.
    if dest.is_symlink() or (dest.exists() and not dest.is_file()):
        print(f"{dest} nie jest zwyklym plikiem (katalog albo dowiazanie) - nie pobieram.", file=sys.stderr)
        return 1
    if dest.is_file():
        got = file_sha256(dest)
        if got == PUBLISHED.sha256:
            print(f"Model juz jest: {dest} (sha256 zgodny) - nie pobieram drugi raz.")
            return 0
        print(f"{dest} juz istnieje i ma inny sha256 ({got}) - nie nadpisuje. Usun go albo podaj inny plik.",
              file=sys.stderr)
        return 1

    print(f"Pobieram {PUBLISHED.url}\n  -> {dest} ({PUBLISHED.size / 1e6:.1f} MB)", flush=True)
    try:
        download(PUBLISHED, dest, opener=deps.opener)
    except FetchError as exc:
        print(f"Nie pobrano modelu: {exc}", file=sys.stderr)
        return 1
    print(f"Zapisano {dest}, sha256 zgodny z opublikowanym ({PUBLISHED.sha256}).")
    return 0


def _loaded(a: argparse.Namespace, deps: Deps) -> Any | None:
    """The policy at `a.path`, or None after saying why there is none."""
    from galaxeo.grasp.policy import CheckpointError

    path = Path(a.path).expanduser()
    if not path.is_file():
        print(f"Nie ma pliku {path}. Pobierz model: python3 grasp_model.py fetch {path}", file=sys.stderr)
        return None
    try:
        return deps.load_policy(path, a.sha256)
    except CheckpointError as exc:
        print(f"Model odrzucony: {exc}", file=sys.stderr)
        return None


def check(a: argparse.Namespace, deps: Deps) -> int:
    path = Path(a.path).expanduser()
    if path.is_file():
        digest = file_sha256(path)
        origin = "opublikowany" if digest == PUBLISHED.sha256 else "INNY niz opublikowany"
        print(f"Model: {path.resolve()}")
        print(f"  rozmiar {path.stat().st_size} B (opublikowany ma {PUBLISHED.size} B), sha256 {digest} ({origin})")

    policy = _loaded(a, deps)
    if policy is None:
        return 1

    print(f"  wczytany bez wykonywania kodu (weights_only), {len(policy.net.state_dict())} wpisow wag, "
          f"dopasowanie scisle")
    for key in sorted(policy.schema):
        print(f"  {key}: {policy.schema[key]!r}")
    print("  statystyki obserwacji (z pliku):")
    for i in range(len(policy.obs_mean)):
        print(f"    {i:2d}  srednia {policy.obs_mean[i]:+.6f}  odchylenie {policy.obs_std[i]:.6f}")
    return 0


def time_inference(a: argparse.Namespace, deps: Deps) -> int:
    from galaxeo.grasp.policy import TIME_MEDIAN_MAX_MS, TIME_P95_MAX_MS

    policy = _loaded(a, deps)
    if policy is None:
        return 1

    t = policy.time(a.n)
    print(f"Inferencja na CPU ({a.n} wywolan po pierwszym): mediana {t.median_ms:.1f} ms, "
          f"p95 {t.p95_ms:.1f} ms, pierwsze {t.first_ms:.1f} ms")
    limits = f"mediana < {TIME_MEDIAN_MAX_MS:.0f} ms, p95 < {TIME_P95_MAX_MS:.0f} ms"
    fast = t.median_ms < TIME_MEDIAN_MAX_MS and t.p95_ms < TIME_P95_MAX_MS
    print(f"Werdykt: OK ({limits})" if fast else f"Werdykt: ZA WOLNO (wymagane: {limits})")
    return 0 if fast else 1


def _positive(text: str) -> int:
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError(f"oczekiwano liczby >= 1, jest {text}")
    return n


def _loading_command(sub: Any, name: str, text: str) -> argparse.ArgumentParser:
    p = sub.add_parser(name, help=text)
    p.add_argument("path", help="plik modelu")
    p.add_argument("--sha256", default=CHECKPOINT_SHA256, help="sha256, ktory plik musi miec (domyslnie opublikowany)")
    return p


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="pobierz opublikowany model przez HTTPS i sprawdz sha256")
    p.add_argument("path", help="plik docelowy (nigdy domyslny: podaj, gdzie ma lezec)")
    p.set_defaults(run=fetch)

    p = _loading_command(sub, "check", "rozmiar, sha256 i wczytanie wag (weights_only, scisle)")
    p.set_defaults(run=check)

    p = _loading_command(sub, "time", "czas inferencji na CPU (mediana < 30 ms, p95 < 45 ms)")
    p.add_argument("--n", type=_positive, default=200, help="ile wywolan po pierwszym (domyslnie 200)")
    p.set_defaults(run=time_inference)
    return ap


def main(argv: list[str] | None = None, deps: Deps | None = None) -> int:
    a = parser().parse_args(argv)
    return a.run(a, deps or Deps())


if __name__ == "__main__":
    sys.exit(main())
