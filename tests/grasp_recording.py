"""The training recording in tests/fixtures/grasp, read by the tests of galaxeo.grasp."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

FIXTURES = Path(os.environ.get("GRASP_FIXTURES", Path(__file__).parent / "fixtures" / "grasp"))
RERECORD = "re-record the fixtures (tests/fixtures/grasp/README.md)"


def npz(name: str, *keys: str) -> dict[str, np.ndarray]:
    """The named arrays of one recorded file; a missing file or key fails the test."""
    path = FIXTURES / name
    if not path.is_file():
        pytest.fail(f"{path} is missing: {RERECORD}")
    with np.load(path) as z:
        missing = [k for k in keys if k not in z.files]
        if missing:
            pytest.fail(f"{name} has no {', '.join(missing)}: {RERECORD}")
        return {k: z[k] for k in keys}


def meta() -> dict[str, Any]:
    path = FIXTURES / "meta.json"
    if not path.is_file():
        pytest.fail(f"{path} is missing: {RERECORD}")
    return json.loads(path.read_text(encoding="utf-8"))


def meta_value(recorded: dict[str, Any], key: str) -> Any:
    if key not in recorded:
        pytest.fail(f"meta.json has no {key!r}: {RERECORD}")
    return recorded[key]


def state_keys() -> list[list[Any]]:
    path = FIXTURES / "state_keys.json"
    if not path.is_file():
        pytest.fail(f"{path} is missing: {RERECORD}")
    return json.loads(path.read_text(encoding="utf-8"))
