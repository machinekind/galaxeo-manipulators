"""What one grasp run leaves in its dump directory, for the operator.

    dumps = Dumps(Path(settings.dump_dir), every=settings.dump_every)
    dumps.tick(record)                          # one line of ticks.jsonl per policy tick
    dumps.frame(n, image, uv, extra, heat)      # NNNN.png and NNNN_heat.png every `every` ticks
    dumps.close(summary)                        # summary.json

`NNNN.png` is the model's image input four times enlarged, with the believed pixel (cross and
its 8 px circle) and the projected centre of the object's top face (small square): the distance
between the square and the real top face is how the operator reads the dot error now that
there is no dry mode. `NNNN_heat.png` is the heat channel the network gets.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any

import cv2
import numpy as np

from .camera import draw_dump
from .rig import plain

@dataclass
class TickRecord:
    """One policy tick, one line of ticks.jsonl."""

    tick: int
    t: float                          # obs.t [s]
    kind: str                         # "policy" | "skipped"
    skip: str                         # "" or the soft guard that skipped the tick
    action: list[float] | None
    uv: list[float] | None
    p_cmd: list[float]
    yaw_cmd: float
    g_cmd: float
    attempts: int
    ik_ok: bool | None
    ik_why: str                       # "" | "ik" | "jump" | "collision"
    ik_pos_err: float | None
    ik_rot_err: float | None
    ik_jump_deg: float | None
    box_hit: bool
    floor_hit: bool
    ceiling_hit: bool
    floor_z: float
    sag: float
    tcp: list[float]
    gap: float
    lag_deg: float
    rate_limited: bool                # judged on the next tick: the previous target against obs.command
    frame_age_s: float | None
    pair_gap_s: float | None
    infer_ms: float | None


class Dumps:
    """The dump directory of one run; nothing is written without a directory."""

    def __init__(self, folder: Path, *, every: int) -> None:
        self.folder = Path(folder).expanduser()
        self.every = int(every)
        self.folder.mkdir(parents=True, exist_ok=True)
        self._ticks: IO[str] | None = (self.folder / "ticks.jsonl").open("w", encoding="utf-8")

    def due(self, n: int) -> bool:
        """Whether policy tick `n` (from 0) gets an image dump."""
        return self.every > 0 and n % self.every == 0

    def tick(self, record: TickRecord) -> None:
        if self._ticks is None:
            return
        self._ticks.write(json.dumps(plain(asdict(record))) + "\n")
        self._ticks.flush()

    def frame(self, n: int, image: np.ndarray, uv: np.ndarray, extra: Mapping[str, np.ndarray],
              heat: np.ndarray) -> None:
        drawn = draw_dump(image, uv, extra=extra)
        cv2.imwrite(str(self.folder / f"{n:04d}.png"), cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))

        grey = np.clip(np.asarray(heat, np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
        cv2.imwrite(str(self.folder / f"{n:04d}_heat.png"), grey)

    def close(self, summary: Mapping[str, Any] | None) -> None:
        if self._ticks is not None:
            self._ticks.close()
            self._ticks = None
        if summary is None:
            return
        # `plain` turns numpy values into plain ones and a non-finite float into null.
        text = json.dumps(plain(dict(summary)), indent=2, ensure_ascii=False)
        (self.folder / "summary.json").write_text(text + "\n", encoding="utf-8")
