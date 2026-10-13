"""galaxeo.robot: the Galaxea A1X as a robot package for the digital twin (`lerobot_mp`).

The twin holds no robot and this package never imports it. The twin finds the A1X through two
entry points registered in `pyproject.toml`:

* `lerobot_mp.robots`: `a1x = "galaxeo.robot:ROBOT_DIR"`, the folder with `robot.yaml`, the MuJoCo
  model and the example workspace `demo.json`;
* `lerobot_mp.backends`: `a1x = "galaxeo.robot.backend:create"`, the CAN-FD hardware backend.
"""

from pathlib import Path

ROBOT_DIR = Path(__file__).parent / "a1x"
