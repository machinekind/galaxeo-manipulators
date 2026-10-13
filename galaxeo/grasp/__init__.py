"""galaxeo.grasp: the wrist-camera grasp student, a policy trained in simulation, as a strategy for
the digital twin (pip install "galaxeo[grasp]": numpy, torch, opencv-contrib-python).

The published model is `hvsr-robotics/a1x-wrist-grasp-sim` on Hugging Face. The twin's
`StrategyRunner` calls the strategy's `act(obs)` at its `hz` and owns claiming the arm, safety
and stopping. Nothing in this package imports the twin: joints [deg] and the wrist camera frame
come from the observation, kinematics from `galaxeo.arm`.

Modules, from the bottom (each imports only modules named before it):

contract  the training contract: constants, units and the helpers `wrap`, `grasp_R`, `yaw_of`
rig       per-rig measured values (a JSON file at a path the caller gives) and the refusal reasons
command   command bookkeeping, a verbatim port of the training environment's step
net       the network: ResNet18 rebuilt in plain torch, without torchvision
policy    the checkpoint (sha256 before unpickling, weights only) and `act` at batch 1
camera    the wrist camera's geometry: camera frame to policy frame, base-frame points to pixels
observer  the 19-vector the policy reads, from the joints paired with the frame
hover     the hover above the believed grasp, and the staged joint plan to it
dumps     ticks.jsonl, summary.json and the model-input dumps of one run
strategy  the twin strategy `Grasp`: settings, start (the refusals), act (the phases), stop

The twin finds `Grasp` through the `lerobot_mp.strategies` entry point `grasp`.

Nothing is imported here, so `contract`, `rig` and `command` import without torch or OpenCV.
"""
