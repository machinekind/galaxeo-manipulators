# Grasp student fixtures

Recorded from the training environment (`galaxeo-manipulators`, `sim/grasp`) by
`scripts/record_grasp_fixtures.py`. The tests of `lerobot_mp.twin.grasp` and of the warm IK step
compare the twin against these files. No test runs the training code.

## Re-recording

The training repository is read only: its code is imported, nothing is written into it and git
is never run there. Run from a scratch directory, with the training repository's own environment:

```
mkdir -p <scratch>/fixtures_out && cd <scratch>/fixtures_out && PYTHONDONTWRITEBYTECODE=1 \
  <galaxeo>/sim/.venv-lerobot/bin/python <twin>/scripts/record_grasp_fixtures.py \
  --galaxeo <galaxeo> --out <scratch>/fixtures_out \
  --seed 50000 --shard-episodes 2 --shard-ticks 40 --book 1000 --proj 300 --fk 300 --hover 200 --ik 200
cp <scratch>/fixtures_out/*.npz <scratch>/fixtures_out/*.json <twin>/tests/fixtures/grasp/
```

`--galaxeo` defaults to `$GALAXEO_REPO`. The recorder refuses to run from, or write into, the
training repository. It always gives the environment the published model's camera,
`wrist_size=(160, 120)` and `wrist_jitter=(3.0, 0.7, 0.3)`; training's defaults (128x96 or
320x240, 10 mm) do not describe it. It takes about 15 s. A test that finds a key missing fails
with "re-record the fixtures (docs/GRASP.md)".

## The substep lag and the `*_fk` fields

After `mj_step`, MuJoCo's positions (`site_xpos`, `xpos`, `xmat`, the rendered image,
`believed_uv`, the TCP inside training's own `obs`) belong to the state BEFORE the last 2 ms
substep, while `q = d.qpos` belongs to the state after it. The twin computes every pose by forward
kinematics from `q`, so it cannot reproduce those recorded values exactly. The recorder therefore
also stores what training's own model gives AT `q` (a scratch `MjData`, `mj_kinematics`, no step):
`tcp_fk`, `tcp_R_fk`, `gl_pos_fk`, `gl_R_fk`, and from them `uv_fk` (the body of
`GraspEnv.believed_uv`) and `obs_fk` (`GraspEnv.student_obs` with `tcp_fk` in place of the lagging
TCP: only indices 7..9 differ from `obs`). Before storing them the recorder checks that its copies
of the two formulas reproduce `obs` and `uv` bit for bit from the lagging poses.

Measured lag in this recording (also in `meta.json`, `substep_lag`):

| file | max \|obs - obs_fk\| | max \|uv - uv_fk\| | max TCP or gripper_link position |
| --- | --- | --- | --- |
| `shard.npz` | 2.40e-3 | 0.27 px | 0.24 mm |
| `hover.npz` | 1.23e-3 | 0.15 px | |
| `projection.npz` | | 0.30 px in the frame, 1.06 px within 640 px of the centre, 10.96 px over all samples | 0.26 mm |

The large projection figure comes from points far outside the frame (u about -1100), where the same
lag moves the pixel by several pixels.

The twin's model gives the `*_fk` poses to 4e-16 m, `uv_fk` to 2e-5 px and `obs_fk` to 2.3e-7.

## Files

Positions are in the training WORLD frame unless the name or the table says base; base = world -
`arm_base` (0, -0.20, 0.70), no rotation. Joints are radians, order joint 1..6.

| file | contents |
| --- | --- |
| `shard.npz` | 59 student-driven ticks of 2 episodes (seeds 50000, 50001; both succeed). `img` (59,120,160,3) uint8, `obs` (59,19) f4, `uv` (59,2) f4, `act` (59,5) f4 (student output for exactly these inputs, batch 1); raw `q`, `gap`, `tcp`, `tcp_R`, `believed_pos`, `believed_yaw`, `yaw_cmd`, `g_cmd`, `prev_action`, `attempts`, `gl_pos`, `gl_R`, `episode`, `tick`; `tcp_fk`, `tcp_R_fk`, `gl_pos_fk`, `gl_R_fk`, `uv_fk` f4, `obs_fk` f4 |
| `bookkeeping.npz` | 1000 random actions in [-1.3, 1.3]^5 through `GraspEnv.step`, forced close and shove off: `action`, state before (`p_cmd0 yaw_cmd0 q_cmd0 p0 g_cmd0 closing0 attempts0`), the env's IK call (`ik_pos ik_yaw ik_R ik_q0`, 60 iterations) and result (`ik_q ik_ok`), state after (`p_cmd1 yaw_cmd1 q_cmd1 g_cmd1 closing1 attempts1 prev_action1`), `ctrl_arm`, `ctrl_grip`, `obs12_18`, `episode`, `tick`. IK accepted 990 of 1000 |
| `projection.npz` | 300 samples: `q`, `gl_pos`, `gl_R`, `believed_world`, `believed_base`, `uv` (3 NaN; 176 in the frame, u from -1152 to 182), `gl_pos_fk`, `gl_R_fk`, `uv_fk` (NaN where `uv` is) |
| `fk.npz` | 300 random `q` in the joint ranges: `tcp_base`, `tcp_R`, `gl_base`, `gl_R`, computed by `mj_kinematics` at `q` (no step, no lag) |
| `heat.npz` | `uv` (7,2): centre, corners, the tick-0 pixel, off-frame, NaN; `heat` (7,1,120,160) from training's `heat_channel` |
| `hover.npz` | 200 episode starts (`reset(seed=69000 + k)`): `p_hover` and `believed` (base frame, `p_hover = believed + [0, 0, 0.06]`), `yaw` (closing axis, = `believed_yaw` = `yaw_cmd` at tick 0), `q_hover` (the commanded joints training solved), `q_measured` (after the 0.1 s settle), `gap0`, `uv0`, `obs0` (tick-0 inputs as training computed them), `uv0_fk`, `obs0_fk` (at `q_measured`), `episode` |
| `ik.npz` | 200 calls of training's `Arm.ik(q0, pos, R, iters=60)`, 10 steps from each of 20 hovers: `q0` (seed), `pos` (base frame), `R` (`_grasp_R(yaw)`), `q`, `ok`; also `yaw`, `hover`, `step`, `j6_limit`. The steps follow `GraspEnv.step`: random actions in [-1.3, 1.3], box, floor, ceiling; the seed is the last accepted `q`, as in training. In every fifth hover the middle step asks for a yaw that needs J6 10 deg beyond its range (`j6_limit`); those 4 are refused (J6 stops at 165 deg), the other 196 are accepted |
| `state_keys.json` | the 130 entries of the checkpoint's `net`, in order, as `[name, shape, dtype]`; `dtype` is `str(tensor.dtype)`, for example `"torch.float32"`, `"torch.int64"` |
| `meta.json` | wrist camera pose in `gripper_link` (OpenCV axes), `fovy_deg`, `f_px`, `cx`, `cy`, `arm_base`, `joint_lo`/`joint_hi`, `obs_mean`/`obs_std` (float32 values from the checkpoint), step constants (`step_xyz`, `step_yaw_deg`, `box_xy`, `hover`, floor and ceiling rules, `control_hz`, `substeps`, `timestep`, `max_steps`, the IK settings), `calibration_joint_offsets_deg` (six zeros: the camera pose was calibrated on raw joint angles), `recorder_args`, `training_repository` (branch and HEAD commit, read from its `.git` files; local changes are not seen), `checkpoint` (path in the training repository), `checkpoint_sha256`, `checkpoint_bytes`, `recorded_with` (versions), `substep_lag` |

The fields that the earlier recording already had are bit-identical to it.
