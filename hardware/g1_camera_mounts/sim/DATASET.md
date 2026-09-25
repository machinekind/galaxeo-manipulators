# Adding the wrist stream to the datasets

Everything here is a change to `sim/`, not to this package. Nothing below is
applied: the mount is proven in simulation (`test_wrist_camera.py`), the stream
is opt-in, and turning it on is a choice about the next dataset, not about this
PR.

## 1. Put the camera on the arm

Both scene builders compile the arm from one cached `MjSpec` and attach a copy
of it per scene, so one call in the factory is the whole change.

`sim/pour_scene.py`:

```python
MOUNT = os.path.join(os.path.dirname(HERE), "hardware", "g1_camera_mounts", "sim")
sys.path.insert(0, MOUNT)
from wrist_camera import attach_wrist_camera            # noqa: E402

WRIST_HAND = os.environ.get("WRIST_CAMERA", "")         # "right", "left" or off


def _arm_spec():
    global _ARM
    if _ARM is None:
        _ARM = mujoco.MjSpec.from_file(os.path.join(HERE, "a1x.xml"))
        if WRIST_HAND:
            attach_wrist_camera(_ARM, hand=WRIST_HAND)
    return _ARM
```

`HERE` is `sim/`, so that path assumes `sim/` and `hardware/` are siblings at
the repo root, which is where they land once this branch and the pour branch
are stacked. The same four lines work in `sim/pickplace_scene.py`, whose
`_arm_spec` is identical. After `spec.attach(..., prefix="arm/")` the camera is
`arm/wrist` and the payload body is `arm/wrist_camera_mount`.

Off by default, because attaching it changes the dynamics: 103.7 g on the wrist
and 57 collision primitives the planner has to route around. A dataset recorded
with the mount and a policy deployed without it (or the other way round) do not
match.

The pour planner rolls the wrist negatively on most seeds, which swings a +Y
payload down toward the table and the glass — that is the reason the left hand
is preferred, and the reason is more durable than the margin.

<!-- generated: dataset-handedness -->
Recommended hand: **left**. The harness picked the same hand on all 3 seed ranges (0-19 → left, 100-119 → left, 200-219 → left), but a range is 20 episodes, not a proof.
The two hands do not see the same task either — glass in frame at the grasp, per range: seeds 0-19, right 6 / left 6; seeds 100-119, right 11 / left 6; seeds 200-219, right 4 / left 8. The bottle mouth is in frame on essentially none of them.
What the left costs is the full-arm audit: 74 payload collisions in 10,000 configurations against the right hand's 61, on the wrong side of v4's 64 — a disclosed miss, with about ±8 of sampling noise on each count. With the §1b patch the planner refuses those configurations; without it, build the right hand and expect the glass contact. The README's *Wrist roll* section has every table.
<!-- /generated -->

Not using `MjSpec`? `python -m wrist_camera --xml mount.xml` writes the same body
as MJCF, with the mesh path relative to wherever you wrote it, to paste inside
`<body name="gripper_link">`.

## 1b. Patch the planner's collision filter — required

`planner/motion.py` decides what a contact means by prefix:

```python
mine = [b.startswith(ARM) or b in carried for b in (b1, b2)]
if mine[0] == mine[1]:            # self-contact, or two things I do not control
    continue
```

The mount compiles as `arm/wrist_camera_mount`, so **every mount-vs-arm and
mount-vs-held-object contact hits that line and is skipped.** Unpatched, the
collision geoms this package adds protect against the table, the bottle and the
glass, and against nothing the arm does to itself.

<!-- generated: filter-counts -->
Measured on pour seed 0, jaw 20 mm, over **6,000 random configurations** of the six arm joints (`--phase filter`): **330** put a mount geom against another `arm/` body with the left hand and `Motion.collides` returned `None` for **275** of them; **333** put a mount geom against another `arm/` body with the right hand and `Motion.collides` returned `None` for **276** of them. With the patch below, the same counts and **0**.
<!-- /generated -->

```diff
--- a/sim/planner/motion.py
+++ b/sim/planner/motion.py
@@
 ARM = "arm/"
+MOUNT_BODIES = ("arm/wrist_camera_mount",)
+# The mount's own parent; MuJoCo's parent filter excludes the pair anyway.
+MOUNT_EXEMPT = ("arm/gripper_link",)
@@ def collides
             b1, b2 = self.body_of_geom[c.geom1], self.body_of_geom[c.geom2]
+            # A wrist payload is bolted to the arm, but it is not "self": the arm
+            # can drive it into its own links and into the object it is carrying,
+            # and both of those are collisions a planner has to refuse.
+            on_mount = [b in MOUNT_BODIES for b in (b1, b2)]
+            if on_mount[0] != on_mount[1]:
+                g = c.geom2 if on_mount[0] else c.geom1
+                other = b2 if on_mount[0] else b1
+                label = self.geom_name[g] or other or "world"
+                if other not in MOUNT_EXEMPT and label not in allow and other not in allow:
+                    return label
+                continue
             mine = [b.startswith(ARM) or b in carried for b in (b1, b2)]
```

`test_wrist_camera.py --phase pour --guard` applies exactly that patch to a copy
of `planner/` on a temp path — it writes nothing under `sim_pour/` — and re-runs
the pour regression with it. One wrinkle if you do it by hand: `pour_scene` puts
its own directory at `sys.path[0]` when it is imported, and that directory
contains `planner/`, so a shadowing copy has to go on the path *after*
`import pour_scene` (and `planner*` purged from `sys.modules`), not before. The README quotes both runs side by side. The same
patch is needed in PR #1's `planner/` if you mount the camera there.

## 2. The feature key

LeRobot v3, alongside the two streams already there, same fps:

```
observation.images.wrist      (480, 640, 3)  uint8 video
```

640 x 480 is the camera's native resolution (`camera_spec.json`
`resolution_px`) and what a UVC board camera delivers, so the sim frame and the
real frame are the same shape with no resize on either side. The laptop stream
is stored at 320 x 240 because its 640 x 480 render exists only to feed the
top-down warp; the wrist frame feeds the policy directly, so it is stored at
full size. If bandwidth matters more than detail, store 320 x 240 and say so in
the dataset card — but do it for both the dataset and the deployment.

No `wrist_K` / `wrist_T` feature, **and therefore no field-of-view jitter** — the
two go together and section 4 below keeps them together. The laptop's calibration
is in the dataset because it is *unknown per session*; the wrist camera's pose in
the gripper is fixed by the mount's datum (gate G4: 481 mm² of measured axial stop
contact, and a grub screw that makes the clocking a plane-on-plane contact rather
than a fit), so it is a constant of the robot, not of the episode. It belongs in
the model card, and the hand-eye prior is `camera_spec.json`.

If you do want lens-to-lens variation in the dataset, turn `fovy_deg` jitter on
*and* add `observation.wrist_K` as a 9-float feature — 36 bytes a frame. What you
must not do is jitter `fovy` and store nothing, which is what the first cut of
this recipe asked for: a ±3° draw on an 85° vertical field is ±4 % of focal
length, and nothing downstream could recover it.

## 3. The recording change

`gen_dataset.py` renders nothing. It spawns one `planner/run_pour.py --lerobot`
per seed range and merges the parts; all rendering is in `run_pour.py`'s
`LeRobotRecorder`, which holds one `mujoco.Renderer` and calls
`update_scene(d, camera="laptop_cam")` once per stored frame. Adding a stream
means one more `update_scene` / `render` on the same renderer — the renderer is
sized 640 x 480 already, which is the wrist size too.

Against `sim/planner/run_pour.py`:

```diff
@@
 from planner.topdown import N as TD_N, SIDE as TD_SIDE, topdown
 from pour_scene import WORKSPACE, build
+
+WRIST = os.environ.get("WRIST_CAMERA", "")      # same switch pour_scene reads
@@ class LeRobotRecorder:
         self.r.update_scene(d, camera="laptop_cam")
         full = self.r.render().copy()
         td = topdown(full, self.K3, self.T_cam2world, self.centre, self.td_side, self.td_n)
         img = cv2.resize(full, self.wh, interpolation=cv2.INTER_AREA)
+        frame = {}
+        if WRIST:
+            self.r.update_scene(d, camera="arm/wrist")
+            frame["observation.images.wrist"] = self.r.render().copy()
         q, g = d.qpos[self.arm.qadr], abs(float(d.qpos[self.arm.fadr[0]]))
         act = np.concatenate([d.ctrl[self.arm.acts], [np.clip(d.ctrl[self.arm.grip], 0.0, 0.05)]])
         self.ds.add_frame({"action": act.astype(np.float32),
                            "observation.state": np.concatenate([q, [g]]).astype(np.float32),
                            "observation.images.laptop": img,
                            "observation.images.topdown": td,
                            "observation.cam_K": self.K, "observation.cam_T": self.T,
-                           "task": TASK})
+                           "task": TASK, **frame})
@@ def open_lerobot(root, repo_id, fps, W, H, td_n=TD_N):
         "observation.images.topdown": {"dtype": "video", "shape": (td_n, td_n, 3), "names": hwc},
+        **({"observation.images.wrist": {"dtype": "video", "shape": (480, 640, 3), "names": hwc}}
+           if WRIST else {}),
         "observation.cam_K": {"dtype": "float32", "shape": (9,), "names": None},
```

And against `sim/gen_dataset.py`, only to pass the switch to the workers, since
it renders nothing itself:

```diff
@@ def main():
     ap.add_argument("--keep-parts", action="store_true", help="do not delete part_k/ after merging")
+    ap.add_argument("--wrist", choices=("right", "left"),
+                    help="also record observation.images.wrist from the G1 wrist camera")
     args = ap.parse_args()
+    if args.wrist:
+        os.environ["WRIST_CAMERA"] = args.wrist       # inherited by every worker
```

`Popen` already passes the parent environment on, so that one line is the whole
change: the flag reaches `pour_scene` and `run_pour` in every worker without a
new argument on either. The DAgger path gets it for free —
`planner/dagger.py` imports `LeRobotRecorder` and `open_lerobot` from
`run_pour`, and `gen_dagger.py` imports `run_workers` from `gen_dataset`. Check
one thing there before trusting it: `dagger.py` buffers frames and rewinds, so
whatever it does to drop a frame has to drop all three streams together.

Cost, measured on a laptop at 20 Hz on pour seed 0: the wrist render is
**5.8 ms** per stored frame, against 15.6 ms of physics (25 steps) and 6.1 ms
for the laptop render — so about **+27 %** of the part of an episode this venv
can measure. The top-down warp, the resize and the video encoder live in
`.venv-lerobot` and are not in that figure, so the real slowdown is smaller.
Dataset size is a third video track of the same pixel count as the laptop
render; do not take a number on faith, run one worker for a few episodes and
read `meta/info.json` before launching 300.

## 4. Randomisation for the wrist stream

The base camera is randomised because its pose is unknown. The wrist camera's
pose is *known* — which is exactly why it must be jittered, or the policy will
learn this one bracket's tolerances instead of a wrist camera.

Do it on the **compiled model**, in `build()`, not at attach time: `_arm_spec`
caches one spec for the whole run, so a draw made there is made once. One line
after `spec.compile()`, from its own RNG stream so a seed keeps its draw:

```python
model = spec.compile()
if WRIST_HAND:
    jitter_camera(model, np.random.default_rng([int(seed), 0xCA11]),
                  pos_mm=2.0, rot_deg=1.5)      # fovy_deg defaults to 0; see section 2
```

What each number stands for, and why not more:

| draw | range | what it covers |
| --- | --- | --- |
| `pos_mm` | ±2 mm per axis | clamp band re-clocking on the liner plus M2 slot play; the datum holds the axial position to well under this |
| `rot_deg` | ±1.5° per axis | a printed strut's own twist and the M12 thread's seating angle. Clocking is *not* in this budget: with the grub screw seated it is a plane on a plane; without it the key alone allows 4.4°, which is not a jitter, it is an assembly error |
| `fovy_deg` | 0 by default | see section 2. Turning it on without storing `wrist_K` puts an unrecoverable ±4 % focal length in the dataset |
| exposure | gain ±25 %, gamma 0.8..1.25 | a wrist camera is 60 mm from the object and swings through the light; it will not hold the base camera's exposure |

The jitter moves only the camera, not the collision geometry: the bracket's
shape is known exactly and a planner must not be taught otherwise. (The same
draw is available at attach time as `attach_wrist_camera(..., jitter=...,
rng=...)`, for anyone who does build a spec per episode.)

Exposure is not a MuJoCo setting; do it on the frame, after the render and
before storing, the same photometric pipeline `train_pour.sh` already uses for
augmentation (affine augmentation stays off — it would break the top-down
stream's relation to the calibration it carries).

**Light the scenes hard.** The bracket does not appear in the frame, and the
reason is not the renderer's near plane — that story was wrong, and it mattered,
because a near-plane artifact would not survive on hardware. The near plane here is
17.3 mm (`stat.extent * vis.map.znear`) and

<!-- generated: mount-depth -->
the mount's surface inside the frustum spans **51.4-77.6 mm** along the optical axis (`validation.json` → `G3_occlusion.recommended_wide.self_and_gripper.mount_depth_in_frustum_mm`), so thousands of its pixels are past it. They are hidden because **the gripper is in front of the strut** on every one of those rays, which is geometry and holds on the real camera too. The CAD ray cast agrees once the gripper is included as an occluder: 3.2 % of the wide frame is mount with the gripper removed, 0 % with it there.
<!-- /generated -->

What the bracket does do is cast a shadow: it darkened 17.7 % of the wide-lens
frame at the home pose of seed 0 and 0 % at the pre-grasp pose of the same scene
(`test_wrist_camera.py --phase frames`, `own_image_share`). That is a large,
moving, pose-dependent brightness pattern that a policy can learn instead of the
task. `pour_scene` already draws one to three lights per seed; keep that stream on
for wrist runs, and treat the exposure jitter in the table above as required rather
than optional.

## 5. On the real arm

1. **Capture.** A UVC board camera at 640 x 480. Give it its own V4L2 node and
   fix exposure and white balance; auto-exposure hunting on a moving wrist
   camera destroys temporal consistency. Record it with the same timestamps as
   the laptop stream and the joint states, and resample all three to the
   dataset's fps (20 Hz) — the wrist camera is the one stream whose content
   changes fast enough that a 50 ms misalignment is visible.

2. **Hand-eye prior.** `camera_spec.json`'s `T_gripper_camera_opencv` is the
   nominal pose of the lens in `gripper_link`, OpenCV convention (z forward,
   y down), metres, from the CAD. Mirror it in Y for the left hand. It is a
   starting guess, not a calibration; `wrist_camera.py`'s `camera_pose()`
   returns the same thing in MuJoCo convention.

3. **Refining it.** `calib/handeye.solve` does this unchanged. It is written for
   eye-to-hand (fixed camera, tags riding the arm) and minimises

   ```
   corners ~= project( T_base2cam . T_frame2base_i . T_mount2frame . T_tag2mount . X )
   ```

   over the constant `T_cam2base` and every constant `T_mount2frame`. Eye-in-hand
   with a tag standing still on the table is the same equation with the terms
   renamed:

   ```
   X_cam = T_cam2gripper^-1 . T_base2gripper_i . T_tag2base . X
   ```

   So put the **calibration card flat on the table** (or any 36h11 plate), wave
   the *wrist* over it, and feed `solve` each pose with

   | solver argument | what to pass |
   | --- | --- |
   | `T_frame2base` | `inv(FK_gripper2base)` at the measured joint angles — the inverse of what `calibrate.py` passes |
   | `T_tag2mount` | the tag's pose in the plate (identity for a single tag) |
   | `mounts_nominal` | the table plate's guessed pose in the base, for `nominal_init` |

   and read the result as: **`fit.T_cam2base` is `T_camera2gripper`** and
   `fit.mounts[...]` is the plate's pose in the base. `handeye_init`'s closed
   form, `rotation_spread` and the acceptance thresholds (`MAX_PX`,
   `MIN_SPREAD`, `MIN_OBS`) all carry over with the same substitution, because
   they are algebra on the same equation. Seed with
   `T_gripper_camera_opencv` above.

   Expect the fit within a few millimetres and a degree or two of the nominal.
   If it is not, the bracket is not seated: check the axial stop against the
   rail back face and the key ears on the rail flats before believing the
   numbers. That is what the datum is for — after a camera swap the extrinsics
   come back, so this is a commissioning step, not a per-session one.

4. **What the stream is worth.** Honest limits from the sim study, before anyone
   trains on it, and visible in `renders/pour_right_sequence.png`: on a body
   grasp the bottle's mouth is *behind* the camera plane, so neither lens sees
   the pour point, and during the roll the glass leaves the frame as well — the
   last two tiles of that sheet are bottle shoulder and sky. The wrist view
   carries the
   approach, the grasp and the glass; the mouth-over-rim part of the task stays
   on the base camera or on a neck grasp. Do not drop the laptop stream.
