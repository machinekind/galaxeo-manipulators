# Fixed wrist-camera mount — revision 4 prototype

**The previous mounts can hit the upper arm in folded poses.** Revision 4 moves
the clamp forward and changes the fixed angle to move the camera forward while
retaining approximately 100 mm nominal working distance. It clears the demonstrated
collision pose, but **does not permit unrestricted full-arm motion**. Include the
payload in self-collision checks. Exporting a URDF does not activate such checks in
your controller.

![Actual arm geometry, previous collision and revised mount](arm_clearance.png)

## What changed

- Clamp centre moves from X=−45 to **X=−26 mm** behind the G1 front body face.
  Its 18 mm band is now **X=−35..−17 mm**. Check that band is unobstructed on your unit.
- Fixed camera pitch is **75° downward**. This is still a rigid bracket with no
  adjustment joint. The nominal board lens moves **28.3 mm forward**.
- Nominal 60 mm model lens centre: **(49.12, 0, 96.59) mm**. It looks toward
  **(75, 0, 0) mm**, 100 mm away. This preserves focus distance instead of placing
  the camera very close to the fingers.
- Ribs rise clear of the gripper rail before extending forward. The collar's
  front face has **3 mm axial separation** from the conservative rail envelope.
- Still **two main printed parts**, two M4 clamp bolts and the camera fasteners.
  The lower clamp has the same physical shape as v3, installed 19 mm farther forward.

## Results, including failures

| Check | Result |
| --- | --- |
| Continuous bound over wrist joints 4/5/6, all their URDF limits | At least **27.1 mm** separation from links 3–6 |
| Same folded pose that intersects the old upper/lower clamp | Revised mount/camera clears arm by at least **13.6 mm** |
| 10,000 sampled full-arm configurations | 9,403 have no detected bare-arm collision |
| Added collisions among those bare-arm-clear samples | **64 board-camera poses; 103 webcam poses** |
| Housing, full jaw envelope, PCB and fastener clearances | Passed for the modelled envelopes |
| Sampled grasp sightlines | 4,500 tested, zero blocked |
| Print meshes | Nine watertight STL files, one CAD solid per part |

The wrist result is a conservative separating-plane bound, including interpolation
error; it is stronger than a coarse angle sweep. It covers the 60 mm housing model,
printed bracket, stated camera envelopes and represented fasteners. It **excludes
links 0/1/2**, which can approach the payload in folded configurations. The 57 mm
alternative is not certified against this 60 mm arm model.

Full-arm counts are deterministic triangle-surface samples, not a probability of
collision or a proof that unlisted configurations are clear. Some vendor meshes are
open. Camera cables, exact electronics/connector details, the environment, printing
error, deflection, strength and physical fit are not validated. This remains a
prototype awaiting the actual camera model and physical assembly.

`arm_validation.json` records the test conditions and failing poses.
`comparison_pose.json` records the identical before/after pose shown above.

## Parts and assembly

Print from **`stl/` only**. Units are millimetres; the parts are already oriented
for printing. `collision/` contains assembly-coordinate meshes for software, not
print files.

| Camera | Upper bracket | Other main part |
| --- | --- | --- |
| 32 × 32 mm SO-101-style board | `board_fixed75_upper_60mm.stl` | `lower_clamp_60mm.stl` |
| Webcam with ¼″-20 socket | `webcam_fixed75_upper_60mm.stl` | `lower_clamp_60mm.stl` |

The repository G1 housing is **60 mm outside diameter**. Measure yours and print
`fit_gauge_60mm.stl` first. A 57 mm alternate set is provided only for a measured
57 mm housing; do not scale the STLs. Gauges have 0.6 mm diametral clearance.

For either mount, use **2 × M4×20 socket screws, 2 ordinary M4 nuts and 2 head
washers**. A roughly 0.5 mm rubber liner provides grip. Fit the halves around the
stationary housing at the stated position, with the bracket above the gripper.
Tighten evenly without forcing the split gaps closed. Confirm cables, connectors
and housing details leave this band clear.

Board camera: use **4 × M2×12 screws/nuts and four 3 mm insulating spacers**
(`m2_spacer_3mm.stl` is included). Slots accept a **26–30 mm square hole pattern**.
The frame has a 20 mm central opening. Check rear components and connectors against
the frame; a generic 32 mm board is not an exact SO-101 camera specification.

Webcam: insert a real **¼″-20 screw from underneath** through the **6.8 mm hole**
into the camera's socket. The plate is 6 mm thick. Select screw length for plate +
washer/pad + the camera's permitted engagement; it must not bottom out. Add a thin
non-slip pad. The checked camera envelope is **45 mm front/back × 74 mm wide ×
45 mm high**, with its socket centred and optical axis parallel to the plate.
A 50 mm-long, 10 mm-diameter driver corridor beneath the screw is checked.

Suggested prototype printing: PETG, 0.2 mm layers, 5 perimeters, 40–50% infill.
Upper brackets are exported sideways and **require supports** below the elevated
rib, camera platform and clamp arc. Check the slicer preview. These settings are
not a load rating. Verify fit, fastener access, jaw travel and cable strain relief
on the stationary arm before a slow, collision-checked motion trial.

## Camera requirements

| Across tested optical-centre cases | Board | Webcam |
| --- | --- | --- |
| Focus distances to sampled grasp region | 95.7–111.2 mm | 86.8–125.4 mm |
| Required horizontal FoV, minimum | 53.8° | 57.9° |
| Required vertical FoV, minimum | 17.2° | 38.3° |

Allow margin above those fields of view. Webcam optical-centre cases cover lens
heights of 10/24/40 mm above its mounting plane and 0/20/22.5 mm forward of its
socket, with no lateral lens offset. The view images assume a 70° × 55° pinhole
camera. Exact camera focus, FoV, orientation, connectors and socket dimensions
remain to be checked. The grasp samples are X=60..76, Z=0, inside each jaw by 2 mm,
with jaw openings of 10/20/40/70/100 mm. Objects and closed fingers can still hide
parts of a real grasp.

## Robot model and editable files

`step/` has editable CAD parts and assembled mounts. The `.glb` files include the
actual complete arm in a reference pose, in metres, for inspection in a 3D viewer.

`collision/a1x_board_camera.urdf` and `collision/a1x_webcam_camera.urdf` add the
complete rigid payload as **one fixed link** on `gripper_link`, with separate mesh
geometries for the bracket, camera envelope and represented fasteners. Robot
meshes remain in their native metre units; payload meshes use scale 0.001.

Committed URDF mesh paths are relative to their `collision/` directory and resolve
inside a full repository checkout. For a loader that requires absolute paths, run
`python arm_check.py --export-urdf --absolute-mesh-paths` locally. Running the audit
or `--export-urdf` without that flag regenerates portable relative paths. After
extracting a standalone download, set `REPO` in `design.py` to your checkout and
export absolute paths. Vendor arm meshes are not redistributed in the download.

These are kinematic
collision models, with no measured payload inertia. They are not a controller
configuration or a replacement for checking continuous motion trajectories.

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python design.py
.venv/bin/python verify.py
.venv/bin/python arm_check.py
.venv/bin/python render.py
```

`render.py` also reads revision 3 from the repository for the before/after image.
For a single static configuration, in degrees, with 40 mm jaw opening:

```sh
.venv/bin/python arm_check.py --kind webcam --jaw 20 \
  --pose -111.7682 105.9152 -1.9636 9.5397 -66.4478 108.1904
```

The command reports bare-arm and payload surface intersections and exits nonzero
if either is detected. A clear endpoint does not certify the path to it.

## Integration status

These are additional exported URDF variants. The main robot URDF is unchanged.
There is no wrist-camera sensor/optical frame, measured payload inertia, training
observation stream or active collision-avoidance configuration in this change.
The training branch still needs explicit MuJoCo and dataset integration.
