# Wrist camera calibration data, 2026-09-26

The eye-in-hand sessions behind `../camera_spec_lashup.json`, kept so the
fit can be redone without the arm (`calib_wrist.py solve`, `spec`).

| file | what |
| --- | --- |
| `wrist_K.npz` | intrinsics of the 1920x1080 wrist frame, rotated 180 deg: 6x4 checkerboard, 39 mm squares, 15 views, 0.85 px |
| `near/` | session at 12-24 cm from the tags: 45 poses, 119 tag views, residual 27.8 px |
| `far/` | session at 15-32 cm: 14 poses, 33 tag views, residual 10.5 px |
| `fit_merged.json` | both sessions in one solve, 152 views, 27.8 px: the pose the spec carries |

Each `poses.json` holds, per frame, the measured joints [rad], the
`gripper_link` pose from forward kinematics, and the detected AprilTag corners
(36h11, ids 3-8 on one sheet, 37 mm), so a re-solve needs no images. The
frames themselves (1920x1080 PNG, ~220 MB with the two failed earlier tries)
are on the Hub as the private dataset `marcinwysocki/a1x_wrist_calib`.

Re-solve:

    python3 calib_wrist.py solve hardware/g1_camera_mounts/calibration/near \
        --K hardware/g1_camera_mounts/calibration/wrist_K.npz --tag 3 4 5 6 7 8 --size 0.037

(`solve` re-detects from PNGs when they sit next to `poses.json`; without them
it uses the corners stored there.)
