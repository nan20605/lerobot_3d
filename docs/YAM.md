# YAM live twin, remote acquisition and calibration

The YAM path follows **measured joint feedback**, using radians for the six arm
joints and a normalized gripper opening (`0` closed, `1` open). It reuses the
existing Viser mesh viewer and manual alignment UI. SO101 teleoperation remains
available through its existing entry point.

The Jetson owns hardware. A client reads small joint-state responses separately
from compressed RGB/depth packets, poses both URDFs, and displays camera-local
point clouds. World-frame fusion is enabled only for mounts marked validated.
Image packets retain their accompanying joint state for wrist-camera transforms;
newer joint feedback drives the arm meshes. Acquisition currently uses host poll
and arrival timestamps, not hardware exposure synchronization or interpolation.

## Setup and nominal model

Install the existing package with its YAM extra (`pip install -e '.[yam]'`).
RealSense acquisition also needs `pyrealsense2` on the Jetson. The I2RT driver
is needed only on a host that explicitly owns motors. Importing the YAM modules,
building models, viewing data or running calibration does not open CAN.

```bash
git clone https://github.com/i2rt-robotics/i2rt third_party/i2rt
git -C third_party/i2rt checkout 120c3c81400171174604e503943f8d1ebc891058
lerobot-yam init-rig
lerobot-yam build
lerobot-yam verify-models
lerobot-yam render
```

`init-rig --metadata capture/metadata.json` optionally imports saved factory
RealSense intrinsics, depth units and depth-to-color transforms. It refuses to
overwrite an existing rig. Configure measured base/mount transforms in the rig
JSON and rebuild both formats together.

The portable output contains seven URDFs: a combined station and, for each arm,
world-rooted, arm-base-rooted (`yam_left_base.urdf` / `yam_right_base.urdf`) and
overhead-optical-rooted versions. It also includes matching actuated MJCF, visual
and collision meshes, rig configuration, the vendor license and a manifest with
source URLs and SHA-256 hashes. This baseline supports YAM v1 with linear_4310
grippers. Confirm installed variants before use. Overhead-rooted exports use the
configured transform; this does not establish physical overhead calibration.

The vendor URDF has no collisions. Exports add a convex hull for each visual
mesh, matching MuJoCo's per-mesh convex contact baseline. Concavities, gripper
contact surfaces and contact parameters still need physical validation. Arm effort
limits match the vendor MJCF's 10 Nm cap; the 20 N gripper cap is provisional,
the mimic finger is passive, and vendor velocity limits remain placeholders.

```bash
lerobot-yam audit-models --output reports/model_audit.json
lerobot-yam check-numerics --output reports/numerical_convergence.json
lerobot-yam export-bundle --output output/yam_twin_nominal.zip
```

The audit checks hashes, local asset paths, loadable URDFs, joint limits and
URDF/MJCF mass, centre of mass and full inertia tensors. Export includes only
manifest-listed model assets and refuses to overwrite an existing archive.
The numerical check compares a small synthetic trajectory at 2 ms, 1 ms and
0.5 ms timesteps. It tests integration convergence, not real motor dynamics.
Simulation uses provisional position PD control without the SDK's gravity
compensation; it is not an identified model of the real controller.

The CAD station's 0.61 m base separation, camera mounts and overhead D405 housing
are nominal geometry. The actual overhead sensor is unknown until configured,
so no overhead renderer camera is invented. Camera masses in the vendor CAD are
unit-density placeholders. Actuator effort limits, gripper gains, contacts,
friction, lighting, sensor noise and environment geometry are not calibrated.
Adjacent base/joint1 bearing contacts are explicitly excluded; inter-arm contact
remains enabled. A successful build/FK test does not establish physical fidelity.

## Receive and display data

```bash
# Jetson: cameras only; does not initialize motors.
lerobot-yam serve --live --camera left=LEFT_SERIAL --camera right=RIGHT_SERIAL

# Client terminal 1: SSH encrypts/authenticates transport.
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18765:127.0.0.1:8765 USER@JETSON

# Client terminal 2:
lerobot-yam view --url http://127.0.0.1:18765
lerobot-yam stream-check --url http://127.0.0.1:18765 --seconds 10 --output receive.json
```

Open `http://127.0.0.1:8080`. Select **Left camera** or **Right camera** for a
colored local point cloud before mount calibration. **Robot world** shows arms
only when joint feedback exists. `--nominal-preview` explicitly opts into labeled
example poses for a camera-only recording. Missing/stale feedback is not replaced
with fabricated measurements.

`serve --replay CAPTURE_DIR` replays an `observation.npz` capture, or the original
`metadata.json` + `left/right_rgb.npy` + `left/right_depth_raw.npy` layout.
`serve --sim --render-cameras` provides MuJoCo observations under the same schema,
clearly labeled simulation. Rendered RGB and depth use separate optical frames
and intrinsics. Raw depth is never assumed registered to RGB.

## Supervised arm startup and commands

First verify CAN naming/bitrate, installed gripper type and a clear workspace.
Only one SDK process may own each arm. **I2RT construction itself enables motors
and can calibrate/move the grippers**, even with network motion disabled. It
therefore requires an operator at the rig and an explicit startup acknowledgement:

```bash
lerobot-yam serve --live --arm left=can_left --arm right=can_right \
  --ack-motor-initialization --port 8767
```

This holds the current pose through the SDK and publishes feedback. It does not
home the arms or enable remote commands. The camera service can read these states
with `--joint-source http://127.0.0.1:8767`, allowing the motor and camera Python
environments to remain separate on the same Jetson. Missing joint service is
reported while cameras continue. SDK shutdown releases torque; support the arms.

For a separately supervised motion test, restart the arm service with
`--enable-motion --token-file PATH_TO_RANDOM_TOKEN`. The token must contain at
least 24 random characters. Use `YamRemoteRobot.get_observation()` followed by
`send_action({"position_rad": [...six radians...], "gripper_open": ...})`.
The server rejects nonfinite values, wrong dimensions, out-of-limit joints,
steps above 2 degrees/2% gripper opening, excessive target rates, stale feedback,
repeated commands and replay data. A one-second command timeout latches command
acceptance off and requests a fresh-position hold. Acquisition failure latches
commands off without sending stale feedback. These bounds are not collision
avoidance, emergency-stop hardware or a production safety controller.

## Calibrate against YAM geometry

Convention: `T_A_B` maps B coordinates into A; metres, radians, wxyz quaternions.
Optical axes are +X right, +Y down, +Z forward. MuJoCo renderer cameras are rotated
by Rx(pi), and principal-point offsets account for integer pixel centres.

Generate `lerobot-yam board`, print the PNG at exactly **238 × 170 mm** without
fit-to-page, then measure it. Set `square_m` and `marker_m` in `board.json` to the
measured dimensions, `measurement_status` to `measured`, and `measurement_source`
to the measurement date, instrument and result. Live sample extraction rejects
unmeasured boards. Reject stretched or uneven prints.

For the station workflow, fix one board where all three cameras can see it at
their relevant poses. **Do not move that board during the entire session.** A
`target-id` identifies this physical placement, not merely the printed design.
The left base defines world; the two wrist fits recover the stationary board in
each arm base, establishing the right-base pose and anchoring overhead without
assuming the vendor CAD separation.

Plan fitting and validation splits before collection: at least **8 diverse
fitting + 3 held-out poses per wrist**, with rotations about at least two axes
and translations spanning over 5 cm; at least **3 fitting + 3 held-out overhead
observations**. Fixed overhead/board observations measure repeatability at that
placement, not accuracy throughout the workspace. Pose changes require a
supervised hardware session. Stop both arms before each capture:

```bash
lerobot-yam capture --url http://127.0.0.1:18765 --require-joints \
  --output captures/pose001
lerobot-yam board-sample --capture captures/pose001 --camera left \
  --split train --target-id fixed-board-session-001 --output samples/left/001.json
# Repeat for right and overhead and for distinct held-out captures using --split validation.
lerobot-yam calibration-set --samples samples/left/*.json --output sets/left.json
lerobot-yam calibration-set --samples samples/right/*.json --output sets/right.json
lerobot-yam calibration-set --samples samples/overhead/*.json --output sets/overhead.json
lerobot-yam station-calibrate --left sets/left.json --right sets/right.json \
  --overhead sets/overhead.json --output calibration/station001
lerobot-yam build --rig calibration/station001/rig.candidate.json --output sim/candidate
lerobot-yam overlay --capture captures/heldout001 --camera left \
  --models sim/candidate --output inspection/left001
```

`capture --require-joints` samples at least 0.75 seconds of feedback by default,
requires at least three distinct observations and advancing camera frames, and
rejects velocity above 0.01 rad/s, joint span above 0.003 rad, gripper span above
0.005, or host-arrival skew above 50 ms. `--stationary-seconds` adjusts the window
(minimum 0.25 seconds). This samples stability; it is not continuous monitoring
or hardware exposure synchronization.

`board-sample` rectifies RGB, detects ChArUco, rejects maximum reprojection error
above 1.5 pixels, and attaches measured-state base-relative URDF FK. D405 inverse
Brown coefficients use librealsense projection semantics. Samples retain hashes
of the image, capture, board, model and intrinsics, plus serial and stream profile.
`calibration-set` rejects duplicate images and mixed identities/profiles. Synthetic
captures require `--allow-simulation` and remain marked synthetic throughout.

`station-calibrate` writes a **candidate** rig and held-out board-consistency
report. Validation samples never modify fitting results. Repeat `overlay` for
both wrists and overhead, preferably on additional independent poses. It renders
the robot at measured joint positions, overlays its silhouette on rectified RGB,
and reports optical-Z residuals after registering raw depth through the factory
depth-to-color transform. It writes comparison PNGs and residual arrays. Occluders,
depth holes and segmentation affect these diagnostics; no automatic acceptance
is inferred from small residuals.

After independent physical checks, record acceptance evidence in the rig before
marking mounts, right base, camera intrinsics and installed hardware validated.
`build --require-calibrated` rejects missing sensors/intrinsics, candidate/nominal
transforms and synthetic calibration. Passing this geometry gate does not certify
dynamics or timing fidelity.

The lower-level `rectify`, `detect-board`, `hand-eye` and `fixed-camera` commands
remain available for existing datasets. `fixed-camera` needs independently
anchored `T_world_target`, `T_camera_target` and explicit splits. The original
mesh workflow is available as `icp --manual`: provide a paired capture, target
`--arm left|right|both`, a binary **raw-depth-coordinate mask**, and an initial
JSON `T_world_camera`. `validate-icp` checks a different capture without fitting.
`apply-candidate` writes a new rig with candidate status. No segmentation-model
installation, arm homing or physical motion is implicit in these offline tools.

The default 10 mm/1 degree consistency thresholds are starting criteria, not a
claim of task accuracy. Absolute alignment, timing, joint-zero accuracy, camera
mount rigidity and measured dynamics require separate physical validation.

## Software verification

After installing the YAM and test extras and cloning the pinned I2RT checkout:

```bash
I2RT_CHECKOUT="$PWD/third_party/i2rt" python -m pytest \
  tests/yam tests/pure tests/hardware_stack/test_viser_viewer.py -q
```

Tests use synthetic observations and transforms. They cover portable archive
relocation and checksums, URDF/MJCF FK and inertials, all root-frame exports, actual
rendered pixel projection and registered depth overlays, actuator stepping,
station calibration with held-out rejection, capture stability, HTTP transport,
stale/replayed-command rejection, and acquisition failure. Rendering needs an available MuJoCo OpenGL backend.
Model tests skip when the vendor checkout is missing; a skipped test is not a
validated model. Hardware calibration, motor startup and command behavior must
still be checked on the actual supervised rig.
