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

The portable output contains combined and individual URDFs, individual URDFs
rooted in the overhead optical frame, a matching actuated MJCF, hashed meshes,
rig configuration and a provenance/limitations manifest. This baseline supports
YAM v1 with linear_4310 grippers. Confirm installed variants before use.

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

1. Save stationary paired captures with `capture --require-joints --output DIR`.
   This rejects missing arms, moving joints and excessive host-arrival skew.
   It is not a hardware synchronization guarantee.
2. Generate `board`, print the PNG at exactly 238 × 170 mm, measure its squares,
   and update the board JSON to the measured dimensions.
3. Run `rectify --capture DIR --camera left --output RECTIFIED`, then
   `detect-board --rgb RECTIFIED/rgb_rectified.png --intrinsics RECTIFIED/intrinsics.json
   --board calibration/board/board.json --output pose.json`. D405 inverse-Brown
   coefficients use librealsense projection semantics, not OpenCV's forward model.
4. For each wrist, collect at least eight diverse fitting poses and three
   held-out poses of a fixed board. Assemble JSON samples with explicit
   `split`, `T_base_gripper` from measured-state URDF FK, and `T_camera_target`
   from detection. `hand-eye --samples samples.json --output candidate.json`
   rejects insufficient pose diversity and reports held-out target consistency.
5. For overhead calibration, use `fixed-camera` with samples containing
   `T_world_target`, `T_camera_target` and `split`. The target's world pose must
   be independently anchored to the arm-base frame. A marker lying on a table
   does not establish that relationship automatically.
6. The original mesh workflow is available as `icp --manual`: provide a paired
   capture, target `--arm left|right|both`, a binary **raw-depth-coordinate mask**,
   and an initial JSON `T_world_camera`. Masking is explicit; no segmentation
   model installation or arm homing is implicit. `validate-icp` evaluates a
   different capture without fitting its transform.
7. `apply-candidate` writes a new rig JSON with status **candidate**, and `build`
   exports both representations. Inspect held-out overlays/point clouds and
   physical geometry before recording acceptance as validated. Candidate or
   nominal data cannot pass `build --require-calibrated`.

The default 10 mm/1 degree consistency thresholds are starting criteria, not a
claim of task accuracy. Absolute alignment, timing, joint-zero accuracy, camera
mount rigidity and measured dynamics require separate physical validation.

## Software verification

After installing the YAM and test extras and cloning the pinned I2RT checkout:

```bash
I2RT_CHECKOUT="$PWD/third_party/i2rt" python -m pytest \
  tests/yam tests/pure tests/hardware_stack/test_viser_viewer.py -q
```

Tests use synthetic observations and transforms. They cover URDF/MJCF FK,
overhead-rooted exports, an actual rendered pixel projection, actuator stepping,
held-out calibration rejection, HTTP transport, stale/replayed-command rejection,
and acquisition failure. Rendering needs an available MuJoCo OpenGL backend.
Model tests skip when the vendor checkout is missing; a skipped test is not a
validated model. Hardware calibration, motor startup and command behavior must
still be checked on the actual supervised rig.
