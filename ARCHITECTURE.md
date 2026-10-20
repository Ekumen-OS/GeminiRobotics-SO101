# GeminiRobotics-SO101: System Architecture

## 1. System Architecture & Data Flow

GeminiRobotics-SO101 integrates a 6-DOF robotic arm with Google Gemini Robotics Vision-Language-Action (VLA) models (`gemini-robotics-er-2-streaming-preview`), MuJoCo physical simulation, OpenCV ArUco homography calibration, and a FastAPI + Gradio 6.0 Web UI.

```
┌────────────────────────────────────────────────────────┐
│             USB Camera / MuJoCo Simulation             │
└───────────────────────────┬────────────────────────────┘
                            │ Raw RGB Frames
                            ▼
┌────────────────────────────────────────────────────────┐
│           vision/frames.py (Dual-Buffer Engine)        │
│  - get_latest_raw_jpeg() -> Pristine Model Frame       │
│  - /api/stream           -> Annotated Operator Stream  │
└─────────────┬────────────────────────────┬─────────────┘
              │ Pristine JPEG              │ MJPEG Feed
              ▼                            ▼
┌───────────────────────────┐  ┌─────────────────────────┐
│  agent/vla_controller.py  │  │   FastAPI + Gradio UI   │
│  (Gemini Live Streaming)  │  │  (Three.js Sim Viewer)  │
└─────────────┬─────────────┘  └─────────────────────────┘
              │ Tool Calls
              ▼
┌───────────────────────────┐
│     agent/tools.py        │
│  (Pick-Loop Guardrails)   │
└─────────────┬─────────────┘
              │ Commanded Poses
              ▼
┌───────────────────────────┐
│      RobotInterface       │
│  SO101Robot / RealSO101   │
└───────────────────────────┘
```

### Vision Pipeline Decoupling
To eliminate UI-to-model visual feedback leakage, `vision/frames.py` implements a **dual-buffer stream engine**:
1. **Model Vision Channel (`get_latest_raw_jpeg`)**: Supplies pristine, untouched RGB camera frames directly to the Gemini Live session and OpenCV detectors. It is guaranteed to be free of bounding boxes, text overlays, crosshairs, or UI annotations.
2. **Operator Display Channel (`/api/stream`)**: Overlays dynamic ArUco calibration outlines, detected object bounding boxes, and model target focus badges for operator inspection. Served as a single long-lived `multipart/x-mixed-replace` MJPEG stream rendered by an `<img>` tag. Browser quality and frame rate are configured via `WEB_JPEG_QUALITY` (default 75) and `WEB_STREAM_FPS` (default 15); the model path is unaffected.
3. **Capture Lifecycle**: In real mode, a single background capture thread (`vision/capture.py:ensure_camera_capture`) produces frames for all consumers. In sim mode, a dedicated worker thread (`vision/sim_runtime.py:ensure_sim_runtime_started`) renders offscreen frames from MuJoCo.

---

## 2. Layered Dependency Architecture

The codebase strictly enforces a bottom-to-top layered dependency direction:

```
robot  <  simulation  <  vision  <  agent  <  ui
```

- **`robot/`**: Hardware driver (`RealSO101Robot`), Cartesian IK (`CartesianIK`), calibration (`ArucoWorkspaceCalibrator`), camera model (`PinholeCamera`), grasp classifier (`classify_grasp`), and settings. Never imports higher layers.
- **`simulation/`**: MuJoCo simulation wrapper (`SO101Robot`), synthetic calibration (`init_sim_workspace_calibration`), and scene XML assets. Imports `robot` only.
- **`vision/`**: Camera capture, dual frame buffers, target blackboard, and jaw-tag locator. Imports `robot` and `simulation`. Never imports `agent` or `ui`.
- **`agent/`**: Gemini Live WebSocket session (`GeminiLiveSession`), tool declarations and execution (`agent/tools.py`), pick loop (`agent/pick_loop.py`), guardrail state machine (`agent/guardrails.py`), and controller (`VLAController`). Imports `vision`, `simulation`, and `robot`. Never imports `ui`.
- **`ui/`**: FastAPI application and Gradio web interface (`ui/app.py`), Three.js 3D viewer, and UI components. Top-level presentation layer.
- **`tools/`**: Standalone operator CLI utilities (`check_env.py`, `aruco_stl.py`, `calibration_sheet.py`, `setup_wizard.py`). May import layers below it; no package below imports `tools/`.

Enforced locally via `[tool.importlinter]` in `pyproject.toml` with `uv run lint-imports`. Cycle-dodging local imports are strictly prohibited.

---

## 3. System Invariants

1. **Headless `MUJOCO_GL` Ordering**: On headless Linux systems without an active display server, `MUJOCO_GL=egl` is set before `mujoco` is imported. Enforced at the module top of `simulation/__init__.py`, `robot/__init__.py`, `ui/app.py`, and `tools/check_env.py`.
2. **Pristine Model Frames**: The Gemini VLA model always receives untouched raw camera frames; visual overlays exist only for operator inspection in the browser (`vision/frames.py`).
3. **Single Shared Capture Thread**: Real-mode camera frames are produced by exactly one long-lived capture thread (`vision/capture.py`). Every consumer reads from the shared buffer; multiple UI tabs or requests never start competing capture loops.
4. **Explicit Mode Routing**: Operational mode (`"real"` or `"sim"`) is explicitly resolved via `RobotInterface.mode` or `resolve_robot_mode()`, never through duck typing on rendering or hardware methods (`agent/tools.py`, `robot/registry.py`).
5. **Fail-Closed Operation**: Hardware disconnections, invalid calibration files, camera read errors, or IK divergence raise specific typed errors and halt robot motion rather than guessing or executing unsafe fallbacks (`robot/errors.py`, `robot/real_robot.py`, `agent/tools.py`).

---

## 4. Tool Execution Contract

Gemini interacts with the robot via 6 primitives defined in `agent/tools.py`:
- `move_to_spatial_point(point=[y, x], z_height_cm=3.0)`
- `open_gripper()`
- `close_gripper()`
- `move_to_preset(pose_name)`
- `move_to_pose(qpos, duration_sec)`
- `get_joint_positions()`

### Typed Payloads & Telemetry (`agent/contracts.py`)
- **`MoveToSpatialPointResponse`**: Returns execution status, executed joint positions, reached FK coordinates, optional alignment measurements (`perception: AlignmentMeasurement`), applied bias correction (`applied_correction_mm`), and grasp verification.
- **`GripperToolResponse`**: Returns gripper aperture in degrees (`gripper_deg`), state (`"open"`, `"closed"`, `"closed_empty"`, `"closed_unverified"`), and grasp classification telemetry (`grasp: GraspClassification`).
- **`ToolErrorResponse`**: Returns human-readable `error`, classified `error_type`, and honest `robot_moved: bool` telemetry.
- **Error Taxonomy (`ErrorType`)**:
  - `argument`: Invalid parameters or unknown presets.
  - `calibration`: Missing, corrupt, or uncalibrated workspace homography.
  - `perception`: Missing camera frame or unlocated target/jaw.
  - `hardware`: Serial bus write/read errors or interrupted motion (`robot.interrupt()`).
  - `internal`: Unexpected exceptions during execution.
  - `descent_gate`: Blocked vertical descent without verified prior alignment.
  - `transport_gate`: Blocked motion while holding an unverified grasp before straight-up lift.
  - `retry_exhausted`: Escalated abort after repeated failed attempts or consecutive refusals.

---

## 5. Geometry & Kinematics

### Coordinate Frame Conventions
The system coordinates motions across five distinct reference frames:
- **Normalized Model Frame** $[y, x] \in [0, 1000]$: Normalized vision coordinates emitted by Gemini ($y$ down, $x$ right).
- **Pixel Frame** $(u, v)$: Raw camera sensor pixels ($u = x \cdot W / 1000, v = y \cdot H / 1000$).
- **Table Workspace Frame** $(X_{\text{table}}, Y_{\text{table}}, Z_{\text{table}})$: Metric plane in mm defined by the 4 ArUco corner tags, with $+Z$ pointing upward normal to the tabletop.
- **Robot Base Frame** $(X_{\text{base}}, Y_{\text{base}}, Z_{\text{base}})$: Metric Cartesian space in mm centered at the robot base pivot, where $+X$ points forward toward the workspace, $+Y$ points left, and $+Z$ points upward. Related to table coordinates via planar offsets:
  $$X_{\text{base}} = X_{\text{table}} + \text{TABLE\_X\_MM}, \quad Y_{\text{base}} = Y_{\text{table}} + \text{TABLE\_Y\_MM}, \quad Z_{\text{base}} = Z_{\text{table}} + \text{TABLE\_Z\_MM}$$
- **Gripper Tool Frame**: Attached to the end-effector grasp center, offset from the fixed-jaw tag by $\mathbf{d}_{\text{tag}} = (-35.4, 11.0, 40.1)$ mm (`GRIPPER_TAG_LOCAL_OFFSET_MM`).

### Workspace Homography Calibration (`robot/calibration.py`)
Physical workspace grounding maps normalized image coordinates $[y, x] \in [0, 1000]$ to table millimetres via a $3 \times 3$ projective homography matrix $H$:

$$\begin{bmatrix} X_{\text{mm}} \\ Y_{\text{mm}} \\ 1 \end{bmatrix} \sim H \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

- **Real Mode**: $H$ is computed from the 4 physical ArUco corner markers (`DICT_4X4_50`, canonical IDs: Top-Left 1, Top-Right 2, Bottom-Right 3, Bottom-Left 0) and saved to `config/workspace_homography.npz`.
- **Simulation Mode**: Computed analytically from the virtual camera (`workspace_cam`, $fovy=54^\circ$) via `simulation/sim_calibration.py` and saved to `config/workspace_homography_sim.npz`.

### Camera Model & Parallax-Free Jaw Locator (`robot/camera_model.py`, `vision/gripper_locator.py`)
The camera's 3D optical centre $C = [C_x, C_y, C_z]$ in table coordinates is reconstructed analytically from the homography and the nominal horizontal field of view `CAMERA_HFOV_DEG` (`PinholeCamera.from_homography`).
When the gripper hovers at height $h$ above the table, perspective parallax shifts the apparent image position of the jaw ArUco tag (ID 10, 25 mm). The locator corrects parallax in the table plane:

$$xy_{\text{table}} = C_{xy} + (H(u, v) - C_{xy}) \frac{C_z - h}{C_z}$$

The tag-to-grasp-site offset (`GRIPPER_TAG_LOCAL_OFFSET_MM`, default `(-35.4, 11.0, 40.1)`) rotated by the forward kinematics orientation is then subtracted to yield the true physical grasp-site position in robot millimetres.

### Table Offsets (`TABLE_X_MM`, `TABLE_Y_MM`, `TABLE_Z_MM`)
Configured in `robot/settings.py` for physical hardware:
- `TABLE_X_MM`, `TABLE_Y_MM`: Planar offset between robot base origin and table mat origin.
- `TABLE_Z_MM`: Table surface height in robot base coordinates (calibrated as $Z_{\text{base}} = Z_{\text{table}} + \text{TABLE\_Z\_MM}$). In sim, all offsets are 0.0.

### Inverse Kinematics & Null-Space Posture Regularization (`robot/cartesian_ik.py`)
The SO-101 has 5 articulated arm joints (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`); the gripper jaw actuator is decoupled from spatial positioning. Numerical IK solves 3D position tracking ($3 \times 5$ Jacobian) using Levenberg-Marquardt Damped Least Squares (DLS):

$$\Delta \mathbf{q} = J^T (J J^T + \lambda^2 I)^{-1} \mathbf{e}$$

The remaining degrees of freedom are resolved through null-space projection to bias postures smoothly toward an ergonomic resting configuration (`nominal_qpos`):

$$\Delta \mathbf{q}_{\text{null}} = (I - J^\dagger J) (\mathbf{q}_{\text{nominal}} - \mathbf{q})$$

### S-Curve Trajectory Interpolation
Joint motions smoothly interpolate between poses using a cosine S-curve profile over duration $T$ with time step $dt$ (`simulation/sim_robot.py`, `robot/real_robot.py`):

$$\alpha(t) = \frac{1}{2} (1 - \cos(\pi t)), \quad t = \frac{\text{step}}{\text{total\_steps}} \in [0, 1]$$
$$\mathbf{q}(t) = (1 - \alpha(t)) \mathbf{q}_{\text{start}} + \alpha(t) \mathbf{q}_{\text{target}}$$

This guarantees zero velocity and jerk at trajectory boundaries ($\dot{\alpha}(0) = \dot{\alpha}(1) = 0$).

---

## 6. Pick-Loop Orchestration & Guardrails

The manipulation loop enforces a structured guardrail state machine (`agent/guardrails.py`, `agent/pick_loop.py`, `agent/tools.py`):

```
       [APPROACH] ──(Aligned Descent)──► [DESCENT] ──(close_gripper)──► [HELD_UNVERIFIED]
           ▲                                                                   │
           │                                                            (Straight Lift)
           │                                                                   ▼
       (Retry) ◄──────────────────── (Grasp Failed) ──────────────────── [LIFT_VERIFY]
                                                                               │
                                                                        (Sensor Confirmed)
                                                                               ▼
                                                                          [TRANSPORT]
```

1. **Target Lock**: On the initial approach hover, the target object is locked (`TargetLock`) using the pre-move frame. Identity is maintained within 40 mm across subsequent moves.
2. **Perception Metrics**: Measures planar offset between jaw and target centre in robot mm (`dx_mm`, `dy_mm`, `distance_mm`, `is_aligned` within `ALIGNMENT_TOLERANCE_MM`).
3. **Bias Corrector**: Residual misalignment observed at hover updates a per-target millimetre correction vector ($c \leftarrow c + r$, $|r| \le 30$ mm). Subsequent hovers and descents apply this correction.
4. **Descent Gate**: Vertical descent ($z < 3.0$ cm) is rejected (`error_type="descent_gate"`) unless preceded by an aligned hover at the same target lock. In sim, `DESCENT_GATE=block`; on hardware, defaults to `warn`.
5. **Transport Gate**: In `HELD_UNVERIFIED` and `CLOSED_EMPTY` phases, all motions except straight-up lift ($z \ge 3$ cm at current XY) are blocked (`error_type="transport_gate"`).
6. **Lift Verification**: After lifting, the physical servo aperture re-read decides the grasp state; vision can only veto (`vision_vote`) if color segmentation detects the object still resting on the table footprint.
7. **Retry & Escalation**: Up to 3 grasp attempts are permitted per target. After 3 missed grasps, 3 consecutive refusals, or 4 hovers without descent, execution halts with `error_type="retry_exhausted"`, requiring operator intervention.

---

## 7. Verification & Test Suite

The test suite in `tests/` is organized strictly by subject:
- `tests/robot/`: Kinematics, calibration, settings, and hardware driver contracts.
- `tests/simulation/`: MuJoCo scene loading, synthetic calibration, and headless EGL rendering.
- `tests/vision/`: Camera selection, dual-buffer integrity, and detector segmentation.
- `tests/agent/`: Gemini Live session mocking, tool execution, and pick-loop state machine.
- `tests/ui/`: FastAPI endpoints, Gradio component event wiring, and viewer assets.
- `tests/tools/`: Preflight diagnostics, STL generation, and workspace setup wizards.
- `tests/e2e/`: End-to-end spatial grounding and guardrail integration tests.
- `tests/harness/sim_pick.py`: Headless scripted sim pick-and-place harness (`SimPickHarness`, `run_scenario`) testing multi-placement runs against ground-truth MuJoCo physics.

### Testing Commands
```bash
# Standard test suite (fast unit and integration tests)
uv run pytest tests/ -v

# Physics-heavy end-to-end scenarios (excluded by default; marked with `slow`)
uv run pytest -m slow -s

# Code coverage across core modules (enforced at >= 78% by pre-push hook)
uv run pytest tests/ -q --cov=robot --cov=vision --cov=agent --cov-fail-under=78
```

---

## 8. Known Limitations

- **Sim Pick Lock Churn**: The end-to-end live simulation pick currently exhibits a `no_lock` $\rightarrow$ `retry_exhausted` failure mode under active investigation.
- **GLFW Desktop Rendering**: Desktop rendering with native GLFW windowing is untested; headless EGL offscreen rendering is the primary validated path.
- **Hardware Convergence Bounds**: Real-arm servo settling tolerances and contact thresholds are empirically calibrated for standard STS3215 servos and have not been validated across varied mechanical wear states.
