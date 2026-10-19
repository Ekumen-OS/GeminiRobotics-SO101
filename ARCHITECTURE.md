# GeminiRobotics-SO101: System Architecture & Testing Specification

## 1. System Architecture & Data Flow

The GeminiRobotics-SO101 platform integrates a 6-DOF robotic arm with Google Gemini Robotics Vision-Language-Action (VLA) models (`gemini-robotics-er-2-preview` / `gemini-robotics-er-2-streaming-preview`), MuJoCo physical simulation, OpenCV ArUco homography calibration, and a FastAPI + Gradio 6.0 Web UI.

```
                  ┌────────────────────────────────────────┐
                  │          Camera / Sim Sensor           │
                  └──────────────────┬─────────────────────┘
                                     │ (Raw RGB Frames)
                                     ▼
                  ┌────────────────────────────────────────┐
                  │    vision/frames.py: Dual Buffer Engine│
                  │  (get_latest_raw_jpeg vs UI annotated) │
                  └─────────┬────────────────────┬─────────┘
                            │                    │
        [Model Vision Path] │                    │ [UI Display Path]
                            ▼                    ▼
             ┌─────────────────────────┐  ┌─────────────────────────┐
             │   VLA Perception Module │  │ Gradio 6.0 Web Viewer   │
             │ (robot/perception.py)   │  │ (3D Three.js + Overlays)│
             └──────────┬──────────────┘  └─────────────────────────┘
                        │
                        ▼
             ┌────────────────────────────────────────┐
             │       agent/vla_controller.py          │
             │   - Gemini Live Streaming Loop         │
             │   - Programmatic Perception Injection  │
             │   - Descent & Transport Gates          │
             │   - Automated Retry State Machine      │
             └──────────┬─────────────────────────────┘
                        │
                        ▼
             ┌────────────────────────────────────────┐
             │       agent/tools.py Tool Executor     │
             │   - Spatial & Pose Command Execution   │
             │   - Measured Perceptual Feedback       │
             └──────────┬─────────────────────────────┘
                        │
                        ▼
             ┌────────────────────────────────────────┐
             │   robot/calibration.py & cartesian_ik  │
             │   - Normalized [y, x] -> Homography    │
             │   - Base Cartesian [X, Y, Z]           │
             │   - 6-DOF Inverse Kinematics           │
             └────────────────────────────────────────┘
```

### Vision Pipeline Decoupling

To eliminate UI-to-model visual feedback leakage, `vision/frames.py` implements a **Dual-Buffer Stream Engine**:

1. **Model Vision Channel (`get_latest_raw_jpeg`)**: Supplies pristine, untouched RGB camera frames to the Gemini Live session and OpenCV perception detectors. It is guaranteed to be free of bounding boxes, text overlays, crosshairs, or UI annotations.
2. **Operator Display Channel (`/api/stream`)**: Overlays dynamic ArUco calibration outlines, detected object bounding boxes, and model target focus badges for human operator inspection. Served as a long-lived `multipart/x-mixed-replace` MJPEG response rendered by a plain `<img>`, so delivery needs throughput, not a per-frame round trip (per-frame Gradio image delivery froze on any link whose round-trip time exceeded the frame interval, e.g. Tailscale/WAN). Browser quality and rate are `WEB_JPEG_QUALITY` (default 75) and `WEB_STREAM_FPS` (default 15); the model path is unaffected.
3. **Shared capture**: camera capture is one background thread (`ensure_camera_capture` / `switch_camera_capture` in `vision/capture.py`), independent of any browser session. Every tab reads the same frame buffers, so opening a second tab joins the running capture instead of replacing it.

---

## 2. Core Feature Inventory (F1 – F14)

| # | Feature | Subsystems | Description |
| --- | --- | --- | --- |
| **F1** | ArUco Marker ID & Corner Harmonization | `robot/calibration.py`, `config/workspace_calibration.json`, `simulation/assets/scene.xml` | Canonical fiducial corner IDs (`1: Top-Left`, `2: Top-Right`, `3: Bottom-Right`, `0: Bottom-Left`) harmonized across physical config, simulation XML, and prompts. |
| **F2** | Homography & Lateral Axis Parity | `config/workspace_homography.npz`, `config/workspace_homography_sim.npz`, `robot/calibration.py` | Lateral $Y$-axis parity guaranteed between real and simulation frames. Target `[760, 484]` maps to $X \approx 0.197$ m, $Y \approx +0.145$ m, yielding inward shoulder pan ($\approx -42^\circ$) inside camera FOV. |
| **F3** | Calibrator Mode Context Routing | `agent/tools.py`, `vision/calibrator.py`, `robot/registry.py` | `execute_tool_call` receives the controller's `mode` explicitly and otherwise reads `robot.mode` (`resolve_robot_mode`); `image_point_to_xyz` then loads the matching homography. No duck-typing on rendering methods. |
| **F4** | Reachability & FOV Kinematic Grounding | `robot/cartesian_ik.py`, `robot/calibration.py` | All normalized coordinates $[y, x] \in [0, 1000]$ within the calibrated mat map to collision-free Cartesian poses ($r \le 391$ mm) contained within the camera FOV. |
| **F5** | Programmatic Perception Metrics, Camera Model & Jaw Locator | `vision/gripper_locator.py`, `vision/detector.py`, `robot/camera_model.py`, `robot/calibration.py`, `robot/perception.py`, `robot/cartesian_ik.py` | Everything is compared **in the table plane, in robot millimetres**. All operational heights (`z_height_cm`, the lift, safe travel, jaw-tag parallax) are **above the table**, with the robot base frame offset by `TABLE_Z_MM` (real only, sim is always 0.0). The target is a *footprint centre* (`locate_target_footprint`: the bottom edge of the blob through the homography, plus half its width along the edge normal; 0.6 mm mean / 1.1 mm max at yaw 0 in sim). The jaw is located from the ArUco **jaw tag** (ID 10, 25 mm): its pixel is intersected with the plane at the tag height from forward kinematics using a pinhole camera recovered from the homography plus a nominal `CAMERA_HFOV_DEG` (`PinholeCamera.from_homography`), the parallax being removed as $xy = C_{xy} + (H(u,v) - C_{xy})\,(C_z - h)/C_z$ (identity at $h = 0$, so target and jaw share one geometry), then the FK-rotated tag-to-site offset (`GRIPPER_TAG_LOCAL_OFFSET_MM`) is subtracted. `gripper_source` is `"aruco"` (or `"kinematic"` when the tag is not seen: FK is self-consistent with the IK and cannot see calibration or encoder bias, so `is_aligned` is then `null`). The homography, its inverse, the reprojection error and the camera are one immutable `GeometrySnapshot` replaced in a single assignment (`ArucoWorkspaceCalibrator.geometry_snapshot()`); the derived camera centre is logged once at INFO (`[Calibration] [CAMERA] centre_mm=[x, y, z] hfov=...`). |
| **F6** | Tool Execution Feedback, Target Lock & mm Bias Corrector | `agent/tools.py`, `agent/guardrails.py`, `agent/pick_loop.py`, `agent/motion.py` | The object being picked is a `TargetLock` created once from the frame captured *before* the move; identity is matched by request point **or** by the detected footprint (within 40 mm) on every later request, never re-refined. Hints name the lock centre. `move_to_spatial_point` hovers and descends at `lock centre + c_mm`; after a measured hover the residual $r$ = lock centre − jaw updates the per-lock correction $c \leftarrow c + r$ (jaw-tag source only, $|r| \le 30$ mm; reported as `applied_correction_mm`), so a constant bias converges in one update. The first move from a low pose (site < 3 cm above table) to a target > 30 mm away is split into lift / travel (≥ 6 cm) / descend. Measurements wait for a frame captured *after* the arm converged (`vision/frames.get_raw_jpeg_after`; sim `move_to_pose` steps until joints converge, `RealSO101Robot` waits for arm joints to settle within `SETTLE_DELTA_DEG` across `SETTLE_WINDOW_S`, compensates steady-state sag up to `MAX_SAG_COMP_DEG` in $\le 2$ rounds with a `CONTACT_GUARD_MM` contact guard, and logs `[SERVO]` tag; warns on timeout if error $\ge$ `CONVERGENCE_TOL_DEG`). `close_gripper` returns the measured grasp classification (`closed`, `closed_empty`, `closed_unverified`). |
| **F7** | Anti-Hallucination Descent Gate | `agent/tools.py`, `agent/guardrails.py` | A descent ($z < 3.0$ cm) needs a *measured hover at the same lock* that is still valid (current FK XY within 10 mm of the commanded hover) and `is_aligned` (jaw within `ALIGNMENT_TOLERANCE_MM`, default 4 mm, of the lock centre). Policy `DESCENT_GATE` is `block` in sim, `warn` in real. A refusal is `error_type="descent_gate"` (misaligned) or `"perception"` (with four honest `reason` values: `no_lock`, `no_hover`, `other_lock`, `arm_moved`), `robot_moved=False`, and its hint names the exact hover call `move_to_spatial_point(point=[y, x], z_height_cm=3.0)` that will succeed; the controller applies the correction, the model must not. With the jaw tag unseen (`is_aligned=null`) `warn` descends with a warning and `block` refuses and escalates immediately (no loop). |
| **F8** | Pick-Loop Phases, Lift Policy, Transport Gate & Escalation | `agent/guardrails.py`, `agent/tools.py`, `agent/vla_controller.py` | Explicit phases: `APPROACH` → (`close_gripper`) → `HELD_UNVERIFIED` / `CLOSED_EMPTY` → (lift) → `TRANSPORT`. In the held phases only the **lift** and `open_gripper` are allowed: the first upward move (z ≥ 3 cm, raised to ≥ 5 cm) is executed straight up at the current FK XY whatever point the model sent; anything else is `error_type="transport_gate"` whose hint names the lift call and `open_gripper()`. After the lift the **sensor decides, vision can only veto** (`decide_grasp`): post-lift aperture empty → `missed`; object blob of the lock colour still overlapping the lock-time bbox (IoU ≥ 0.5) → `missed (object_still_on_table)`; camera unusable (< 3 of 4 mat markers) → `unverified` and a re-verify, no attempt consumed; else `confirmed`. `TRANSPORT` moves to raw destinations (no refinement, lock, gate or corrector) and re-reads the aperture after every move (a drop → `CLOSED_EMPTY`). `retry_exhausted` after 3 missed/unverified grasps per lock, 3 consecutive refusals, 3 descent refusals without a successful descent, or 4 hovers without a descent; only `open_gripper` works while escalated. Guardrail state is mutated in place under an `epoch` (`begin_instruction`, `on_interrupt`) so a handler thread outliving an interrupt cannot write to the next instruction. |
| **F9** | Clean Diagnostic Logging & UI Separation | `agent/vla_controller.py`, `agent/tools.py`, `ui/components/chat.py` | Standardized ISO-8601 millisecond logs routed to `sys.stdout`/`sys.stderr`. Raw JSON payloads, IK solver iterations, and tracebacks are excluded from Chatbot viewport. |
| **F10**| Comprehensive 4-Tier E2E Test Suite | `tests/e2e/test_e2e_spatial_grounding.py`, `tests/e2e/test_e2e_perception_guardrails.py` | 4-tier opaque-box test suite verifying all features across isolation, boundaries, cross-feature interactions, and full real-world scenarios. |
| **F11**| Continuous Integration & Regression Shield | `tests/` (all modules) | Deterministic test harness with zero external network dependencies and 100% pass rate. |
| **F12**| Adversarial Hardening | `tests/robot/test_cartesian_ik.py`, `tests/vision/test_detector.py`, `tests/ui/test_ui.py` | Stress testing against extreme coordinate noise, stream concurrency hammers, degenerate corner inputs, and servo angle clamping. |
| **F13**| Physical Grasp Sensor & Sensor-First Lift Verdict | `robot/grasp.py`, `agent/guardrails.py`, `agent/pick_loop.py`, `robot/real_robot.py`, `simulation/sim_robot.py`, `agent/tools.py` | Bus register and physics-driven aperture and load telemetry (`classify_grasp`) decide whether the jaws closed on air or stalled on something; the post-lift re-read plus the lock-based vision veto (`vision_vote`) feed `decide_grasp`. Sensor or vision unavailability yields fail-closed unverified states that block transport. The simulated gripper has box finger pads (40 mm gap, 1.5 N·m torque cap) so a 30 mm cube is actually lifted. |
| **F14**| Fail-Closed Error Taxonomy & Visual Verification Safety | `robot/errors.py`, `agent/live_session.py`, `robot/real_robot.py`, `agent/tools.py`, `vision/frames.py` | Hierarchical error taxonomy (`RoboticsError`, `HardwareError`, `CalibrationUnavailable`, `PerceptionUnavailable`, `ToolArgumentError`, `SessionError`). Fails closed on driver read/write, calibration corruption, and missing camera frames (no synthetic placeholder fallback in real mode). Tool error responses communicate typed `error_type` and honest `robot_moved` telemetry. |

---

## 3. Subsystem Interface Contracts

Any agent modifying the codebase MUST preserve the following schemas and functional contracts:

### 3.1. Calibration Engine (`robot/calibration.py`) ↔ Tool Adapter (`agent/tools.py`)

- **Table & Workspace Offsets (`TABLE_X_MM`, `TABLE_Y_MM`, `TABLE_Z_MM`)**: All operational coordinates and heights (`z_height_cm`, lift elevation, safe travel, jaw-tag parallax) are defined relative to the table surface and mat origin. On physical hardware:
  - Table surface is offset from robot base $z=0$ by `TABLE_Z_MM` (calibrated via: hover at 3 cm, measure fingertip height above table with ruler, `TABLE_Z_MM = FK z − measured height`; e.g., $-24.0$ mm).
  - Physical mounting offset between robot base origin and table mat origin is configured via `TABLE_X_MM` and `TABLE_Y_MM`.
  - In simulation, all table offsets are strictly 0.0.
- **FK Reached & Actual Position Logging**: After every spatial motion (`_approach_move`, `_transport_move`, `_lift`), the tool adapter queries forward kinematics (`_site_xyz`) and emits standardized ISO-8601 `[EXEC]` logs containing:
  - Reached FK XYZ in robot base coordinates (meters).
  - Table-relative coordinates (table $Z$ in cm, table $XY$ in mm).
  - Inverse projection through the homography back to normalized Gemini coordinates $[y, x]$ and camera pixel $(u, v)$.
  - Kinematic tracking error ($dX, dY, dZ$ in mm).
  - Visual jaw tag detection report (ArUco ID 10 pixel and table $XY$, or occluded/absent notification).
- **`ArucoWorkspaceCalibrator.gemini_to_robot_xyz(point_1000, img_shape=(480, 640), z_m=0.03) -> np.ndarray`**:
  - **Inputs:** `point_1000` as normalized $[y, x] \in [0, 1000]$, image dimensions $(H, W)$, target height $z$ in meters.
  - **Returns:** NumPy array `[X, Y, Z]` in meters in robot base frame.
  - **Invariants:** Lateral $Y$ coordinate must have identical sign conventions in simulation and physical hardware.
- **`ArucoWorkspaceCalibrator.robot_mm_to_pixel(x_mm, y_mm) -> tuple[float, float] | None`**:
  - **Inputs:** Physical robot frame millimeters $(X, Y)$.
  - **Returns:** Sub-pixel image coordinate $(p_x, p_y)$ or `None` if uncalibrated/singular.

### 3.2. Perception Engine (`robot/perception.py`) ↔ Tools & VLA Controller

Payload definitions are formalized as `TypedDict` models in [`robot/contracts.py`](robot/contracts.py):

- **`compute_alignment_mm(target_xy_mm, jaw_xy_mm, source, tolerance_mm=None) -> AlignmentMeasurement`**:
  - **Typed contract:** [`AlignmentMeasurement`](robot/contracts.py): `gripper_xy_mm`, `target_xy_mm`, `measured_offset` (`dx_mm`, `dy_mm`, `distance_mm`; target minus jaw), `is_aligned`, `alignment_tolerance_mm`, and `gripper_source`.
  - **Invariant:** `is_aligned` is `true`/`false` only for the jaw-tag source (Euclidean distance against the tolerance); the kinematic fallback and an unlocated jaw give `null`.
- **`vision_vote(frame_rgb, lock, calibrator)`** ([`agent/pick_loop.py`](agent/pick_loop.py)): `"on_table"`, `"not_seen_on_table"` or `"unavailable"`; fed to **`decide_grasp(aperture_after_lift, vote)`** ([`agent/guardrails.py`](agent/guardrails.py)). `verify_grasp_elevation` no longer exists.
- **`locate_jaw_xy_mm(frame_rgb, snapshot, qpos_rad, tag_offset_mm=None) -> JawLocation | None`** ([`vision/gripper_locator.py`](vision/gripper_locator.py)): grasp-site XY in robot mm with parallax removed, or `None` (tag unseen, no calibration, no camera model).
- **`locate_target_footprint(obj, homography, image_shape) -> {centre_xy_mm, centre_1000, width_mm, bbox_px}`** ([`vision/detector.py`](vision/detector.py)); `refine_point_to_nearest_object` returns the footprint centre (default radius 80 units, optional colour filter) and keeps `bottom_contact_1000` on the object dictionary.
  - **Fail-Closed Invariant:** When perception fails to locate the target or the jaw and no fallback is valid, tools raise `PerceptionUnavailable`; a camera frame that never arrives after a motion makes only the *measurement* unavailable (the response carries a warning, no `perception`), not the motion a failure.

### 3.3. Tool Registry & Executor (`agent/tools.py`) ↔ Gemini FunctionResponse

Tools are defined via a unified `TOOL_REGISTRY` map of `ToolSpec` dataclasses. Each tool specification
co-locates:
- **`name`**, **`description`**, **`parameters`**: the Google GenAI function declaration fields.
- **`handler`**: Execution callback `handler(ctx: ToolContext, args: dict[str, Any]) -> dict`.
- **`blocking`**: Boolean indicating whether execution requires model reasoning pauses or physical movement.
- **`prompt_line`**: Formatted prompt documentation string for system instructions.

`ROBOT_TOOL_DECLARATIONS` and `_BLOCKING_TOOLS` are programmatically derived from `TOOL_REGISTRY`.
The system prompt (`agent/prompts/vla_system.md`) uses `{tool_list}` which is dynamically generated
at runtime via `render_prompt_tool_list()`, guaranteeing synchronization between registered tools
and the model's behavioral prompt.

Payload definitions are formalized as `TypedDict` models in [`agent/contracts.py`](agent/contracts.py):

- **`move_to_spatial_point` Return Payload**:
  - **Typed contract:** [`MoveToSpatialPointResponse`](agent/contracts.py) (incorporates optional [`AlignmentMeasurement`](robot/contracts.py) under `perception`, `applied_correction_mm`, `target_xy_mm`, `gripper_source`, `grasp_verification` (`verdict`, `reason`, `vision`, `transport_allowed`), `guardrail_warning` and, for a lift, `lift_note`).
- **`close_gripper` / `open_gripper` Return Payload**:
  - **Typed contract:** [`GripperToolResponse`](agent/contracts.py).
  - Incorporates measured `gripper_state` (`"open"`, `"closed"`, `"closed_empty"`, or `"closed_unverified"`), measured `gripper_deg`, context-sensitive `visual_hint`, and optional [`GraspClassification`](robot/contracts.py) under `grasp`:
    - `grasped: bool | None`: `True` if stalled on object, `False` if closed on air / below load, `None` if sensor unavailable.
    - `aperture_deg: float`: measured aperture from bus or physics.
    - `closed_on_air_threshold_deg: float`: threshold (calibrated empty aperture `GRIPPER_EMPTY_APERTURE_DEG`, else `GRIPPER_CLOSE_DEG`, plus `GRASP_EMPTY_MARGIN_DEG`).
    - `load: float | None`: present load register or actuator force.
    - `reason: Literal["closed_on_air", "aperture", "no_load", "sensor_unavailable"]`.
- **Tool Error Return Payload**:
  - **Typed contract:** [`ToolErrorResponse`](agent/contracts.py).
  - Schema:
    - `error: str`: Human-readable description of the failure cause.
    - `error_type: Literal["argument", "calibration", "perception", "hardware", "internal", "descent_gate", "transport_gate", "retry_exhausted"]`: Machine-readable classification of the failure subsystem or guardrail trigger.
    - `robot_moved: bool`: Honest indication of whether physical robot actuators moved prior to or during the failure.
  - When hardware failures occur (`error_type="hardware"`), the executor immediately halts physical motion (`robot.interrupt()`) before returning the error payload. Guardrail gate rejections (`descent_gate`, `transport_gate`) guarantee `robot_moved=False`.

### 3.4. Failure Policy: Block and Inform

When an operational, perceptual, or hardware boundary fails, the system strictly **fails closed**:
- **Hardware Failures**: Physical motor bus write or read failures raise `HardwareError`, halt the
  arm immediately via `robot.interrupt()`, and report `robot_moved` honestly without cached position
  fallbacks. An interrupt during motion surfaces as `error_type="hardware"`, `robot_moved=true`,
  never as success.
- **Calibration Integrity**: Uncalibrated workspaces or corrupted calibration files immediately
  raise `CalibrationUnavailable`, rejecting spatial motions rather than executing open-loop guesses.
- **Perception Failures**: Camera frame decoding errors or detection failures raise
  `PerceptionUnavailable` rather than falling back to synthetic successes or comforting defaults.
- **Descent Gate Rejection (`error_type="descent_gate"` / `"perception"`)**: A vertical descent
  ($z < 3.0$ cm) without a measured, still-valid, aligned hover at the same lock is blocked under
  `DESCENT_GATE=block` (warned under `warn`) with `robot_moved=False`. Reasons: `descent_gate`
  (measured misalignment in mm), or `perception` with four honest rejection reasons: `no_lock` (no
  active target lock), `no_hover` (no measured hover at this target), `other_lock` (last measured
  hover was at another target), or `arm_moved` (arm moved since hover was measured; FK XY > 10 mm).
  Every refusal names the exact next call with the lock centre.
- **Transport Gate Rejection (`error_type="transport_gate"`)**: In `HELD_UNVERIFIED` / `CLOSED_EMPTY`
  every motion except the lift (straight up, z ≥ 3 cm) is rejected with `robot_moved=False`; the hint
  names the lift call and `open_gripper()`.
- **Retry Exhaustion Escalation (`error_type="retry_exhausted"`)**: 3 descent refusals without a
  successful descent, 4 hovers without a descent, 3 missed or unverified grasps at one lock, 3
  consecutive refusals without a successful motion, or an unobservable alignment under
  `DESCENT_GATE=block` halt arm motion and tell Gemini to notify the human operator.
  `open_gripper` stays available.
- **Vision Stream Integrity**: Real-mode camera streams NEVER serve simulated or placeholder
  frames when physical cameras disconnect or buffers empty (`get_latest_raw_jpeg()` returns `None`).
- **Model Grounding**: Failure payloads are returned to Gemini with typed `error_type` and
  `robot_moved` flags, enabling the model to reason about errors or halt gracefully. `SessionError`
  is a controller-level exception for WebSocket session lifecycle, rather than a tool return type.

---

## 4. E2E Test Infrastructure & Verification Methodology

All automated tests in `tests/` adhere to an **opaque-box, requirement-driven verification philosophy**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TIER 4: REAL-WORLD APPLICATION SCENARIOS               │
│  - Complete End-to-End Pick-and-Place with Alignment & Grasp Confirmation    │
│  - Multi-Turn Retry State Machine under Simulated Physical Grasp Slips      │
│  - Retry Exhaustion, Arm Halting & Conversational Escalation to User        │
│  - Full Workspace 100-Point Reachability & Spatial Grounding Execution      │
├─────────────────────────────────────────────────────────────────────────────┤
│                    TIER 3: CROSS-FEATURE COMBINATIONS & WORKFLOWS           │
│  - Homography Projection ↔ Inverse Kinematics ↔ Camera FOV Consistency      │
│  - Programmatic Perception Metric Extraction ↔ Tool Feedback Enrichment     │
│  - Descent Gate Rejection ↔ Corrective Offset Hint ↔ Re-alignment Move      │
│  - Post-Lift Sensor Verdict + Vision Veto ↔ Transport Guardrail Blocking    │
│  - UI Event Blocking ↔ Model Stream Execution ↔ Guaranteed Interactive Reset│
├─────────────────────────────────────────────────────────────────────────────┤
│                   TIER 2: BOUNDARY VALUES, CORNER CASES & RESILIENCE        │
│  - Extreme Normalized Coordinates ([0,0], [1000,1000], Negative, >1000)     │
│  - Workspace Mat Physical Boundary Edges & Out-of-Reach Kinematic Limits    │
│  - Alignment Tolerance Boundaries (distance_mm vs ALIGNMENT_TOLERANCE_MM)   │
│  - Partial Marker Occlusion, Missing Calibration Config & Corrupt NPZ Files │
│  - Coordinate Parser Fuzzing with Malformed LLM Speech & Mixed Delta Tokens │
├─────────────────────────────────────────────────────────────────────────────┤
│                   TIER 1: ISOLATED FEATURE COVERAGE (UNIT MECHANICS)        │
│  - ArUco Marker ID & Corner Definitions (1:TL, 2:TR, 3:BR, 0:BL)            │
│  - Homography Forward (Gemini→Base XYZ) & Inverse (Base XYZ→Pixel) Mapping  │
│  - Mode-Aware Calibrator Context Routing (Sim vs Real)                      │
│  - Red Cube [760, 484] Canonical Grounding & Shoulder Pan Sign Verification │
│  - Alignment Metrics in mm (dx_mm, dy_mm, distance_mm, is_aligned)          │
│  - Lift Verdict Table (aperture × vision vote → confirmed/missed/unverified)│
│  - Tool Response Schemas & Visual Hint Injection                            │
│  - ISO-8601 Millisecond Timestamp Formatting & Stdout/Stderr Routing        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Verification Invariants

1. **Behavior Over Implementation:** Tests evaluate observable behaviors, return schemas, error codes, and kinematic constraints rather than private attributes.
2. **Authoritative Expected Outputs:** Assertions are derived from physical geometry invariants, MuJoCo model specifications (`simulation/assets/scene.xml`), and formal JSON configs.
3. **Strict Offline Determinism:** All Gemini Live API WebSocket sessions and serial hardware buses are mocked via deterministic fixtures (`AsyncMock`, `RobotInterface` mock classes) to ensure deterministic execution, zero network flakiness, and zero token costs.
4. **Adversarial Resilience:** Fuzzing probes edge coordinates, corrupt calibration files, and concurrent stream operations to ensure graceful error recovery without process termination.

---

## 5. Verification Commands

When implementing changes or verifying project health, execute:

```bash
# 0. Sync dependencies and install pre-commit hooks (one-time setup)
uv sync --all-extras
uv run pre-commit install

# 1. Run complete automated test suite (slow physics runs are excluded by default)
uv run pytest tests/ -v
#    ...and the slow 10-placement pick-and-place scenarios (prints the metrics table)
uv run pytest -m slow -s

# 2. Run focused spatial grounding E2E tests
uv run pytest tests/e2e/test_e2e_spatial_grounding.py -v

# 3. Run focused perception guardrails E2E tests
uv run pytest tests/e2e/test_e2e_perception_guardrails.py -v

# 4. Run static lint and code formatting check
uv run ruff check .
uv run ruff format --check .

# 5. Run static type checking (scope is defined in [tool.mypy] `files` in pyproject.toml)
uv run mypy

# 6. Run test suite with core coverage gate (fails under 78%)
uv run pytest tests/ -q --cov=robot --cov=vision --cov=agent --cov-report=term-missing
```

---

## 6. Repository Layout & File Responsibilities

```
GeminiRobotics-SO101/
├── AGENTS.md                         # Operational rules and agent instructions
├── ARCHITECTURE.md                   # System architecture, interface contracts & test spec
├── README.md                         # Human-facing project guide, setup & hardware manual
├── pyproject.toml                    # Dependencies, uv configuration & ruff settings
├── config/
│   ├── workspace_calibration.json    # Canonical ArUco corner definitions & dimensions
│   ├── workspace_homography.npz      # Calibrated real physical homography matrix (local artifact, gitignored)
│   └── workspace_homography_sim.npz  # Calibrated simulation synthetic homography matrix (local artifact, gitignored)
├── robot/
│   ├── settings.py                   # Central Settings dataclass and environment configuration
│   ├── logger.py                     # ISO-8601 millisecond logger and stream router
│   ├── contracts.py                  # Subsystem TypedDict contracts for perception payloads
│   ├── registry.py                   # Thread-safe global active-robot registry
│   ├── calibration.py                # ArUco detection, homography & coordinate transformations
│   ├── camera_model.py               # Pinhole camera from the homography + nominal intrinsics
│   ├── perception.py                 # Alignment in robot mm (and the legacy image-space helper)
│   ├── cartesian_ik.py               # Numerical damped least-squares inverse kinematics
│   ├── real_robot.py                 # Physical SO-101 Feetech servo driver (via LeRobot)
│   ├── interface.py                  # Abstract base RobotInterface contract
│   ├── constants.py                  # Joint limits, home poses, and serial port defaults
│   └── theory.md                     # Mathematical derivations for kinematics and homography
├── simulation/
│   ├── assets/scene.xml              # MuJoCo scene definition with camera & ArUco markers
│   ├── assets/so101.xml              # Robot MJCF: finger pads, grasp-site TCP, 25 mm jaw tag
│   ├── sim_robot.py                  # MuJoCo simulation robot wrapper
│   └── sim_calibration.py            # Synthetic homography calibrator for virtual camera
├── vision/
│   ├── camera.py                     # Camera enumeration, V4L2 device inspection & backend selection
│   ├── capture.py                    # Shared capture thread manager & stream_video loop
│   ├── detector.py                   # HSV salient color segmentation & contour detection
│   ├── frames.py                     # Dual-buffer engine (pristine raw vs operator annotated) + frame timestamps
│   ├── gripper_locator.py            # Jaw-tag locator: parallax-free grasp-site XY in mm
│   ├── calibrator.py                 # Calibrator singleton registry (real & sim modes)
│   ├── targets.py                    # Model-target blackboard & sleek bounding box overlay
│   ├── overlays.py                   # Dynamic operator overlay toggles, accessors & workspace overlay drawing
│   ├── sim_runtime.py                # Headless MuJoCo simulation worker & frame sink
│   └── patterns.py                   # Standby test pattern & placeholder generators
├── agent/
│   ├── contracts.py                  # Subsystem TypedDict contracts for tool responses
│   ├── prompts/
│   │   └── vla_system.md             # Externalized VLA system prompt template
│   ├── live_session.py               # Gemini Live WebSocket session manager
│   ├── response_assembler.py         # Pure streaming response markdown assembler
│   ├── guardrails.py                 # Pick-loop phases, TargetLock, escalation, grasp verdict
│   ├── pick_loop.py                  # Lift vision vote and refusal-hint builders
│   ├── motion.py                     # Safe approach segments & cancellable multi-segment runs
│   ├── tools.py                      # Tool registry, declarations, handlers & prompt generation
│   └── vla_controller.py             # Closed-loop visual verification loop & guardrails
├── tools/
│   ├── calibration_sheet.py          # Printable ArUco marker PNG & calibration mat generator
│   └── setup_wizard.py               # Interactive CLI wizard for table dimensions & markers
├── ui/
│   ├── app.py                        # FastAPI application mounting Gradio Web UI
│   ├── state.py                      # Lazy kinematics 3D state synchronization manager
│   └── components/                   # Gradio chat, controls, and vision viewports
└── tests/                            # 4-tier suite; tests/harness/sim_pick.py = scripted pick-and-place vs MuJoCo truth
```

---

## 7. Layered Architecture & Import-Linter Contract

The codebase strictly enforces a bottom-to-top layered dependency direction:

```
robot  <  simulation  <  vision  <  agent  <  ui
```

- **`robot/`** (bottom layer): Defines core interfaces, kinematics, inverse kinematics, calibration geometry, perception metrics, and the active-robot registry (`robot/registry.py`). Never imports `simulation`, `vision`, `agent`, or `ui`.
- **`simulation/`**: Wraps MuJoCo simulation, environment logic, and synthetic homography calculation. Imports `robot` only.
- **`vision/`**: Owns camera hardware capture, V4L2 enumeration, dual frame buffers, the ArUco calibrator singleton registry, overlay blackboard, and sim runtime. Imports `robot` and `simulation`. Never imports `agent` or `ui`.
- **`agent/`**: Owns Gemini Live session management, VLA orchestration, and primitive robot tool executors. Imports `vision`, `simulation`, and `robot`. Never imports `ui`.
- **`ui/`** (top layer): Gradio and FastAPI presentation layer only. Pure consumer of all underlying layers.
- **`tools/`** (operator package): Standalone CLI utilities (`tools/calibration_sheet.py`, `tools/setup_wizard.py`). May import layers below it; no architectural package may import `tools` (only `main.py` dispatches into it).

### Enforcement
This architecture is enforced locally by the pre-commit hook (there is no CI) using `import-linter` via `[tool.importlinter]` in `pyproject.toml`:
- `uv run lint-imports` validates that all layer boundaries and tool isolation contracts are kept.
- Cycle-dodging function-local imports are strictly prohibited.

---

## 8. Known Loose Ends

- **Real-mode routing fix not hardware-verified:** First real-mode session after PR A: issue one `move_to_spatial_point` and confirm the log shows the real calibration (not `config/workspace_homography_sim.npz`) and the arm lands where the camera says.
- **Log flood on a failing render:** Rate-limited to once per 5 seconds.
- **Untested path:** A desktop session with a display (GLFW backend) was not exercised after the EGL change. `tests/simulation/test_headless_gl.py` only asserts the backend choice, not a real GLFW render.
- **Unreachable code:** The `mode == "sim"` branch of `_stream_video_impl` (`vision/capture.py`) cannot be reached from the app (sim mode never starts a capture thread). Cleaned up in Phase 4 B7.
- **3e known gaps not in Phase 4:**
  - UI gripper controls do not update `GuardrailState.last_commanded_gripper`, so the phase derived at the next instruction can be wrong.
  - `target_ambiguous` is a width proxy (blob > 55 mm), not a real two-object detection.
  - Transport and place use no correction ($c = 0$), so placement is off by the arm bias (~7 mm at a 1.5° pan offset). Carrying the grasp lock's correction to the destination would fix it.
  - Real arm servo gravity sag is confirmed on hardware and compensated in software (`_wait_for_convergence` settles within 0.1° over 0.15 s, shifts goal by steady-state error up to ±6.0° over ≤ 2 rounds with a 3 mm contact guard); servo registers still run LeRobot's default P-only gains.
- **Host-environment note:** `gh` moved from `/usr/bin/gh` to Homebrew, which broke git's HTTPS credential helper until `gh auth setup-git` was re-run. Relevant if `git push` fails with `could not read Username`.
- **Real LeRobot calibration cache:** Lives in `~/.cache/huggingface/lerobot/calibration/robots/so101_follower/`.

