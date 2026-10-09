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
2. **Operator Display Channel (`/api/stream`, `/api/sim_stream`)**: Overlays dynamic ArUco calibration outlines, detected object bounding boxes, and model target focus badges for human operator inspection. Both are long-lived `multipart/x-mixed-replace` MJPEG responses rendered by a plain `<img>`, so delivery needs throughput, not a per-frame round trip (per-frame Gradio image delivery froze on any link whose round-trip time exceeded the frame interval, e.g. Tailscale/WAN). Browser quality and rate are `WEB_JPEG_QUALITY` (default 75) and `WEB_STREAM_FPS` (default 15); the model path is unaffected.
3. **Shared capture**: camera capture is one background thread (`ensure_camera_capture` / `switch_camera_capture` in `vision/capture.py`), independent of any browser session. Every tab reads the same frame buffers, so opening a second tab joins the running capture instead of replacing it.

---

## 2. Core Feature Inventory (F1 – F14)

| # | Feature | Subsystems | Description |
| --- | --- | --- | --- |
| **F1** | ArUco Marker ID & Corner Harmonization | `robot/calibration.py`, `config/workspace_calibration.json`, `simulation/assets/scene.xml` | Canonical fiducial corner IDs (`1: Top-Left`, `2: Top-Right`, `3: Bottom-Right`, `0: Bottom-Left`) harmonized across physical config, simulation XML, and prompts. |
| **F2** | Homography & Lateral Axis Parity | `config/workspace_homography.npz`, `config/workspace_homography_sim.npz`, `robot/calibration.py` | Lateral $Y$-axis parity guaranteed between real and simulation frames. Target `[760, 484]` maps to $X \approx 0.197$ m, $Y \approx +0.145$ m, yielding inward shoulder pan ($\approx -42^\circ$) inside camera FOV. |
| **F3** | Calibrator Mode Context Routing | `agent/tools.py`, `vision/calibrator.py`, `robot/registry.py` | `execute_tool_call` receives the controller's `mode` explicitly and otherwise reads `robot.mode` (`resolve_robot_mode`); `image_point_to_xyz` then loads the matching homography. No duck-typing on rendering methods. |
| **F4** | Reachability & FOV Kinematic Grounding | `robot/cartesian_ik.py`, `robot/calibration.py` | All normalized coordinates $[y, x] \in [0, 1000]$ within the calibrated mat map to collision-free Cartesian poses ($r \le 391$ mm) contained within the camera FOV. |
| **F5** | Programmatic Perception Metrics Engine | `robot/perception.py`, `robot/vision_detector.py` | Programmatic extraction of measured gripper $[y_g, x_g]$, target $[y_t, x_t]$, relative offset $[dy, dx]$, Euclidean distance, alignment status, and post-lift elevation check. |
| **F6** | Tool Execution Feedback Enrichment | `agent/tools.py` | `move_to_spatial_point`, `close_gripper`, and `open_gripper` return payloads enriched with verified perceptual measurements and context-sensitive visual hints. |
| **F7** | Anti-Hallucination Descent Gate | `agent/vla_controller.py`, `agent/tools.py` | Strict controller verification: prevents descent ($z < 3.0$ cm) when measured offset exceeds tolerance ($|dy| > 25$ or $|dx| > 25$), mandating corrective hover moves. |
| **F8** | Anti-Hallucination Transport Gate & Retry Loop | `agent/vla_controller.py` | Blocks transport if target object remains detected on tabletop post-lift. Enforces an automated retry loop (up to 3 attempts) before escalating to the user. |
| **F9** | Clean Diagnostic Logging & UI Separation | `agent/vla_controller.py`, `agent/tools.py`, `ui/components/chat.py` | Standardized ISO-8601 millisecond logs routed to `sys.stdout`/`sys.stderr`. Raw JSON payloads, IK solver iterations, and tracebacks are excluded from Chatbot viewport. |
| **F10**| Comprehensive 4-Tier E2E Test Suite | `tests/test_e2e_spatial_grounding.py`, `tests/test_e2e_perception_guardrails.py` | 4-tier opaque-box test suite verifying all features across isolation, boundaries, cross-feature interactions, and full real-world scenarios. |
| **F11**| Continuous Integration & Regression Shield | `tests/` (all modules) | Deterministic test harness with zero external network dependencies and 100% pass rate. |
| **F12**| Adversarial Hardening | `tests/test_cartesian_ik.py`, `tests/test_perception.py`, `tests/test_ui.py` | Stress testing against extreme coordinate noise, stream concurrency hammers, degenerate corner inputs, and servo angle clamping. |
| **F13**| Physical Grasp Sensor & Classification | `robot/grasp.py`, `robot/real_robot.py`, `simulation/sim_robot.py`, `agent/tools.py` | Bus register and physics-driven aperture and load telemetry. Pure classifier `classify_grasp` detects jaw stall vs empty air closure with configurable margins and fail-closed safety. |
| **F14**| Fail-Closed Error Taxonomy & Visual Verification Safety | `robot/errors.py`, `agent/errors.py`, `robot/real_robot.py`, `agent/tools.py`, `vision/frames.py` | Hierarchical error taxonomy (`RoboticsError`, `HardwareError`, `CalibrationUnavailable`, `PerceptionUnavailable`, `ToolArgumentError`, `SessionError`). Fails closed on driver read/write, calibration corruption, and missing camera frames (no synthetic placeholder fallback in real mode). Tool error responses communicate typed `error_type` and honest `robot_moved` telemetry. |

---

## 3. Subsystem Interface Contracts

Any agent modifying the codebase MUST preserve the following schemas and functional contracts:

### 3.1. Calibration Engine (`robot/calibration.py`) ↔ Tool Adapter (`agent/tools.py`)

- **`ArucoWorkspaceCalibrator.gemini_to_robot_xyz(point_1000, img_shape=(480, 640), z_m=0.03) -> np.ndarray`**:
  - **Inputs:** `point_1000` as normalized $[y, x] \in [0, 1000]$, image dimensions $(H, W)$, target height $z$ in meters.
  - **Returns:** NumPy array `[X, Y, Z]` in meters in robot base frame.
  - **Invariants:** Lateral $Y$ coordinate must have identical sign conventions in simulation and physical hardware.
- **`ArucoWorkspaceCalibrator.robot_mm_to_pixel(x_mm, y_mm) -> tuple[float, float] | None`**:
  - **Inputs:** Physical robot frame millimeters $(X, Y)$.
  - **Returns:** Sub-pixel image coordinate $(p_x, p_y)$ or `None` if uncalibrated/singular.

### 3.2. Perception Engine (`robot/perception.py`) ↔ Tools & VLA Controller

Payload definitions are formalized as `TypedDict` models in [`robot/contracts.py`](robot/contracts.py):

- **`compute_visual_alignment(frame_rgb, target_point_1000, gripper_point_1000=None, robot_state=None, tolerance=None, calibrator=None) -> VisualAlignmentResponse`**:
  - **Typed contract:** [`VisualAlignmentResponse`](robot/contracts.py) (incorporates [`MeasuredOffset`](robot/contracts.py)).
- **`verify_grasp_elevation(pre_lift_frame_rgb, post_lift_frame_rgb, target_point_1000=None, max_table_presence_dist=80.0, calibrator=None) -> GraspElevationResponse`**:
  - `calibrator` (both functions): the `ArucoWorkspaceCalibrator` used to mask detection to the workspace polygon. `robot/` never fetches it itself; `agent/tools.py` passes `vision.get_calibrator()`.
  - **Typed contract:** [`GraspElevationResponse`](robot/contracts.py).
  - **Fail-Closed Invariant:** If detector raises an exception on either frame, returns `object_elevated=None, object_detected_on_table=False, grasped=None, confidence=0.0`. Never returns fail-open `True`.

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
  - **Typed contract:** [`MoveToSpatialPointResponse`](agent/contracts.py) (incorporates optional [`VisualAlignmentResponse`](robot/contracts.py) under `perception`).
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
    - `error_type: Literal["argument", "calibration", "perception", "hardware", "internal"]`: Machine-readable classification of the failure subsystem.
    - `robot_moved: bool`: Honest indication of whether physical robot actuators moved prior to or during the failure.
  - When hardware failures occur (`error_type="hardware"`), the executor immediately halts physical motion (`robot.interrupt()`) before returning the error payload.

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
│  - Post-Lift Grasp Elevation Check ↔ Transport Guardrail Blocking           │
│  - UI Event Blocking ↔ Model Stream Execution ↔ Guaranteed Interactive Reset│
├─────────────────────────────────────────────────────────────────────────────┤
│                   TIER 2: BOUNDARY VALUES, CORNER CASES & RESILIENCE        │
│  - Extreme Normalized Coordinates ([0,0], [1000,1000], Negative, >1000)     │
│  - Workspace Mat Physical Boundary Edges & Out-of-Reach Kinematic Limits    │
│  - Alignment Tolerance Boundaries (|dy|=25.0, |dy|=25.1, |dx|=25.0, dx=25.1)│
│  - Partial Marker Occlusion, Missing Calibration Config & Corrupt NPZ Files │
│  - Coordinate Parser Fuzzing with Malformed LLM Speech & Mixed Delta Tokens │
├─────────────────────────────────────────────────────────────────────────────┤
│                   TIER 1: ISOLATED FEATURE COVERAGE (UNIT MECHANICS)        │
│  - ArUco Marker ID & Corner Definitions (1:TL, 2:TR, 3:BR, 0:BL)            │
│  - Homography Forward (Gemini→Base XYZ) & Inverse (Base XYZ→Pixel) Mapping  │
│  - Mode-Aware Calibrator Context Routing (Sim vs Real)                      │
│  - Red Cube [760, 484] Canonical Grounding & Shoulder Pan Sign Verification │
│  - Visual Alignment Metrics Calculation (dy, dx, distance, is_aligned)      │
│  - Grasp Elevation Verification (object_elevated, object_detected_on_table) │
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
# 0. Install pre-commit hooks (one-time setup)
uv run pre-commit install

# 1. Run complete automated test suite
uv run pytest tests/ -v

# 2. Run focused spatial grounding E2E tests
uv run pytest tests/test_e2e_spatial_grounding.py -v

# 3. Run focused perception guardrails E2E tests
uv run pytest tests/test_e2e_perception_guardrails.py -v

# 4. Run static lint and code formatting check
uv run ruff check .
uv run ruff format --check .

# 5. Run static type checking (scope is defined in [tool.mypy] `files` in pyproject.toml)
uv run mypy
```

---

## 6. Repository Layout & File Responsibilities

```
GeminiRobotics-SO101/
├── AGENTS.md                         # Operational rules and agent instructions
├── ARCHITECTURE.md                   # System architecture, interface contracts & test spec
├── README.md                         # Human-facing project guide, setup & hardware manual
├── behind_the_code.md                # Engineering deep dive and design decisions
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
│   ├── perception.py                 # Visual alignment [dy, dx] & grasp elevation metrics
│   ├── vision_detector.py            # HSV salient color segmentation & contour detection
│   ├── cartesian_ik.py               # Numerical damped least-squares inverse kinematics
│   ├── real_robot.py                 # Physical SO-101 Feetech servo driver (via LeRobot)
│   ├── interface.py                  # Abstract base RobotInterface contract
│   ├── constants.py                  # Joint limits, home poses, and serial port defaults
│   └── theory.md                     # Mathematical derivations for kinematics and homography
├── simulation/
│   ├── assets/scene.xml              # MuJoCo scene definition with camera & ArUco markers
│   ├── sim_robot.py                  # MuJoCo simulation robot wrapper
│   ├── sim_calibration.py            # Synthetic homography calibrator for virtual camera
│   └── env.py                        # Gym-style RL environment wrapper
├── vision/
│   ├── camera.py                     # Camera enumeration, V4L2 device inspection & backend selection
│   ├── capture.py                    # Shared capture thread manager & stream_video loop
│   ├── frames.py                     # Dual-buffer engine (pristine raw vs operator annotated)
│   ├── calibrator.py                 # Calibrator singleton registry (real & sim modes)
│   ├── targets.py                    # Model-target blackboard & sleek bounding box overlay
│   ├── overlays.py                   # Dynamic operator overlay toggles & accessors
│   ├── sim_runtime.py                # Headless MuJoCo simulation worker & frame sink
│   └── patterns.py                   # Standby test pattern & placeholder generators
├── agent/
│   ├── contracts.py                  # Subsystem TypedDict contracts for tool responses
│   ├── prompts/
│   │   └── vla_system.md             # Externalized VLA system prompt template
│   ├── live_session.py               # Gemini Live WebSocket session manager
│   ├── response_assembler.py         # Pure streaming response markdown assembler
│   ├── tools.py                      # Tool registry, declarations, handlers & prompt generation
│   └── vla_controller.py             # Closed-loop visual verification loop & guardrails
├── tools/
│   ├── calibration_sheet.py          # Printable ArUco marker PNG & calibration mat generator
│   └── setup_wizard.py               # Interactive CLI wizard for table dimensions & markers
├── ui/
│   ├── app.py                        # FastAPI application mounting Gradio Web UI
│   ├── state.py                      # Lazy kinematics 3D state synchronization manager
│   └── components/                   # Gradio chat, controls, and vision viewports
└── tests/                            # Comprehensive 4-tier automated test suite
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

