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
                  │    ui/stream.py: Dual Buffer Engine    │
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

To eliminate UI-to-model visual feedback leakage, `ui/stream.py` implements a **Dual-Buffer Stream Engine**:

1. **Model Vision Channel (`get_latest_raw_jpeg`)**: Supplies pristine, untouched RGB camera frames to the Gemini Live session and OpenCV perception detectors. It is guaranteed to be free of bounding boxes, text overlays, crosshairs, or UI annotations.
2. **Operator Display Channel (`/api/stream`, `/api/sim_stream`)**: Overlays dynamic ArUco calibration outlines, detected object bounding boxes, and model target focus badges for human operator inspection.

---

## 2. Core Feature Inventory (F1 – F12)

| # | Feature | Subsystems | Description |
| --- | --- | --- | --- |
| **F1** | ArUco Marker ID & Corner Harmonization | `robot/calibration.py`, `config/workspace_calibration.json`, `simulation/assets/scene.xml` | Canonical fiducial corner IDs (`1: Top-Left`, `2: Top-Right`, `3: Bottom-Right`, `0: Bottom-Left`) harmonized across physical config, simulation XML, and prompts. |
| **F2** | Homography & Lateral Axis Parity | `config/workspace_homography.npz`, `config/workspace_homography_sim.npz`, `robot/calibration.py` | Lateral $Y$-axis parity guaranteed between real and simulation frames. Target `[760, 484]` maps to $X \approx 0.197$ m, $Y \approx +0.145$ m, yielding inward shoulder pan ($\approx -42^\circ$) inside camera FOV. |
| **F3** | Calibrator Mode Context Routing | `agent/tools.py`, `ui/stream.py`, `ui/state.py` | `image_point_to_xyz` queries active robot runtime mode (`sim` vs `real`) dynamically, loading the corresponding homography without hardcoded assumptions. |
| **F4** | Reachability & FOV Kinematic Grounding | `robot/cartesian_ik.py`, `robot/calibration.py` | All normalized coordinates $[y, x] \in [0, 1000]$ within the calibrated mat map to collision-free Cartesian poses ($r \le 391$ mm) contained within the camera FOV. |
| **F5** | Programmatic Perception Metrics Engine | `robot/perception.py`, `robot/vision_detector.py` | Programmatic extraction of measured gripper $[y_g, x_g]$, target $[y_t, x_t]$, relative offset $[dy, dx]$, Euclidean distance, alignment status, and post-lift elevation check. |
| **F6** | Tool Execution Feedback Enrichment | `agent/tools.py` | `move_to_spatial_point`, `close_gripper`, and `open_gripper` return payloads enriched with verified perceptual measurements and context-sensitive visual hints. |
| **F7** | Anti-Hallucination Descent Gate | `agent/vla_controller.py`, `agent/tools.py` | Strict controller verification: prevents descent ($z < 3.0$ cm) when measured offset exceeds tolerance ($|dy| > 25$ or $|dx| > 25$), mandating corrective hover moves. |
| **F8** | Anti-Hallucination Transport Gate & Retry Loop | `agent/vla_controller.py` | Blocks transport if target object remains detected on tabletop post-lift. Enforces an automated retry loop (up to 3 attempts) before escalating to the user. |
| **F9** | Clean Diagnostic Logging & UI Separation | `agent/vla_controller.py`, `agent/tools.py`, `ui/components/chat.py` | Standardized ISO-8601 millisecond logs routed to `sys.stdout`/`sys.stderr`. Raw JSON payloads, IK solver iterations, and tracebacks are excluded from Chatbot viewport. |
| **F10**| Comprehensive 4-Tier E2E Test Suite | `tests/test_e2e_spatial_grounding.py`, `tests/test_e2e_perception_guardrails.py` | 4-tier opaque-box test suite verifying all features across isolation, boundaries, cross-feature interactions, and full real-world scenarios. |
| **F11**| Continuous Integration & Regression Shield | `tests/` (all modules) | Deterministic test harness running 157 tests with zero external network dependencies and 100% pass rate. |
| **F12**| Adversarial Hardening | `tests/test_cartesian_ik.py`, `tests/test_perception.py`, `tests/test_ui.py` | Stress testing against extreme coordinate noise, stream concurrency hammers, degenerate corner inputs, and servo angle clamping. |

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

- **`compute_visual_alignment(frame_rgb, target_point_1000, gripper_point_1000=None, robot_state=None, tolerance=25.0) -> dict`**:
  - **Returns:**
  
    ```python
    {
        "gripper_point_1000": [float, float],
        "target_point_1000": [float, float],
        "measured_offset": {
            "dy": float,  # target_y - gripper_y
            "dx": float,  # target_x - gripper_x
            "distance": float,  # hypot(dy, dx)
        },
        "is_aligned": bool,  # abs(dy) <= tolerance and abs(dx) <= tolerance
        "alignment_tolerance": float,
    }
    ```

- **`verify_grasp_elevation(pre_lift_frame_rgb, post_lift_frame_rgb, target_point_1000=None, max_table_presence_dist=80.0) -> dict`**:
  - **Returns:**
  
    ```python
    {
        "object_elevated": bool,  # True if present pre-lift and absent post-lift
        "object_detected_on_table": bool,  # True if object still detected on table
        "grasped": bool,  # Confirmed grasp
        "confidence": float,  # Confidence score [0.0, 1.0]
    }
    ```

### 3.3. Tool Executor (`agent/tools.py`) ↔ Gemini FunctionResponse

- **`move_to_spatial_point` Return Payload**:
  
  ```python
  {
      "status": "success",
      "target_xyz_m": [float, float, float],
      "executed_qpos_deg": [float, float, float, float, float, float],
      "approach": "top_down" | "horizontal",
      "perception": dict,  # Output from compute_visual_alignment (when camera available)
      "visual_hint": str,  # Actionable instruction for the model's next turn
      "warnings": list[str],  # Optional joint limit warnings
  }
  ```

- **`close_gripper` / `open_gripper` Return Payload**:
  Must include `visual_hint` guiding the model to perform immediate post-action visual inspection before chaining further commands.

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
│  - Mode-Aware Calibrator Context Routing (Sim vs Real vs Twin)              │
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
│   ├── calibration.py                # ArUco detection, homography & coordinate transformations
│   ├── perception.py                 # Visual alignment [dy, dx] & grasp elevation metrics
│   ├── vision_detector.py            # HSV salient color segmentation & contour detection
│   ├── cartesian_ik.py               # Numerical damped least-squares inverse kinematics
│   ├── real_robot.py                 # Physical SO-101 Feetech servo driver (via LeRobot)
│   ├── twin_robot.py                 # Digital twin synchronizer
│   ├── interface.py                  # Abstract base RobotInterface contract
│   ├── constants.py                  # Joint limits, home poses, and serial port defaults
│   └── theory.md                     # Mathematical derivations for kinematics and homography
├── agent/
│   ├── live_session.py               # Gemini Live WebSocket session manager
│   ├── tools.py                      # Primitive robot tool declarations & execution adapters
│   └── vla_controller.py             # Closed-loop visual verification loop & guardrails
├── simulation/
│   ├── assets/scene.xml              # MuJoCo scene definition with camera & ArUco markers
│   ├── sim_robot.py                  # MuJoCo simulation robot wrapper
│   ├── sim_calibration.py            # Synthetic homography calibrator for virtual camera
│   └── env.py                        # Gym-style RL environment wrapper
├── ui/
│   ├── app.py                        # FastAPI application mounting Gradio Web UI
│   ├── state.py                      # Thread-safe global RobotStateManager
│   ├── stream.py                     # Dual-buffer frame engine & dynamic target overlays
│   └── components/                   # Gradio chat, controls, and vision viewports
└── tests/                            # Comprehensive 4-tier automated test suite
```
