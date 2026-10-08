# ![GeminiRobotics-SO101 Banner](banner.png)

## Welcome to **GeminiRobotics-SO101** 🤖

This repository combines Google's Gemini Robotics AI models (`gemini-robotics-er`) and capabilities with the **SO-101 Robotic Arm** setup.

At the core of our user experience is a unified **FastAPI + Gradio 6.0 Web VLA Interface** powered by OpenCV and Three.js.

This interactive web application serves as the unified dashboard for physical hardware manipulation, real-time video observation streaming, interactive 3D WebAssembly simulation viewing, degree-tuned actuator adjustments, and live multimodal action execution via the **Gemini Live API**!

<p align="center">
  <img src="ui.jpeg" alt="GeminiRobotics-SO101 Gradio Web UI" />
</p>

We use Python version **3.13.7** and the extremely fast, deterministic package manager `uv` for managing all environments and dependencies.

> [!IMPORTANT]
> This repo is actively under development as we integrate intelligent robotics behaviors and VLA trajectory inference driven by Gemini Robotics!

## Architecture & Modules

The repository strictly enforces a layered dependency architecture (`robot < simulation < vision < agent < ui`, with operator tooling in `tools/` importing layers below it):

1. **[`/robot`](robot/) (Core Robot Definition, Kinematics & Registry)**
   - **Constants & Poses**: Definitional parameters, joint names, hardware limits, and standard target postures ([`constants.py`](robot/constants.py)).
   - **Kinematics & Theory**: Forward kinematics ([`kinematics.py`](robot/kinematics.py)), numerical damped least-squares Inverse Kinematics with null-space projection ([`cartesian_ik.py`](robot/cartesian_ik.py)), and mathematical theoretical foundations ([`theory.md`](robot/theory.md)).
   - **Workspace Calibration**: ArUco fiducial detection, perspective homography mapping, and dual real/sim homography persistence ([`calibration.py`](robot/calibration.py), [`config/workspace_calibration.json`](config/workspace_calibration.json)).
   - **Programmatic Perception**: OpenCV visual alignment offset engine and post-lift grasp verification ([`perception.py`](robot/perception.py), [`vision_detector.py`](robot/vision_detector.py)).
   - **Hardware Driver**: Physical serial bus interface for Feetech servos ([`real_robot.py`](robot/real_robot.py)) adhering to [`RobotInterface`](robot/interface.py).
   - **Active Robot Registry**: Thread-safe global active-robot registry ([`registry.py`](robot/registry.py)).

2. **[`/simulation`](simulation/) (MuJoCo Physics & Environments)**
   - **Physics Simulation**: High-level MuJoCo model manipulation ([`sim_robot.py`](simulation/sim_robot.py)), fixed workspace camera (`workspace_cam`), ArUco corner fiducials, centered 3D STL meshes, and passive 3D interactive rendering ([`simulate.py`](simulation/simulate.py)).
   - **Synthetic Workspace Calibration**: Analytical pinhole camera projection and exact geometric homography computation ([`sim_calibration.py`](simulation/sim_calibration.py), [`config/workspace_homography_sim.npz`](config/workspace_homography_sim.npz)).
   - **Reinforcement Learning & Warp**: Gymnasium reach target environments ([`env.py`](simulation/env.py)) and hardware-accelerated batched simulations using NVIDIA Warp ([`warp_env.py`](simulation/warp_env.py)).
   - **Validation & Test Suites**: Diagnostic verifiers including linear Cartesian sweeps ([`test_movement.py`](simulation/test_movement.py)) and system health checks ([`check_env.py`](simulation/check_env.py)).

3. **[`/vision`](vision/) (Camera Capture, Dual Buffers & Sim Runtime)**
   - **Camera Driver & Enumeration**: V4L2 device capability querying and camera discovery ([`camera.py`](vision/camera.py)).
   - **Shared Capture Manager**: Multi-consumer shared capture thread and video stream production ([`capture.py`](vision/capture.py)).
   - **Dual Frame Buffer Engine**: Separation between untouched raw model observations (`get_latest_raw_jpeg`) and operator annotated display frames (`get_latest_jpeg`) ([`frames.py`](vision/frames.py)).
   - **Calibrator Registry & Blackboard**: Mode-aware workspace calibrator caching ([`calibrator.py`](vision/calibrator.py)) and model-target blackboard with sleek corner-notched bounding box overlays ([`targets.py`](vision/targets.py)).
   - **Simulation Frame Runtime**: Dedicated background sim rendering worker ([`sim_runtime.py`](vision/sim_runtime.py)).

4. **[`/agent`](agent/) (Gemini Live API & VLA Controller)**
   - **Live WebSocket Session**: Real-time bidirectional streaming session manager (`GeminiLiveSession` in [`live_session.py`](agent/live_session.py)) supporting Google AI Studio and Vertex AI credentials.
   - **Robot Tooling & Execution**: Tool declarations (`ROBOT_TOOL_DECLARATIONS`) and execution adapters (`execute_tool_call` in [`tools.py`](agent/tools.py)) supporting smooth pose interpolation, direct joint control, gripper actuation, named preset poses, and calibrated spatial pointing with vertical offset control.
   - **VLA Controller & Verification Loop**: Asynchronous controller (`VLAController` in [`vla_controller.py`](agent/vla_controller.py)) enforcing a strict **"Move -> Look -> Correct"** visual verification loop, post-action camera frame observation injection with 0.3s vision settling delays, continuous video frame ingestion (~1 FPS), enriched orchestration prompt guidance, spatial grounding for real & sim modes, and emergency stop handling.
   - **Verbose Diagnostic Terminal Logging**: ISO-8601 millisecond-precision diagnostic logging (`[YYYY-MM-DD HH:MM:SS.mmm] [MODULE] [TAG] message`) routing model decisions, tool call payloads (`call.args`), and execution results (`result_dict`) to stdout/stderr while keeping Chat UI reasoning clean.

5. **[`/tools`](tools/) (Operator Tooling & Wizards)**
   - **Printable Calibration Sheets**: True-to-scale workspace calibration mat poster and ArUco marker PNG generator ([`calibration_sheet.py`](tools/calibration_sheet.py)).
   - **Interactive Setup Wizard**: Terminal CLI wizard configuring table dimensions, clearances, and marker reuse ([`setup_wizard.py`](tools/setup_wizard.py)).

6. **[`/ui`](ui/) (Unified Web VLA Interface & 3D Viewer)**
   - **FastAPI + Gradio Server**: Unified web server mounting static simulation assets (`/sim_assets`, `/ui_assets`), dynamic simulation state endpoint (`/api/sim_state`), `/api/stream` and `/api/sim_stream` MJPEG endpoints, and Gradio 6.0 Blocks layout with TehnoX dark theme ([`app.py`](ui/app.py)).
   - **Interactive 3D Simulation Viewer**: Embedded Three.js WebAssembly 3D viewer ([`assets/sim_viewer.html`](ui/assets/sim_viewer.html)) rendering real-time kinematics and centered mesh poses in `sim` mode.
   - **Lazy Kinematics State Management**: Centralized robot state manager and lazy kinematics transforms ([`state.py`](ui/state.py)).
   - **Multimodal Chat & Dynamic UI Blocking**: Real-time conversational AI interface with full input blocking during active model turns, dynamic status banner (`🟢 Status: Ready` / `⏳ Status: Working...`) ([`components/chat.py`](ui/components/chat.py)), and degree-tuned servo control sliders ([`components/controls.py`](ui/components/controls.py)).

## Getting Started

### Requirements

- Python 3.13.7
- `uv` package manager ([Installation Guide](https://docs.astral.sh/uv/getting-started/installation/))
- Clone this repository:

```bash
git clone https://github.com/Ekumen-OS/GeminiRobotics-SO101.git
cd GeminiRobotics-SO101
```

- Setup the repository environment and install dependencies:

```bash
uv sync
source .venv/bin/activate
```

### Environment Configuration (`.env`)

To enable Gemini Live API features and multimodal chat, configure your API credentials by copying `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` with your preferred authentication method:

```bash
# Option A: Google AI Studio API Key (Recommended)
GOOGLE_API_KEY=your_google_ai_studio_api_key_here

# Option B: Google Cloud Vertex AI
# GOOGLE_CLOUD_PROJECT=your_project_id
# GOOGLE_CLOUD_LOCATION=us-central1

# Physical SO-101 Serial Port (optional, defaults to /dev/tty.usbmodem5B415332861)
# ROBOT_PORT=/dev/ttyACM0

# Grasp Sensing & Verification (Phase 3a)
# GRIPPER_EMPTY_APERTURE_DEG=-74.5
# GRASP_EMPTY_MARGIN_DEG=6.0
# GRIPPER_SETTLE_S=0.6
# GRASP_LOAD_MIN=20.0
```

- If using physical hardware, use the LeRobot package to find the serial port of your SO-101 robotic arm and calibrate it ([LeRobot SO-101 Guide](https://huggingface.co/docs/lerobot/so101)).
- You are now ready to launch the Web UI and control your robot!

### Configuration Settings Reference

| Environment Variable | Default | Description |
| :--- | :--- | :--- |
| `GRIPPER_EMPTY_APERTURE_DEG` | `-75.0` | Aperture the gripper servo reports when closed on nothing. Set from step 1 of the checklist below; the arm's calibration decides this value, not the model. |
| `GRASP_EMPTY_MARGIN_DEG` | `6.0` | Margin in degrees above the empty aperture to classify as `closed_on_air` (threshold = empty aperture + margin). |
| `GRIPPER_SETTLE_S` | `0.6` | Stabilization settling delay in seconds after gripper motion before querying servo registers. |
| `GRASP_LOAD_MIN` | *unset* | Minimum motor load threshold required to verify grasp (optional secondary guardrail). |
| `ROBOT_PORT` / `SO101_PORT` | `/dev/tty.usbmodem...` | Serial device path for physical Feetech bus adapter. |
| `CAMERA_NAME` / `CAMERA_INDEX` | *auto* | Device substring or index for external camera selection. |
| `WEB_STREAM_FPS` | `15` | Frame rate for operator MJPEG stream. |
| `WEB_JPEG_QUALITY` | `75` | JPEG encoding quality for operator MJPEG stream. |

### Real-Arm Grasp Calibration Checklist

Follow this 4-step procedure when deploying to a physical SO-101 robotic arm to tune the grasp classifier:

1. **Close on Air**: Execute `close_gripper` with no object between jaws. Inspect terminal log `[Tools] [GRASP] aperture=...` and record the fully closed aperture $A_{\text{empty}}$ (e.g., $-74.5^\circ$).
2. **Close on Object**: Place your standard target object (e.g., 30 mm calibration cube) between the jaws and execute `close_gripper`. Record the stalled aperture $A_{\text{object}}$ (e.g., $-34.5^\circ$).
3. **Calculate Gap**: Compute $\Delta A = A_{\text{object}} - A_{\text{empty}}$ (e.g., $-34.5^\circ - (-74.5^\circ) = 40.0^\circ$).
4. **Set Anchor and Margin**: In your `.env`, set `GRIPPER_EMPTY_APERTURE_DEG` to $A_{\text{empty}}$ and `GRASP_EMPTY_MARGIN_DEG` to approximately 40% of the gap:
   $$\text{GRASP\_EMPTY\_MARGIN\_DEG} \approx 0.40 \times \Delta A$$
   *(e.g., $A_{\text{empty}} = -74.5^\circ$, margin $= 0.40 \times 40.0^\circ = 16.0^\circ$, threshold $= -58.5^\circ$). Without the anchor the threshold is measured from $-75^\circ$, which is only correct if your calibration happens to close there.*
5. **Leave `GRASP_LOAD_MIN` unset** until you have read a few `load=` values from the log: the STS3215 `Present_Load` register is raw sign-magnitude (bit 10 is direction), so a reverse-direction load reads above 1023.

## Running the Robot (`main.py`)

[`main.py`](main.py) is the unified entry point for running the SO-101 robotic arm in physical control, MuJoCo simulation, or automated verification. By default, running `main.py` launches the modern **FastAPI + Gradio VLA Web UI Server** on port `7860`:

```bash
# Instantly launch the Unified VLA Web UI server (default mode is 'real')
# Open your browser at http://localhost:7860/
uv run python main.py

# Launch Web UI specifying an explicit physical serial port (or set ROBOT_PORT in .env)
uv run python main.py --mode real --port /dev/ttyACM0

# Launch Web UI with interactive 3D simulation viewer
# (the former 'twin' split view was removed; only 'real' and 'sim' are supported)
uv run python main.py --mode sim

# Development auto-reload (restarts server when Python files change)
uv run python main.py --mode sim --reload
```

> [!NOTE]
> The `--port` argument defaults to the value set in `ROBOT_PORT` (or `SO101_PORT`) in your `.env` file, falling back to `/dev/tty.usbmodem5B415332861`. If your serial adapter differs, configure `ROBOT_PORT` in `.env` or pass `--port <path>` (e.g., `/dev/ttyACM0` on Linux or `/dev/tty.usbmodem...` on macOS).
>
> **Linux Serial Permissions**: On Linux, ensure your user has read/write permissions for the serial port device (e.g., `/dev/ttyACM0`). You can grant immediate access with `sudo chmod 666 /dev/ttyACM0` or permanent access with `sudo usermod -a -G dialout $USER` (requires logout or `newgrp dialout`). You can run `uv run python main.py --task check_environment` to verify port access.

## Gemini Live API & Robot Tool Capabilities

When interacting with the AI in the Web UI chat or via the agent controller, Gemini has access to real-time function calling tools:

| Tool Name | Description | Parameters |
| :--- | :--- | :--- |
| `send_message` | Speak, narrate observations, explain planned steps, or confirm completion. | `message` (string) |
| `move_to_pose` | Smoothly interpolate arm to 6-DOF joint target in degrees. | `qpos` (array of 6 degs), `duration_sec` |
| `set_joint_positions` | Instant snap joint position command in degrees. | `qpos` (array of 6 degs) |
| `open_gripper` | Fully open the gripper end-effector. | None |
| `close_gripper` | Close the gripper end-effector. | None |
| `move_to_preset` | Move arm to named keyframe (`HOME`, `MIDDLE`, `SWEEP_HOME`). | `pose_name` (string) |
| `move_to_spatial_point` | Ground 2D image coordinates $[y, x] \in [0, 1000]$ into 3D Cartesian coordinates via ArUco calibration and solve IK at target $z$ height. | `point` (array of 2 numbers), `z_height_cm` (optional, default: 3.0) |
| `get_joint_positions` | Read current joint angles in degrees. | None |
| `get_end_effector_pose` | Read 3D Cartesian position and orientation matrix (sim). | None |

## Model-Driven Closed-Loop Orchestration & Workspace Calibration

### 1. ArUco Workspace Calibration & Sheet Generator (`robot/calibration.py`)

- Detects 4-corner ArUco fiducials (`DICT_4X4_50`, canonical IDs: `1: Top-Left`, `2: Top-Right`, `3: Bottom-Right`, `0: Bottom-Left`) defining the physical workspace mat.
- Computes perspective homography mapping camera pixels directly to physical robot base millimeters with lateral $Y$-axis parity.
- Configurable via `config/workspace_calibration.json`. Full architecture details are in [`ARCHITECTURE.md`](ARCHITECTURE.md).
- **Interactive Setup Wizard & Marker / Sheet Generator**:

  ```bash
  # Launch interactive calibration wizard (auto-detects existing markers and asks whether to reuse or replace):
  uv run python main.py --task generate_markers

  # Reuse existing physical markers and update dimensions in JSON without regenerating files:
  uv run python main.py --task generate_markers --reuse

  # Force regeneration of marker PNGs and calibration mat sheet:
  uv run python main.py --task generate_markers --regenerate
  ```

  Features & Artifacts:
  - **Auto-Detection**: Automatically detects existing marker files in `assets/aruco_markers/` and prompts whether to reuse them or regenerate.
  - `assets/aruco_markers/workspace_calibration_sheet.png` — An all-in-one printable tabletop calibration mat with 4 corner markers, robot base alignment guide, workspace boundaries, and coordinate axes.
  - `assets/aruco_markers/marker_{0..3}.png` — Individual high-resolution marker stickers.

### 2. Model-as-Orchestrator Closed-Loop Architecture (`agent/vla_controller.py`)

- Powered by `gemini-robotics-er-2-streaming-preview` via the Gemini Live API bidirectional WebSocket session with thinking enabled (`include_thoughts=True`).
- **Continuous Visual Perception**: Ingests video frames at ~1 FPS both during motion execution and continuously in the background while the model reasons between tool calls.
- **Strict "Move -> Look -> Correct" Visual Verification Loop**:
  - Enforces a mandatory 3-phase cycle for spatial manipulation:
    1. **Move**: Move the arm/gripper towards approach hover altitude above the target object using `move_to_spatial_point`.
    2. **Look**: Model explicitly pauses, inspects the freshly settled post-action camera frame, detects current gripper position $[y_g, x_g]$ and target coordinates $[y_t, x_t]$, and computes the relative offset error $[dy, dx] = [y_t - y_g, x_t - x_g]$.
    3. **Correct**: If visual alignment error exceeds threshold ($\| [dy, dx] \| > \epsilon$), the model executes corrective micro-adjustments before descending to grasp depth.
- **Programmatic Perception Feedback & Anti-Hallucination Guardrails (`robot/perception.py`)**:
  - Automatically calculates real OpenCV visual offsets $[dy, dx]$, distances, and binary alignment flags (`is_aligned` within tolerance $\le 25$) returned in tool payloads.
  - Enforces strict controller gates blocking table descent when misaligned ($|dy| > 25$ or $|dx| > 25$) and blocking transport if the object remains on the table post-lift.
  - Automates a 3-attempt grasp retry loop with graceful escalation if missed.
- **Post-Action Observation Frame Injection**: Injects a fresh settled camera frame (with a 0.3s physical stabilization delay) immediately after every blocking tool execution before returning `FunctionResponse`, providing ground-truth post-action state for immediate inspection and error correction.
- **UI Input Blocking & Dynamic Status Indication**: Prevents concurrent user input or race conditions during model turns. When an instruction is submitted, the prompt textbox, execution button, homing button, clear button, and randomize target button are disabled, and the banner transitions to `⏳ Status: Working...`. Once the model finishes its execution turn (or on exception), the UI is safely restored to `🟢 Status: Ready` via a robust `try...finally` block.
- **Verbose Timestamped Terminal Logging vs. Chatbot CoT**:
  - Maintains a clean separation between user-facing conversational narration and operator diagnostics.
  - High-level thoughts (`> 💭 *...*`) and responses stream cleanly to the Gradio Chatbot.
  - Low-level JSON tool call payloads (`call.args`), execution results (`result_dict`), WebSocket frame dispatches, and error traces are printed to `stdout`/`stderr` with ISO-8601 millisecond timestamps (`[YYYY-MM-DD HH:MM:SS.mmm] [MODULE] [TAG] message`).
- **Minimalist Multi-Target Bounding Box Overlay**: Renders clean, color-coded bounding boxes with corner accents and floating coordinate badges on the video stream showing all detected objects and where Gemini is focusing in real time.
- **Continuous Live ArUco Calibration**: Dynamically tracks visible fiducials and computes perspective homography continuously on the physical USB camera feed, eliminating camera perspective drift.
- **Direct Primitive Tool Calling**: Rather than relying on a rigid state machine, the model orchestrates multi-step actions (Hover $\rightarrow$ Look $\rightarrow$ Correct $\rightarrow$ Descend $\rightarrow$ Grasp $\rightarrow$ Retract $\rightarrow$ Transport $\rightarrow$ Release) by invoking tool primitives.

## Testing & Diagnostics

Verify system health, MuJoCo simulation physics, inverse kinematics, and unit test suites:

```bash
# 0. Install pre-commit hooks for automated code formatting and lint checks
uv run pre-commit install

# 1. Run complete automated unit test suite (Workspace Calibration, Tool Adapters, VLA Controller, UI)
uv run pytest tests/ -v

# 2. Verify movement & kinematics via automated X, Y, and Z Cartesian line sweeps
uv run python main.py --task test_simulation

# 3. Check system diagnostics, dependency imports, and MuJoCo FPS throughput
uv run python main.py --task check_environment

# 4. Test Gemini Live API WebSocket session and tool calling (requires .env)
uv run python test_live.py
```

## Kinematics & Mathematics Reference

For a complete explanation of Denavit-Hartenberg (DH) parameters, transformation matrices, spatial Jacobians, and under-actuated 5-DOF damped least-squares Inverse Kinematics, consult our documentation:

- [SO-101 Robot Arm Kinematics Theory (`robot/theory.md`)](robot/theory.md)
- [Behind the Code: Engineering GeminiRobotics-SO101 (`behind_the_code.md`)](behind_the_code.md)

---

Made with ❤️ in Colombia 🇨🇴
