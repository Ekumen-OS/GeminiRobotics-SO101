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

To keep code modular and hardware, simulation, agent, and user interfaces cleanly isolated, the repository is structured into four primary packages:

1. **[`/robot`](robot/) (Core Robot Definition & Kinematics)**
   - **Constants & Poses**: Definitional parameters, joint names, hardware limits, and standard target postures ([`constants.py`](robot/constants.py)).
   - **Kinematics & Theory**: Forward kinematics ([`kinematics.py`](robot/kinematics.py)), numerical damped least-squares Inverse Kinematics with null-space projection ([`cartesian_ik.py`](robot/cartesian_ik.py)), and mathematical theoretical foundations ([`theory.md`](robot/theory.md)).
   - **Workspace Calibration**: ArUco fiducial detection, perspective homography mapping, and printable calibration mat sheet generator ([`calibration.py`](robot/calibration.py), [`config/workspace_calibration.json`](config/workspace_calibration.json)).
   - **Trajectory Planning**: $C^2$-continuous quintic minimum-jerk motion interpolation and phased pick-and-place generation ([`trajectory.py`](robot/trajectory.py)).
   - **Hardware & Digital Twin**: Physical serial bus interfaces for Feetech servos ([`real_robot.py`](robot/real_robot.py)), simulation controller ([`simulation/sim_robot.py`](simulation/sim_robot.py)), and simultaneous simulation-to-hardware streaming ([`twin_robot.py`](robot/twin_robot.py)) adhering to [`RobotInterface`](robot/interface.py).

2. **[`/agent`](agent/) (Gemini Live API & VLA Controller)**
   - **Live WebSocket Session**: Real-time bidirectional streaming session manager (`GeminiLiveSession` in [`live_session.py`](agent/live_session.py)) supporting Google AI Studio and Vertex AI credentials.
   - **Closed-Loop SPAV Engine**: Evaluative Sense-Plan-Act-Verify state machine with visual grasp verification and visual offset retry correction (`SPAVStateMachine` in [`spav.py`](agent/spav.py)).
   - **Robot Tooling & Execution**: Tool declarations (`ROBOT_TOOL_DECLARATIONS`) and execution adapters (`execute_tool_call` in [`tools.py`](agent/tools.py)) supporting smooth pose interpolation, direct joint control, gripper actuation, named preset poses, calibrated spatial pointing, and automated SPAV pick-and-place dispatch.
   - **VLA Controller**: Asynchronous controller (`VLAController` in [`vla_controller.py`](agent/vla_controller.py)) bridging synchronous UI streaming with Gemini Live WebSocket turns, continuous video frame ingestion, and emergency stop handling.

3. **[`/simulation`](simulation/) (MuJoCo Physics & Environments)**
   - **Physics Simulation**: High-level MuJoCo model manipulation ([`sim_robot.py`](simulation/sim_robot.py)), centered 3D STL meshes, and passive 3D interactive rendering ([`simulate.py`](simulation/simulate.py)).
   - **Reinforcement Learning & Warp**: Gymnasium reach target environments ([`env.py`](simulation/env.py)) and hardware-accelerated batched simulations using NVIDIA Warp ([`warp_env.py`](simulation/warp_env.py)).
   - **Validation & Test Suites**: Diagnostic verifiers including linear Cartesian sweeps ([`test_movement.py`](simulation/test_movement.py)) and system health checks ([`check_env.py`](simulation/check_env.py)).

4. **[`/ui`](ui/) (Unified Web VLA Interface & 3D Viewer)**
   - **FastAPI + Gradio Server**: Unified web server mounting static simulation assets (`/sim_assets`, `/ui_assets`), dynamic simulation state endpoint (`/api/sim_state`), and Gradio 6.0 Blocks layout with TehnoX dark theme ([`app.py`](ui/app.py)).
   - **Interactive 3D Simulation Viewer**: Embedded Three.js WebAssembly 3D viewer ([`assets/sim_viewer.html`](ui/assets/sim_viewer.html)) rendering real-time kinematics and centered mesh poses in `sim` and `twin` modes.
   - **Thread-Safe State Management**: Centralized robot state manager (`RobotStateManager` in [`state.py`](ui/state.py)) for synchronized joint telemetry and actuator feedback.
   - **Vision Streaming & Camera Enumeration**: High-throughput Motion JPEG (MJPG) streaming over OpenCV and native macOS hardware camera enumeration via AVFoundation ([`stream.py`](ui/stream.py), [`components/vision.py`](ui/components/vision.py)).
   - **Multimodal Chat & Controls**: Real-time conversational AI interface with tool execution status updates ([`components/chat.py`](ui/components/chat.py)) and degree-tuned servo control sliders ([`components/controls.py`](ui/components/controls.py)).

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
```

- If using physical hardware, use the LeRobot package to find the serial port of your SO-101 robotic arm and calibrate it ([LeRobot SO-101 Guide](https://huggingface.co/docs/lerobot/so101)).
- You are now ready to launch the Web UI and control your robot!

## Running the Robot (`main.py`)

[`main.py`](main.py) is the unified entry point for running the SO-101 robotic arm in physical control, MuJoCo simulation, digital twin mode, or automated verification. By default, running `main.py` launches the modern **FastAPI + Gradio VLA Web UI Server** on port `7860`:

```bash
# Instantly launch the Unified VLA Web UI server (default mode is 'real')
# Open your browser at http://localhost:7860/
uv run python main.py

# Launch Web UI specifying an explicit physical serial port
uv run python main.py --mode real --port /dev/tty.usbmodem5B415332861

# Launch Web UI with interactive 3D simulation viewer (sim) or digital twin split view (twin)
uv run python main.py --mode sim
uv run python main.py --mode twin --port /dev/tty.usbmodem5B415332861

# Development auto-reload (restarts server when Python files change)
uv run python main.py --mode sim --reload
```

> [!NOTE]
> The `--port` argument defaults to `/dev/tty.usbmodem5B415332861`. If your serial adapter differs, replace it with the actual device path of your SO-101 arm. You can find your hardware port using the `lerobot-find-port` utility.

## Gemini Live API & Robot Tool Capabilities

When interacting with the AI in the Web UI chat or via the agent controller, Gemini has access to real-time function calling tools:

| Tool Name | Description | Parameters |
| :--- | :--- | :--- |
| `move_to_pose` | Smoothly interpolate arm to 6-DOF joint target in degrees. | `qpos` (array of 6 degs), `duration_sec` |
| `set_joint_positions` | Instant snap joint position command in degrees. | `qpos` (array of 6 degs) |
| `open_gripper` | Fully open the gripper end-effector. | None |
| `close_gripper` | Fully close the gripper end-effector. | None |
| `move_to_preset` | Move arm to named keyframe (`HOME`, `MIDDLE`, `SWEEP_HOME`). | `pose_name` (string) |
| `move_to_spatial_point` | Ground 2D image coordinates $[y, x] \in [0, 1000]$ into 3D Cartesian coordinates via ArUco calibration and solve IK. | `point` (array of 2 numbers) |
| `execute_spav_pick_and_place` | Closed-loop multi-turn Sense-Plan-Act-Verify pick-and-place with vision self-correction. | `instruction` (string) |
| `get_joint_positions` | Read current joint angles in degrees. | None |
| `get_end_effector_pose` | Read 3D Cartesian position and orientation matrix (sim/twin). | None |

## Closed-Loop SPAV Pipeline & Workspace Calibration

### 1. ArUco Workspace Calibration & Sheet Generator (`robot/calibration.py`)
- Detects 4-corner ArUco fiducials (`DICT_4X4_50`, IDs 0..3) defining the physical workspace mat.
- Computes perspective homography mapping camera pixels directly to physical robot base millimeters.
- Configurable via `config/workspace_calibration.json`.
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

### 2. Quintic Minimum-Jerk Trajectory Planner (`robot/trajectory.py`)
- Implements $C^2$-continuous 5th-order polynomial interpolation: $s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5$.
- Guarantees zero velocity and zero acceleration at trajectory boundaries ($v(0)=v(T)=0$, $a(0)=a(T)=0$), completely eliminating motor jerk and torque spikes on Feetech STS3215 servos.
- Discrete 6-phase pick-and-place trajectories: Pre-grasp Hover $\rightarrow$ Linear Descent $\rightarrow$ Grasp $\rightarrow$ Retract $\rightarrow$ Transit to Place $\rightarrow$ Release.

### 3. Closed-Loop SPAV State Machine (`agent/spav.py`)
- **State 1 (Sense & Plan)**: Analyzes camera snapshot with `gemini-robotics-er-2-preview` to detect target object and destination.
- **State 2 (Act)**: Transforms coordinates via homography and executes minimum-jerk pick trajectory.
- **State 3 (Visual Verification)**: Captures post-grasp hover snapshot and verifies if object is securely in jaws.
- **State 4 (Self-Correction Decision)**: On success, transports to destination bin; on failure, applies visual delta offset $[\Delta y, \Delta x]$ and retries (up to 3 attempts).

## Testing & Diagnostics

Verify system health, MuJoCo simulation physics, inverse kinematics, and unit test suites:

```bash
# 1. Run complete automated unit test suite (Calibration, Trajectory, SPAV State Machine)
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
