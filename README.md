# ![GeminiRobotics-SO101 Banner](assets/docs/banner.jpg)

## GeminiRobotics-SO101 🤖

GeminiRobotics-SO101 connects [Google's Gemini Robotics multimodal models](https://ai.google.dev/gemini-api/docs/robotics-overview) (`gemini-robotics-er-2-streaming-preview`) to the low-cost 6-DOF [SO-101 robotic arm](https://huggingface.co/docs/lerobot/en/so101) via a unified web interface built with FastAPI, Gradio 6.0, OpenCV, and Three.js.

The user enters natural language instructions, and Gemini orchestrates pick-and-place tasks through continuous video streaming and real-time tool calling in both physical hardware and MuJoCo simulation.

<p align="center">
  <img src="assets/docs/new_ui.jpeg" alt="GeminiRobotics-SO101 Web UI" />
</p>

## Hardware

- **Arm**: Feetech STS3215-based SO-101 6-DOF robotic arm.
- **Camera**: 8MP Arducam  mounted ~45–60 cm above the tabletop looking down at the workspace.
- **Tabletop Mat**: Printable 4-corner ArUco calibration mat (`DICT_4X4_50`, IDs 1, 2, 3, 0).
- **Jaw Tag**: 25 mm ArUco marker (`DICT_4X4_50`, ID 10) mounted on the gripper fixed jaw for parallax-free visual closed-loop alignment.

## Installation

The project uses Python 3.13 and [`uv`](https://docs.astral.sh/uv/) for deterministic environment management:

```bash
git clone https://github.com/Ekumen-OS/GeminiRobotics-SO101.git
cd GeminiRobotics-SO101

# Simulation only (MuJoCo + Web UI)
uv sync --extra ui

# Real hardware (adds LeRobot and serial drivers)
uv sync --extra hw --extra ui

source .venv/bin/activate
```

## Configuration

Copy `.env.example` to `.env`. Only `GOOGLE_API_KEY` (or Vertex AI credentials) is required to run:

```bash
cp .env.example .env
```

All runtime settings (serial port, table surface height, camera selection, stream quality, and grasp margins) are documented directly in [`.env.example`](.env.example).

## Workspace Calibration

Spatial grounding maps image pixels $[y, x] \in [0, 1000]$ directly into physical robot base coordinates (mm) using perspective homography:

1. **Interactive Workspace Setup (`--setup`)**:
   Describe the workspace rectangle layout (**rectangle only**, axis-aligned with the robot: $+X$ forward along `HOME`, $+Y$ to robot's left; free-form or rotated layouts are not supported).
   ```bash
   uv run python main.py --setup
   ```
   The interactive wizard asks for measurements in mm to marker centres from the **shoulder-pan rotation axis**:
   - **near X**: distance forward from pan axis to the near marker pair;
   - **depth**: distance from the near pair to the far pair;
   - **width**: distance from the right pair to the left pair;
   - **lateral offset**: Y of the rectangle centre line ($0$ = centred on the robot, negative = to robot's right);
   - **marker size**.

   The model-derived $+38.8$ mm offset in X is added automatically to record corner coordinates in the robot base frame (`config/workspace_calibration.json`).
   The wizard evaluates Cartesian hover reachability across the rectangle: **markers outside reach are expected and fine**, because only the camera needs to see them.
   It generates an A4 printable placement guide diagram (`assets/aruco_markers/workspace_placement_guide.png`), marker PNG tiles (`marker_{0..3,10}.png`), and `jaw_tag_sheet.png`. Use `--reuse` to update dimensions without regenerating files, `--regenerate` to overwrite, or `--non-interactive` to use defaults.

   ![Workspace Placement Guide](assets/docs/placement_guide.png)

   To verify calibration, start the Web UI (`uv run python main.py`) and enable "📐 Show Calibration Overlay" (`ui/components/vision.py`) to verify the registration on the live camera stream. Also check the `[Calibration] [CAMERA]` `height_mm` log against your physical lens height measurement.
2. **3D-printed tags with `tools/aruco_stl.py`**:
   Generates watertight binary STL meshes for single-color grooved printing or Bambu Studio / AMS dual-color (`_white.stl` + `_black.stl`) tiles:
   ```bash
   uv run python -m tools.aruco_stl --help
   ```
   Key CLI options:
   - `--marker-id`: Marker ID (default: 10 for jaw tag).
   - `--marker-size`: Active marker square side in mm (default: 19.5312 for a 25.0 mm outer tile).
   - `--margin`: Quiet-zone border per side in mm (default: 2.73438 -> 25.0 mm total outer tile).
   - `--thickness`: Total tile thickness in mm (default: 1.5).
   - `--groove-depth`: Recess depth in mm (default: 0.6 = 3 layers at 0.2 mm).
   - `--dictionary`: OpenCV dictionary (default: `DICT_4X4_50`).
   - `--all-workspace-markers`: Also generates 40 mm table corner tags (IDs 0, 1, 2, 3).
3. **Jaw tag**:
   Mount the 25 × 25 × 1.5 mm ID 10 tile (white border included) flush on the camera-facing flat face of the fixed-jaw housing (`80 mm` from the fixed fingertip line along the jaw toward the wrist, `9.5 mm` from the fixed fingertip across the jaw toward the moving jaw). Default offset is `GRIPPER_TAG_LOCAL_OFFSET_MM=-25.4,25.5,78.6` (tag centre minus grasp site in gripper coordinates). This value was tuned on the real gripper: the caliper estimate was 12 mm off along the jaw axis.
4. **Save calibration**:
   In the Web UI, open the **ArUco Workspace Calibration** accordion and click **💾 Save Calibration** once all 4 corner markers are detected.
5. **Real-arm grasp calibration**:
   - Run `close_gripper` on air: note logged aperture $A_{\text{empty}}$ (e.g. $-74.5^\circ$) and set `GRIPPER_EMPTY_APERTURE_DEG`.
   - Run `close_gripper` on a 30 mm cube: note stalled aperture $A_{\text{object}}$ and set `GRASP_EMPTY_MARGIN_DEG` to $\approx 0.40 \times (A_{\text{object}} - A_{\text{empty}})$.

## Preflight Diagnostics

Before starting the server, run the non-moving preflight check to verify dependencies, credentials, configs, port availability, and hardware access:

```bash
# Verify simulation environment (MuJoCo scene, IK solver, headless offscreen render)
uv run python main.py --mode sim --task check_environment

# Verify real hardware (serial bus permissions, motor calibration, camera capture)
uv run python main.py --mode real --task check_environment
```

Each check outputs `[OK]`, `[WARN]`, or `[FAIL]` with fix hints. Exits 0 on success, 1 on failure.

## Running the Web UI

Launch the Unified FastAPI + Gradio Web UI at `http://localhost:7860/`:

```bash
# Simulation mode (MuJoCo physics + Three.js browser viewer)
uv run python main.py --mode sim

# Real robot mode (physical arm + live UVC camera MJPEG feed)
uv run python main.py --mode real

# Override serial port explicitly (defaults to ROBOT_PORT from .env)
uv run python main.py --mode real --port /dev/ttyACM0
```

The interface features live MJPEG camera streaming (`/api/stream`), conversational prompt execution with Gemini Live, manual joint and Cartesian jog controls, and an interactive 3D Three.js digital twin synchronized to arm telemetry at 30 Hz.

## Model Tools

Gemini Live calls these 6 primitives (defined in [`agent/tools.py`](agent/tools.py)):

| Tool | Parameters | Description |
|---|---|---|
| `move_to_spatial_point` | `point: [y, x]`, `z_height_cm: float = 3.0` | Move gripper to image coordinate $[0, 1000]$ at table height $z$. |
| `open_gripper` | *(none)* | Open gripper jaws fully. |
| `close_gripper` | *(none)* | Close jaws to grasp; returns aperture and grasp classification. |
| `move_to_preset` | `pose_name: str` | Move arm to named keyframe (`HOME` or `OBSERVE`). |
| `move_to_pose` | `qpos: [deg, ...]`, `duration_sec: float` | Interpolate arm joints directly to target angles. |
| `get_joint_positions` | *(none)* | Query current arm joint angles in degrees. |

## Troubleshooting

- **Serial device permissions (Linux)**: Ensure read/write access to `/dev/ttyACM0`:
  ```bash
  sudo chmod 666 /dev/ttyACM0            # Immediate temporary fix
  sudo usermod -a -G dialout $USER       # Permanent fix (requires re-login)
  ```
- **Camera busy / V4L2 device locks**: Ensure no other application (Zoom, OBS, browser) is using the USB camera index configured in `CAMERA_INDEX`. List attached video devices with `v4l2-ctl --list-devices`.
- **Headless rendering (Linux / SSH)**: MuJoCo uses EGL for offscreen rendering when `DISPLAY` is unset. Ensure NVIDIA/EGL drivers are present.
- **"Workspace is not calibrated"**: Place the 4-corner calibration mat in camera view, check `/api/stream`, and click **💾 Save Calibration** in the UI accordion.

## Development

```bash
# Install all development extras and pre-commit hooks
uv sync --all-extras
uv run pre-commit install

# Run unit and integration tests
uv run pytest tests/ -v

# Run physics-heavy slow tests (seeded MuJoCo pick-and-place harness)
uv run pytest -m slow -s

# Code formatting, linting, type-checking, and import contracts
uv run ruff check --fix .
uv run ruff format .
uv run mypy
uv run lint-imports
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for data flows, subsystem interfaces, and invariants, and [AGENTS.md](AGENTS.md) for coding guidelines.
