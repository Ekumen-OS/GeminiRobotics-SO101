# GeminiRobotics-SO101

Welcome to **GeminiRobotics-SO101** 🤖

This repository combines Google's Gemini Robotics AI models and capabilities with the **SO-101 Robotic Arm** setup (building upon the simplified physical assembly, calibration, kinematics, and simulation tools from LeDroid-101 and HuggingFace LeRobot).

We use Python version **3.13.7** and the extremely fast and deterministic package manager `uv` for managing all environments and dependencies.

> [!IMPORTANT]
> This repo is actively under development as we integrate intelligent robotics behaviors driven by Gemini Robotics!

## Architecture & Modules

To keep code modular and hardware/simulation concerns cleanly isolated, the repository is structured into two main packages:

1. **[`/robot`](robot/) (Core Robot Definition)**
   - **Constants & Poses**: Definitional parameters, joint names, actuator names, and standard target postures ([`constants.py`](robot/constants.py)).
   - **Kinematics & Theory**: Forward kinematics ([`kinematics.py`](robot/kinematics.py)), numerical damped least-squares Inverse Kinematics ([`cartesian_ik.py`](robot/cartesian_ik.py)), and mathematical theoretical foundations ([`theory.md`](robot/theory.md)).
   - **Hardware & Digital Twin**: Physical serial bus interfaces for Feetech servos ([`real_robot.py`](robot/real_robot.py)) and simultaneous simulation-to-hardware streaming ([`twin_robot.py`](robot/twin_robot.py)).

2. **[`/simulation`](simulation/) (MuJoCo Physics & Environments)**
   - **Physics Simulation**: High-level MuJoCo model manipulation ([`sim_robot.py`](simulation/sim_robot.py)) and passive 3D interactive rendering ([`simulate.py`](simulation/simulate.py)).
   - **Reinforcement Learning & Warp**: Gymnasium reach target environments ([`env.py`](simulation/env.py)) and hardware-accelerated batched simulations using NVIDIA Warp ([`warp_env.py`](simulation/warp_env.py)).
   - **Validation & Test Suites**: Diagnostic verifiers including linear Cartesian sweeps ([`test_movement.py`](simulation/test_movement.py)) and system health checks ([`check_env.py`](simulation/check_env.py)).

## Getting Started

### Requirements

* Python 3.13.7
* `uv` package manager ([Instructions here](https://docs.astral.sh/uv/getting-started/installation/))
* Clone this repository:

```bash
git clone https://github.com/johannstark/GeminiRobotics-SO101.git
cd GeminiRobotics-SO101
```

* Setup the repository environment and install dependencies:

```bash
uv sync
source .venv/bin/activate
```

* Use the LeRobot package to find the serial port of your physical SO-101 robotic arm and calibrate it, using [this guide](https://huggingface.co/docs/lerobot/so101).
* You are now ready to execute simulations and physical robot controls!

## Running the Robot (`main.py`)

The [`main.py`](main.py) script is the unified entry point for running the SO-101 robotic arm in MuJoCo simulation, digital twin mode, real-world control, or system verification. By default, launching in `sim` mode instantly boots an interactive 3D scene where you can control the arm freely using the MuJoCo UI Actuator sliders!

```bash
# Instantly boot the interactive MuJoCo 3D simulation scene (default mode & task)
uv run python main.py --mode sim

# Run automated linear Cartesian motion sweeps to test movement & kinematics
uv run python main.py --task test_simulation

# Run comprehensive system health diagnostics and simulation speed benchmarks
uv run python main.py --task check_environment
```

> [!NOTE]
> The `--port` argument is required for ***twin*** and ***real*** modes. Replace `<serial_port>` with the actual USB device path of your SO-101 arm (e.g., `/dev/tty.usbmodem1201`).
> Find your serial port using the `lerobot-find-port` command.

```bash
# Connect to physical arm for interactive real-world control
uv run python main.py --mode real --port /dev/tty.usbmodem1201

# Run Digital Twin mode (simulated joint commands mirrored directly to real servos)
uv run python main.py --mode twin --port /dev/tty.usbmodem1201
```

## Testing & Movement Validation

Whenever you want to test that the robot movement, inverse kinematics solver, MuJoCo physics, and robot definitions are working properly, invoke our built-in test commands:

```bash
# 1. Verify movement & kinematics via automated X, Y, and Z Cartesian line sweeps
uv run python main.py --task test_simulation

# 2. Check system diagnostics, dependency imports, and MuJoCo FPS throughput
uv run python main.py --task check_environment
```

## Kinematics & Mathematics Reference

For a complete explanation of Denavit-Hartenberg (DH) parameters, transformation matrices, spatial Jacobians, and under-actuated 5-DOF damped least-squares Inverse Kinematics, consult our documentation:
* [SO-101 Robot Arm Kinematics Theory (`robot/theory.md`)](robot/theory.md)

---

Made with ❤️ for Robotics and AI
