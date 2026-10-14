# SO-101 Robot Arm Kinematics Theory

This document explains the mathematical foundations for understanding the motion and positioning of the 6-Degrees of Freedom (DOF) SO-101 Robot Arm used in the GeminiRobotics-SO101 project.

## Table of Contents

1. [Introduction to Robot Kinematics](#1-introduction-to-robot-kinematics)
2. [Robot Architecture and Joints](#2-robot-architecture-and-joints)
3. [Forward Kinematics (FK)](#3-forward-kinematics-fk)
    - [Denavit-Hartenberg (DH) Convention](#denavit-hartenberg-dh-convention)
    - [Transformation Matrices](#transformation-matrices)
4. [Inverse Kinematics (IK)](#4-inverse-kinematics-ik)
    - [Analytical vs Numerical IK](#analytical-vs-numerical-ik)
    - [Addressing Under-Actuation and Null-Space Projection](#addressing-under-actuation-and-null-space-projection)

---

## 1. Introduction to Robot Kinematics

**Kinematics** is the study of motion without considering the forces that cause it. In robotics, kinematics relates the joint variables (angles of the servo motors) to the position and orientation of the end-effector (the gripper) in 3D Cartesian space (X, Y, Z coordinates plus roll, pitch, yaw angles).

- **Forward Kinematics (FK):** Given the joint angles ($\theta_1, \theta_2, ..., \theta_6$), what is the position and orientation of the gripper?
- **Inverse Kinematics (IK):** Given a desired position and orientation for the gripper, what should the joint angles be to reach it?

## 2. Robot Architecture and Joints

The SO-101 is a 6-DOF robotic manipulator using Feetech STS3215 servos. Its structure mimics a human arm:

1. **Base (Shoulder Pan):** Rotates the entire arm around the vertical (Z) axis.
2. **Shoulder Lift:** Moves the upper arm up and down.
3. **Elbow Flex:** Bends the forearm relative to the upper arm.
4. **Wrist Flex:** Pitches the wrist up and down.
5. **Wrist Roll:** Rotates the gripper around the forearm axis.
6. **Gripper:** The end-effector that opens and closes (technically a 6th degree of freedom, but treated separately from the spatial positioning of the wrist).

## 3. Forward Kinematics (FK)

The standard way to compute Forward Kinematics is by using the **Denavit-Hartenberg (DH) Convention**. This method assigns a coordinate frame to each link of the robot and defines four parameters to transform from one frame to the next.

### Denavit-Hartenberg (DH) Convention

The four DH parameters are:

- $a_i$ (Link length): Distance along $X_i$ from $Z_{i-1}$ to $Z_i$.
- $\alpha_i$ (Link twist): Angle around $X_i$ from $Z_{i-1}$ to $Z_i$.
- $d_i$ (Link offset): Distance along $Z_{i-1}$ from $X_{i-1}$ to $X_i$.
- $\theta_i$ (Joint angle): Angle around $Z_{i-1}$ from $X_{i-1}$ to $X_i$.

*Note: The exact physical link lengths ($a$ and $d$) for your specific SO-101 build may vary slightly based on 3D printed tolerances.*

### Transformation Matrices

The homogeneous transformation matrix $A_i$ from frame $i-1$ to frame $i$ is calculated using the DH parameters:

$$ 
A_i = \begin{bmatrix}
\cos\theta_i & -\sin\theta_i\cos\alpha_i & \sin\theta_i\sin\alpha_i & a_i\cos\theta_i \\
\sin\theta_i & \cos\theta_i\cos\alpha_i & -\cos\theta_i\sin\alpha_i & a_i\sin\theta_i \\
0 & \sin\alpha_i & \cos\alpha_i & d_i \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

The overall transformation from the base frame to the end-effector frame ($T_{0}^6$) is the product of all individual link transformation matrices:

$$ T_{0}^6 = A_1 \cdot A_2 \cdot A_3 \cdot A_4 \cdot A_5 \cdot A_6 $$

The resulting $4 \times 4$ matrix contains the Rotation matrix $R$ ($3 \times 3$) and the Position vector $P$ ($3 \times 1$) of the end-effector in the base frame:

$$ 
T_{0}^6 = \begin{bmatrix} R & P \\ 0 & 1 \end{bmatrix} 
$$

## 4. Inverse Kinematics (IK)

Inverse kinematics is non-linear and can have zero solutions (target out of reach), one solution, or multiple solutions (e.g., reaching the same point with elbow "up" or elbow "down").

### Analytical vs Numerical IK

1. **Analytical IK:** Involves finding closed-form algebraic equations for $\theta_1$ through $\theta_6$.
2. **Numerical IK (Damped Least Squares & Null-Space Projection):** Uses iterative optimization algorithms to converge on joint angle configurations by stepping along spatial velocity gradients. In GeminiRobotics-SO101, we implement robust **Levenberg-Marquardt Damped Least Squares (DLS)** combined with secondary-task **Null-Space Posture Regularization** to avoid kinematic singularities and prevent arm collapse.

### Addressing Under-Actuation and Null-Space Projection

While the SO-101 command vector contains 6 motor inputs, anatomical analysis reveals that only **5 articulated spatial joints** contribute to positioning and orienting the wrist frame:

1. **Base (Shoulder Pan)**: $Z$-axis rotation
2. **Shoulder Lift**: $Y$-axis elevation
3. **Elbow Flex**: $Y$-axis pitch
4. **Wrist Flex**: $Y$-axis pitch
5. **Wrist Roll**: $X/Z$ tool axial twist

The 6th actuator drives a single-hinge opening jaw on the gripper assembly and does not shift the coordinate frame triad (`gripperframe`). Consequently, enforcing full 6-DOF tracking ($\mathbb{R}^6$) on a 5-DOF arm introduces over-determined numerical deadlocks during linear Cartesian moves. 

In our practical implementation, we decouple the spatial IK solver to optimize primary **3D Position Tracking** ($3 \times 5$ full-rank Jacobian) while utilizing the remaining 2D internal null-space to pull joint postures smoothly toward an ergonomic nominal posture (`nominal_qpos`, regularizing elbow and shoulder pitch). Furthermore, if the solver encounters local minima near joint bounds ($\text{error} > 1\,\text{cm}$), it automatically recovers by retrying from a neutral workspace seed.

## 5. Visual Grounding & Perspective Homography Calibration

To enable Gemini VLA models to interact with objects seen via the camera feed (in both real physical mode and MuJoCo simulation), the system maps 2D visual detections into physical 3D end-effector targets via the `move_to_spatial_point` tool adapter:

1. **Normalized Image Coordinate Frame**: The model identifies target objects using normalized coordinates $[y, x] \in [0, 1000]$, where $(0, 0)$ is the top-left corner and $(1000, 1000)$ is the bottom-right corner.
2. **Perspective Homography Transformation**:
   - **Real Physical Mode**: When 4 physical ArUco workspace markers (`DICT_4X4_50`, IDs 0..3) are detected in view, OpenCV computes a 3x3 projective homography matrix $H_{\text{real}} \in \mathbb{R}^{3 \times 3}$:
   - **Real Physical Mode**: When 4 physical ArUco workspace markers (`DICT_4X4_50`, canonical IDs: Top-Left 1, Top-Right 2, Bottom-Right 3, Bottom-Left 0) are detected in view, OpenCV computes a 3x3 projective homography matrix $H_{\text{real}} \in \mathbb{R}^{3 \times 3}$:
     $$\begin{bmatrix} X_{\text{mm}} \\ Y_{\text{mm}} \\ 1 \end{bmatrix} \sim H \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$
     The video stream continuously tracks visible markers in real time, automatically updating $H_{\text{real}}$ to account for camera tilt, focal distance, and perspective skew, converting pixel coordinates directly into millimeter coordinates in the robot base frame.
     The video stream continuously tracks visible markers in real time, automatically updating $H_{\text{real}}$ to account for camera tilt, focal distance, and perspective skew, converting pixel coordinates directly into millimeter coordinates in the robot base frame with lateral parity guaranteed.
   - **Simulation Mode**: Uses **Synthetic Homography Calibration** (`simulation/sim_calibration.py`) where the 4 known 3D workspace corners are projected analytically through the virtual MuJoCo camera (`workspace_cam`, $fovy=54^\circ$) to compute an exact homography $H_{\text{sim}} \in \mathbb{R}^{3 \times 3}$ with sub-millimeter precision.
3. **Variable Height Control**: The `z_height_cm` parameter allows the model to command target heights above the tabletop surface:
   - $Z = 0.00\,\text{m}$ ($0\,\text{cm}$): Table surface contact for grasping.
   - $Z = 0.03 - 0.05\,\text{m}$ ($3-5\,\text{cm}$): Pre-grasp hover and transit clearance.
4. **Visual Bounding Box & Target Overlay**: When a spatial target point is selected or predicted by Gemini, the system projects the 2D image coordinate and renders a translucent bounding box and coordinate tag directly onto the observation video stream.
5. **IK Resolution & Execution**: The mapped $[X, Y, Z]$ coordinate is dispatched to `CartesianIK.solve_ik()`, which computes target joint angles $\mathbf{q}^*$, validates them against `JOINT_LIMITS`, and executes smooth motion via `RobotInterface.move_to_pose()`.

## 6. Smooth S-Curve Motion Interpolation

To prevent torque-off faults and eliminate mechanical chattering on 3D-printed joints and Feetech STS3215 servos, joint motions are interpolated using smooth cosine S-curve profiles:

$$\alpha(t) = \frac{1}{2} \left( 1 - \cos(\pi \cdot t) \right), \quad t = \frac{\text{step}}{\text{num\_steps}} \in [0, 1]$$
$$\mathbf{q}(t) = (1 - \alpha(t)) \cdot \mathbf{q}_{\text{start}} + \alpha(t) \cdot \mathbf{q}_{\text{target}}$$

This profile satisfies zero-velocity boundary conditions:

- $\alpha(0) = 0, \quad \alpha(1) = 1$
- $\dot{\alpha}(0) = 0, \quad \dot{\alpha}(1) = 0$

Guaranteeing smooth acceleration and deceleration curves without sudden jerk spikes on actuator gearboxes.

## 7. Closed-Loop Post-Action Observation & Visual Verification

To achieve robust visual manipulation in real-world environments with friction and gripper-object dynamics, the system operates as a closed-loop controller:

1. **Post-Action State Capture**: After executing any blocking action (e.g., `close_gripper`, `move_to_spatial_point`), the controller introduces a $0.3\,\text{s}$ stabilization delay to let actuators settle and eliminate motion blur, then captures and transmits a post-action observation frame to the VLA model.
2. **Visual Grasp Verification**: The model inspects the frame to verify object containment between the gripper jaws and detect any post-contact displacement.
3. **Homography-Guided Self-Correction**: If a grasp attempt fails (object remains ungrasped on the tabletop), the model calculates visual offset corrections in normalized coordinate space $[y, x]$ and autonomously triggers a retry sequence (up to 3 attempts) without human intervention.
2. **Programmatic Perception Metrics Engine (`robot/perception.py`)**: Computes ground-truth offsets between the locked target centre and the jaw in the table plane, in robot millimetres ($d_x$, $d_y$, distance, and `is_aligned` within `ALIGNMENT_TOLERANCE_MM`), returning them directly in tool payloads to eliminate open-loop hallucination. Parallax of the raised jaw tag at height $h$ is removed with the camera centre $C$ recovered from the homography: $xy = C_{xy} + (H(u,v) - C_{xy})(C_z - h)/C_z$.
3. **Grasp Verification**: After a straight-up lift the gripper sensor re-read decides; vision can only veto, when a blob of the locked object's colour still overlaps its table footprint (`vision_vote`, `decide_grasp`).
4. **Homography-Guided Self-Correction & Escalation**: If misalignment exceeds grasping tolerance or a grasp attempt fails (object remains on the table), the controller enforces corrective micro-adjustments or automated retry cycles (up to 3 attempts) before escalating to the user.

> 💡 **Verifying Kinematics & Motion**: 
> You can verify the integrity of the kinematics solver, MuJoCo simulation scene, and motion tracking at any time by executing:
>
> ```bash
> uv run pytest tests/ -v
> ```
