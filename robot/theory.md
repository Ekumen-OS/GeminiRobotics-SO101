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

In our practical implementation, we decouple the spatial IK solver to optimize primary **3D Position Tracking** ($3 \times 5$ full-rank Jacobian) while utilizing the remaining 2D internal null-space to pull joint postures smoothly toward an ergonomic workspace anchor (`SWEEP_HOME`).

> 💡 **Verifying Kinematics & Motion**: 
> You can verify the integrity of the kinematics solver, MuJoCo simulation scene, and linear Cartesian motion tracking at any time by executing the dedicated test command:
> ```bash
> uv run python main.py --task test_simulation
> ```
