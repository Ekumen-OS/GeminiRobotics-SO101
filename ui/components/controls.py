"""Controls component for joint sliders and reset buttons."""

import gradio as gr
import numpy as np

from robot.constants import JOINT_LIMITS
from robot.errors import RoboticsError
from robot.interface import RobotInterface
from ui.components.telemetry import read_measured_joint_degrees, render_joint_state_panel
from ui.state import set_target_joints_deg


def build_controls_interface(
    robot: RobotInterface | None, mode: str, init_deg: list[float]
) -> tuple[list[gr.Slider], gr.HTML]:
    """Build the robot movement control sliders and telemetry panels in accordions.

    Args:
        robot: Initialized robot hardware or simulation controller instance.
        mode: Execution operational mode ('real' or 'sim').
        init_deg: Initial joint angles in degrees.

    Returns:
        Tuple containing the list of 6 joint sliders and the HTML telemetry component.
    """
    is_online = True
    if mode == "real":
        is_online = getattr(robot, "connected", True) if robot is not None else False

    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            with gr.Accordion("🎮 Robot Movement Control", open=False):
                with gr.Row():
                    s_pan = gr.Slider(
                        minimum=JOINT_LIMITS["shoulder_pan"][0],
                        maximum=JOINT_LIMITS["shoulder_pan"][1],
                        step=0.5,
                        value=init_deg[0],
                        label="Shoulder Pan (°)",
                        interactive=is_online,
                    )
                    s_lift = gr.Slider(
                        minimum=JOINT_LIMITS["shoulder_lift"][0],
                        maximum=JOINT_LIMITS["shoulder_lift"][1],
                        step=0.5,
                        value=init_deg[1],
                        label="Shoulder Lift (°)",
                        interactive=is_online,
                    )
                    e_flex = gr.Slider(
                        minimum=JOINT_LIMITS["elbow_flex"][0],
                        maximum=JOINT_LIMITS["elbow_flex"][1],
                        step=0.5,
                        value=init_deg[2],
                        label="Elbow Flex (°)",
                        interactive=is_online,
                    )
                with gr.Row():
                    w_flex = gr.Slider(
                        minimum=JOINT_LIMITS["wrist_flex"][0],
                        maximum=JOINT_LIMITS["wrist_flex"][1],
                        step=0.5,
                        value=init_deg[3],
                        label="Wrist Flex (°)",
                        interactive=is_online,
                    )
                    w_roll = gr.Slider(
                        minimum=JOINT_LIMITS["wrist_roll"][0],
                        maximum=JOINT_LIMITS["wrist_roll"][1],
                        step=0.5,
                        value=init_deg[4],
                        label="Wrist Roll (°)",
                        interactive=is_online,
                    )
                    grip = gr.Slider(
                        minimum=JOINT_LIMITS["gripper"][0],
                        maximum=JOINT_LIMITS["gripper"][1],
                        step=0.5,
                        value=init_deg[5],
                        label="Gripper (°)",
                        interactive=is_online,
                    )

                with gr.Row():
                    status_display = gr.Textbox(
                        label="Motion Status",
                        value=(
                            "System initialized. Move sliders to command joints."
                            if is_online
                            else "⚠️ Hardware Offline. Arm not connected."
                        ),
                        interactive=False,
                    )

        with gr.Column(scale=1, min_width=320):
            with gr.Accordion("📊 Joint Telemetry", open=False):
                measured_initial = read_measured_joint_degrees(robot) if is_online else None
                joint_state_panel = gr.HTML(
                    value=render_joint_state_panel(init_deg, measured_initial)
                )

    sliders = [s_pan, s_lift, e_flex, w_flex, w_roll, grip]

    def apply_joint_angles(j1: float, j2: float, j3: float, j4: float, j5: float, j6: float):
        target_deg = [j1, j2, j3, j4, j5, j6]
        if robot is not None and not getattr(robot, "connected", True):
            return (
                "⚠️ Robot is disconnected. Command dropped.",
                render_joint_state_panel(target_deg, None),
            )
        set_target_joints_deg(target_deg)
        target_qpos = np.radians(target_deg, dtype=np.float32)
        try:
            if robot is not None:
                robot.set_joint_positions(target_qpos)
            msg = f"⚡ Command routed to servos: {[round(val, 1) for val in target_deg]}°"
        except (RoboticsError, OSError, RuntimeError) as e:
            msg = f"⚠️ Command failed: {e}"
        measured_deg = read_measured_joint_degrees(robot)
        panel_html = render_joint_state_panel(target_deg, measured_deg)
        return msg, panel_html

    for s in sliders:
        s.release(
            fn=apply_joint_angles,
            inputs=sliders,
            outputs=[status_display, joint_state_panel],
            queue=False if mode == "sim" else True,
        )

    # Sync ONLY the HTML panel continuously with actual robot state using Gradio Timer
    sync_timer = gr.Timer(value=0.2, active=True)

    def sync_robot_state_html():
        if robot is None or not hasattr(robot, "get_joint_positions"):
            return render_joint_state_panel(init_deg, None)

        try:
            curr_rad = robot.get_joint_positions()
            curr_deg = [round(float(v), 1) for v in np.degrees(curr_rad)]
            if hasattr(robot, "get_commanded_joint_positions"):
                cmd_rad = robot.get_commanded_joint_positions()
                commanded_deg = [round(float(v), 1) for v in np.degrees(cmd_rad)]
            else:
                commanded_deg = curr_deg
            return render_joint_state_panel(commanded_deg, curr_deg)
        except (RoboticsError, OSError, RuntimeError):
            return render_joint_state_panel(init_deg, None)

    sync_timer.tick(
        fn=sync_robot_state_html,
        inputs=None,
        outputs=[joint_state_panel],
    )

    return sliders, joint_state_panel
