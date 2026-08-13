"""Controls component for joint sliders and reset buttons."""

import gradio as gr
import numpy as np

from robot.constants import JOINT_LIMITS, PRESET_POSES
from robot.interface import RobotInterface
from ui.components.telemetry import read_measured_joint_degrees, render_joint_state_panel
from ui.state import set_target_joints_deg


def build_controls_interface(
    robot: RobotInterface | None, mode: str, init_deg: list[float]
) -> tuple[list[gr.Slider], gr.HTML]:
    """Build the robot movement control sliders.

    Args:
        robot: Initialized robot hardware or simulation controller instance.
        mode: Execution operational mode ('real', 'sim', or 'twin').
        init_deg: Initial joint angles in degrees.
    """
    gr.Markdown("### 🎮 Robot Movement Control")

    with gr.Row():
        s_pan = gr.Slider(
            minimum=JOINT_LIMITS["shoulder_pan"][0],
            maximum=JOINT_LIMITS["shoulder_pan"][1],
            step=0.5,
            value=init_deg[0],
            label="Shoulder Pan (°)",
        )
        s_lift = gr.Slider(
            minimum=JOINT_LIMITS["shoulder_lift"][0],
            maximum=JOINT_LIMITS["shoulder_lift"][1],
            step=0.5,
            value=init_deg[1],
            label="Shoulder Lift (°)",
        )
        e_flex = gr.Slider(
            minimum=JOINT_LIMITS["elbow_flex"][0],
            maximum=JOINT_LIMITS["elbow_flex"][1],
            step=0.5,
            value=init_deg[2],
            label="Elbow Flex (°)",
        )
    with gr.Row():
        w_flex = gr.Slider(
            minimum=JOINT_LIMITS["wrist_flex"][0],
            maximum=JOINT_LIMITS["wrist_flex"][1],
            step=0.5,
            value=init_deg[3],
            label="Wrist Flex (°)",
        )
        w_roll = gr.Slider(
            minimum=JOINT_LIMITS["wrist_roll"][0],
            maximum=JOINT_LIMITS["wrist_roll"][1],
            step=0.5,
            value=init_deg[4],
            label="Wrist Roll (°)",
        )
        grip = gr.Slider(
            minimum=JOINT_LIMITS["gripper"][0],
            maximum=JOINT_LIMITS["gripper"][1],
            step=0.5,
            value=init_deg[5],
            label="Gripper (°)",
        )

    with gr.Row():
        status_display = gr.Textbox(
            label="Motion Status",
            value="System initialized. Move sliders to command joints.",
            interactive=False,
            scale=4,
        )
        reset_btn = gr.Button("🏠 Reset to HOME Pose", variant="secondary", scale=1)

    joint_state_panel = gr.HTML(
        value=render_joint_state_panel(init_deg, read_measured_joint_degrees(robot, init_deg))
    )

    sliders = [s_pan, s_lift, e_flex, w_flex, w_roll, grip]

    def apply_joint_angles(j1: float, j2: float, j3: float, j4: float, j5: float, j6: float):
        target_deg = [j1, j2, j3, j4, j5, j6]
        set_target_joints_deg(target_deg)
        target_qpos = np.radians(target_deg, dtype=np.float32)
        if robot is not None:
            robot.set_joint_positions(target_qpos)
            if mode != "sim":
                robot.step(1)
        msg = f"⚡ Command routed to servos: {[round(val, 1) for val in target_deg]}°"
        measured_deg = read_measured_joint_degrees(robot, target_deg)
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
            return gr.skip()

        try:
            curr_rad = robot.get_joint_positions()
            curr_deg = [round(float(v), 1) for v in np.degrees(curr_rad)]
            return render_joint_state_panel(curr_deg, curr_deg)
        except Exception:
            return gr.skip()

    sync_timer.tick(
        fn=sync_robot_state_html,
        inputs=None,
        outputs=[joint_state_panel],
    )

    def reset_to_home():
        home_q = PRESET_POSES["HOME"]
        home_deg = [round(float(np.degrees(x)), 1) for x in home_q]
        set_target_joints_deg(home_deg)
        if robot is not None:
            robot.move_to_pose(home_q, duration_sec=1.0)
            if mode != "sim":
                robot.step(1)
        msg = f"🏠 Moving arm back to HOME pose: {home_deg}°"
        measured_deg = read_measured_joint_degrees(robot, home_deg)
        panel_html = render_joint_state_panel(home_deg, measured_deg)
        return [*home_deg, msg, panel_html]

    reset_btn.click(
        fn=reset_to_home,
        inputs=None,
        outputs=[*sliders, status_display, joint_state_panel],
        queue=False if mode == "sim" else True,
    )

    return sliders, joint_state_panel
