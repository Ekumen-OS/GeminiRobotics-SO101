"""Chat component for VLA instruction integration."""

import logging
from typing import Any

import gradio as gr
import numpy as np

from agent.vla_controller import VLAController
from robot.errors import CalibrationUnavailable
from ui.state import get_ui_joint_positions
from vision.calibrator import get_calibrator
from vision.frames import get_latest_raw_jpeg, get_raw_jpeg_after

logger = logging.getLogger(__name__)

READY_PROMPT_PLACEHOLDER = (
    'e.g., "Pick up red cube and place in sorting basket..." (Press Enter to Send)'
)


def build_chat_interface(controller: VLAController, api_available: bool = True) -> tuple:
    """Build the Chatbot interface within the current Gradio context.

    Args:
        controller: Initialized VLAController instance.
        api_available: Whether the Gemini API is available.

    Returns:
        Tuple of (prompt_input, chatbot, status_banner, exec_btn, home_btn, clear_btn, rand_btn)
    """
    calib = get_calibrator()
    is_calibrated = calib is not None and calib.is_calibrated if controller.mode == "real" else True

    if not is_calibrated:
        initial_msg = (
            "⚠️ **Workspace Not Calibrated**\n\n"
            f"Ready in `{controller.mode.upper()}` mode. Physical workspace calibration is "
            "required before executing spatial commands.\n\n"
            "👉 Please place the 4 ArUco markers in view and click **'💾 Save Calibration'** "
            "under the **🎯 ArUco Workspace Calibration** panel."
        )
    else:
        initial_msg = (
            f"🤖 **System Active**: Ready in `{controller.mode.upper()}` mode with calibrated "
            "workspace. Type commands below to invoke spatial reasoning."
        )

    chatbot = gr.Chatbot(
        label="Gemini Robotics Assistant (gemini-robotics-er)",
        height=600,
        elem_id="vla-chatbot",
        value=[
            {
                "role": "assistant",
                "content": initial_msg,
            }
        ],
    )
    if not api_available:
        gr.Markdown("⚠️ **Gemini API unavailable — manual control only.** Check your credentials.")

    if not api_available:
        initial_status = "⚠️ **Status: Gemini API Unavailable**"
    elif not is_calibrated:
        initial_status = "⚠️ **Status: Workspace Not Calibrated**"
    else:
        initial_status = "🟢 **Status: Ready**"

    status_banner = gr.Markdown(
        value=initial_status,
        elem_id="vla-status-banner",
    )

    prompt_input = gr.Textbox(
        label="Instruction Prompt",
        placeholder=(
            READY_PROMPT_PLACEHOLDER
            if is_calibrated
            else "⚠️ Calibrate workspace before issuing commands..."
        ),
        lines=1,
        max_lines=4,
        autofocus=True,
        interactive=api_available,
        elem_id="vla-prompt",
    )
    with gr.Row(equal_height=True, elem_classes=["chat-action-row"]):
        exec_btn = gr.Button(
            "🚀 Send",
            variant="primary",
            scale=1,
            interactive=api_available,
            elem_classes=["chat-action-btn"],
        )
        home_btn = gr.Button(
            "🏠 Reset to Home",
            variant="secondary",
            scale=1,
            elem_classes=["chat-action-btn"],
        )
        rand_btn = (
            gr.Button(
                "🎲 Randomize Cube",
                variant="secondary",
                scale=1,
                elem_classes=["chat-action-btn"],
            )
            if controller.mode == "sim"
            else None
        )
        clear_btn = gr.Button(
            "🗑️ Clear Chat",
            variant="secondary",
            scale=1,
            elem_classes=["chat-action-btn"],
        )

    return prompt_input, chatbot, status_banner, exec_btn, home_btn, clear_btn, rand_btn


def wire_chat_events(
    controller: VLAController,
    prompt_input: gr.Textbox,
    chatbot: gr.Chatbot,
    status_banner: gr.Markdown,
    exec_btn: gr.Button,
    home_btn: gr.Button,
    clear_btn: gr.Button,
    sliders: list[gr.Slider],
    rand_btn: gr.Button | None = None,
) -> dict[str, Any]:
    """Wire up the event listeners for the chat interface.

    Args:
        controller: The VLAController instance.
        prompt_input: The Gradio Textbox for user prompts.
        chatbot: The Gradio Chatbot component.
        status_banner: The Gradio Markdown component for status indication.
        exec_btn: The execute/send button.
        home_btn: The reset to home button.
        clear_btn: The clear chat button.
        sliders: The list of 6 Gradio joint sliders to update.
        rand_btn: Optional randomize target button for sim mode.

    Returns:
        Dictionary containing wired handler callbacks for testing and programmatic invocation.
    """
    from robot.constants import PRESET_POSES

    action_buttons = [exec_btn, home_btn, clear_btn]
    if rand_btn is not None:
        action_buttons.append(rand_btn)
    num_actions = len(action_buttons)

    def process_instruction(
        msg: str = "",
        history: list | None = None,
    ):
        msg = msg or ""
        history = list(history) if history is not None else []
        if not msg.strip():
            yield (
                gr.skip(),
                history,
                gr.skip(),
                *[gr.skip()] * num_actions,
                *[gr.skip()] * 6,
            )
            return

        # Append the new user message to the UI history
        history.append({"role": "user", "content": msg})

        # Append empty assistant message that will be streamed into
        history.append({"role": "assistant", "content": ""})

        # Turn start: block all user inputs and show Working status
        yield (
            gr.update(
                value="",
                interactive=False,
                placeholder="⏳ Executing instruction...",
            ),
            history,
            gr.update(value="⏳ **Status: Working...**"),
            *[gr.update(interactive=False) for _ in action_buttons],
            *[gr.skip()] * 6,
        )

        try:
            for partial_text in controller.process_instruction_stream(
                instruction=msg,
                get_latest_jpeg_fn=get_latest_raw_jpeg,
                get_fresh_jpeg_fn=get_raw_jpeg_after,
            ):
                history[-1]["content"] = partial_text
                yield (
                    gr.skip(),
                    history,
                    gr.skip(),
                    *[gr.skip()] * num_actions,
                    *[gr.skip()] * 6,
                )
        except Exception as exc:
            try:
                controller.interrupt()
            except Exception:
                pass
            err_msg = f"\n\n⚠️ **Runtime Error**: `{exc}`"
            history[-1]["content"] = (history[-1]["content"] or "") + err_msg
        finally:
            # After execution completes or errors, sync sliders with final robot state
            final_deg = [gr.skip()] * 6
            if controller.robot is not None:
                measured = get_ui_joint_positions(controller.robot)
                if measured is not None:
                    final_deg = measured

            is_calibrated = True
            if controller.mode == "real":
                calib = get_calibrator()
                if calib is None or not getattr(calib, "is_calibrated", False):
                    is_calibrated = False
                elif hasattr(calib, "require_calibrated"):
                    try:
                        calib.require_calibrated()
                    except CalibrationUnavailable:
                        is_calibrated = False

            if not controller.is_api_available:
                ready_status = "⚠️ **Status: Gemini API Unavailable**"
                default_placeholder = "⚠️ Gemini API Unavailable..."
            elif not is_calibrated:
                ready_status = "⚠️ **Status: Workspace Not Calibrated**"
                default_placeholder = "⚠️ Calibrate workspace before issuing commands..."
            else:
                ready_status = "🟢 **Status: Ready**"
                default_placeholder = READY_PROMPT_PLACEHOLDER

            # Re-enable inputs on completion / finally
            btn_updates = [
                gr.update(interactive=controller.is_api_available if btn is exec_btn else True)
                for btn in action_buttons
            ]

            yield (
                gr.update(
                    interactive=controller.is_api_available,
                    placeholder=default_placeholder,
                ),
                history,
                gr.update(value=ready_status),
                *btn_updates,
                *final_deg,
            )

    def reset_to_home() -> list[float]:
        if controller.robot is not None:
            home_target = PRESET_POSES["HOME"]
            try:
                if hasattr(controller.robot, "move_to_pose"):
                    controller.robot.move_to_pose(home_target, duration_sec=1.0)
                elif hasattr(controller.robot, "set_joint_positions"):
                    controller.robot.set_joint_positions(home_target)
            except Exception as exc:
                logger.warning(f"Failed to reset to home pose: {exc}")
        home_deg = [round(float(np.degrees(v)), 1) for v in PRESET_POSES["HOME"]]
        return home_deg

    def randomize_cube() -> None:
        if controller.robot is not None and hasattr(controller.robot, "randomize_target"):
            new_pos = controller.robot.randomize_target()
            logger.info(f"Random target placed at XYZ: {[round(v, 3) for v in new_pos]}")

    def clear_chat() -> tuple[list, dict]:
        controller.clear_history()
        calib = get_calibrator()
        is_calibrated = (
            calib is not None and calib.is_calibrated if controller.mode == "real" else True
        )

        if not controller.is_api_available:
            status = "⚠️ **Status: Gemini API Unavailable**"
        elif not is_calibrated:
            status = "⚠️ **Status: Workspace Not Calibrated**"
        else:
            status = "🟢 **Status: Ready**"

        if not is_calibrated:
            msg = (
                "⚠️ **Workspace Not Calibrated**\n\n"
                f"Ready in `{controller.mode.upper()}` mode. Physical workspace calibration is "
                "required before executing spatial commands.\n\n"
                "👉 Please place the 4 ArUco markers in view and click **'💾 Save Calibration'** "
                "under the **🎯 ArUco Workspace Calibration** panel."
            )
        else:
            msg = (
                f"🤖 **System Active**: Ready in `{controller.mode.upper()}` mode with calibrated "
                "workspace. Type commands below to invoke spatial reasoning."
            )

        return [{"role": "assistant", "content": msg}], gr.update(value=status)

    outputs = [prompt_input, chatbot, status_banner, *action_buttons, *sliders]

    exec_btn.click(
        fn=process_instruction,
        inputs=[prompt_input, chatbot],
        outputs=outputs,
    )
    prompt_input.submit(
        fn=process_instruction,
        inputs=[prompt_input, chatbot],
        outputs=outputs,
    )
    home_btn.click(fn=reset_to_home, inputs=None, outputs=sliders)
    if rand_btn is not None:
        rand_btn.click(fn=randomize_cube, inputs=None, outputs=None)
    clear_btn.click(fn=clear_chat, inputs=None, outputs=[chatbot, status_banner])

    return {
        "process_instruction": process_instruction,
        "reset_to_home": reset_to_home,
        "randomize_cube": randomize_cube,
        "clear_chat": clear_chat,
    }
