"""Chat component for VLA instruction integration."""

import gradio as gr
import numpy as np

from agent.vla_controller import VLAController
from ui.stream import get_latest_jpeg


def build_chat_interface(controller: VLAController, api_available: bool = True) -> tuple:
    """Build the Chatbot interface within the current Gradio context.

    Args:
        controller: Initialized VLAController instance.
        api_available: Whether the Gemini API is available.

    Returns:
        Tuple of (prompt_input, chatbot, exec_btn, stop_btn, clear_btn)
    """
    chatbot = gr.Chatbot(
        label="Gemini Robotics Assistant (gemini-robotics-er)",
        height=440,
        elem_id="vla-chatbot",
        value=[
            {
                "role": "assistant",
                "content": (
                    f"🤖 **System Active**: Ready in `{controller.mode.upper()}` mode. "
                    "Type commands below to invoke spatial reasoning."
                ),
            }
        ],
    )
    if not api_available:
        gr.Markdown("⚠️ **Gemini API unavailable — manual control only.** Check your credentials.")

    prompt_input = gr.Textbox(
        label="Instruction Prompt",
        placeholder='e.g., "Pick up red cube and place in sorting basket..."',
        lines=2,
        interactive=api_available,
        elem_id="vla-prompt",
    )
    with gr.Row():
        exec_btn = gr.Button(
            "🚀 Execute Instruction", variant="primary", scale=2, interactive=api_available
        )
        stop_btn = gr.Button("🛑 Stop", variant="stop", scale=1)
        clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", scale=1)

    return prompt_input, chatbot, exec_btn, stop_btn, clear_btn


def wire_chat_events(
    controller: VLAController,
    prompt_input: gr.Textbox,
    chatbot: gr.Chatbot,
    exec_btn: gr.Button,
    stop_btn: gr.Button,
    clear_btn: gr.Button,
    sliders: list[gr.Slider],
) -> None:
    """Wire up the event listeners for the chat interface.

    Args:
        controller: The VLAController instance.
        prompt_input: The Gradio Textbox for user prompts.
        chatbot: The Gradio Chatbot component.
        exec_btn: The execute button.
        stop_btn: The stop button.
        clear_btn: The clear chat button.
        sliders: The list of 6 Gradio joint sliders to update.
    """

    def process_instruction(
        msg: str = "",
        history: list | None = None,
    ):
        msg = msg or ""
        history = history or []
        if not msg.strip():
            yield "", history, *[gr.skip()] * 6
            return

        # Append the new user message to the UI history
        history.append({"role": "user", "content": msg})

        # Append empty assistant message that will be streamed into
        history.append({"role": "assistant", "content": ""})
        yield "", history, *[gr.skip()] * 6

        for partial_text in controller.process_instruction_stream(
            instruction=msg, get_latest_jpeg_fn=get_latest_jpeg
        ):
            history[-1]["content"] = partial_text
            yield "", history, *[gr.skip()] * 6

        # After execution completes, sync sliders with final robot state
        final_deg = [gr.skip()] * 6
        if controller.robot is not None and hasattr(controller.robot, "get_joint_positions"):
            try:
                curr_rad = controller.robot.get_joint_positions()
                final_deg = [round(float(v), 1) for v in np.degrees(curr_rad)]
            except Exception:
                pass

        yield "", history, *final_deg

    def clear_chat() -> list:
        controller.clear_history()
        # Reset UI to show initial state
        return [
            {
                "role": "assistant",
                "content": (
                    f"🤖 **System Active**: Ready in `{controller.mode.upper()}` mode. "
                    "Type commands below to invoke spatial reasoning."
                ),
            }
        ]

    exec_btn.click(
        fn=process_instruction,
        inputs=[prompt_input, chatbot],
        outputs=[prompt_input, chatbot, *sliders],
    )
    prompt_input.submit(
        fn=process_instruction,
        inputs=[prompt_input, chatbot],
        outputs=[prompt_input, chatbot, *sliders],
    )
    stop_btn.click(fn=lambda: controller.interrupt(), inputs=None, outputs=None)
    clear_btn.click(fn=clear_chat, inputs=None, outputs=[chatbot])
