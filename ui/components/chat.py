"""Chat component for VLA instruction integration."""

import gradio as gr

from agent.vla_controller import VLAController


def build_chat_interface(controller: VLAController) -> None:
    """Build the Chatbot interface within the current Gradio context.

    Args:
        controller: Initialized VLAController instance.
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
    prompt_input = gr.Textbox(
        label="Instruction Prompt",
        placeholder='e.g., "Pick up red cube and place in sorting basket..."',
        lines=2,
        interactive=True,
        elem_id="vla-prompt",
    )
    with gr.Row():
        exec_btn = gr.Button("🚀 Execute Instruction", variant="primary", scale=2)
        clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", scale=1)

    def process_instruction(
        msg: str = "",
        history: list | None = None,
    ) -> tuple[str, list]:
        msg = msg or ""
        history = history or []
        if not msg.strip():
            return "", history

        # In the future, context_images could be passed from the vision component
        response_text = controller.process_instruction(instruction=msg, context_images=None)

        # Append the new messages to the UI history
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": response_text})

        return "", history

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
        outputs=[prompt_input, chatbot],
    )
    prompt_input.submit(
        fn=process_instruction,
        inputs=[prompt_input, chatbot],
        outputs=[prompt_input, chatbot],
    )
    clear_btn.click(fn=clear_chat, inputs=None, outputs=[chatbot])
