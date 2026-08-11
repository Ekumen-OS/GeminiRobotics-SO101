"""Chat component for VLA instruction integration."""

import gradio as gr


def build_chat_interface(mode: str) -> None:
    """Build the Chatbot interface within the current Gradio context.

    Args:
        mode: Execution operational mode ('real', 'sim', or 'twin').
    """
    chatbot = gr.Chatbot(
        label="Gemini Robotics Assistant (gemini-robotics-er)",
        height=440,
        elem_id="vla-chatbot",
        value=[
            {
                "role": "assistant",
                "content": (
                    f"🤖 **System Active**: Ready in `{mode.upper()}` mode. "
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

    def vla_placeholder_response(
        msg: str = "",
        history: list | None = None,
    ) -> tuple[str, list]:
        msg = msg or ""
        if not msg.strip():
            history = history or []
            return "", history
        history = history or []
        response = (
            f"⚡ **[gemini-robotics-er]**: Received action command *"
            f'"{msg}"* for mode `{mode.upper()}`. Live trajectory generation over '
            "visual observations will execute in upcoming inference pipeline."
        )
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": response})
        return "", history

    exec_btn.click(
        fn=vla_placeholder_response,
        inputs=[prompt_input, chatbot],
        outputs=[prompt_input, chatbot],
    )
    prompt_input.submit(
        fn=vla_placeholder_response,
        inputs=[prompt_input, chatbot],
        outputs=[prompt_input, chatbot],
    )
    clear_btn.click(lambda: [], inputs=None, outputs=[chatbot])
