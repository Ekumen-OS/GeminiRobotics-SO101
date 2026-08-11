"""Controller for managing Vision-Language-Action (VLA) agent interactions."""


class VLAController:
    """Core controller for managing interaction with gemini-robotics-er.

    This class handles chat history, state management, and interaction with
    the VLA model API, abstracting it from the frontend UI.
    """

    def __init__(self, mode: str = "sim") -> None:
        """Initialize the VLA Controller.

        Args:
            mode: The operational mode of the robot ('real', 'sim', 'twin').
        """
        self.mode = mode
        self.history: list[dict[str, str]] = []

        # System prompt initialization based on mode
        system_prompt = (
            f"You are the Gemini Robotics Assistant, controlling the SO-101 robot arm. "
            f"The system is currently in '{self.mode.upper()}' mode. "
            f"Your capabilities include spatial reasoning and trajectory generation."
        )
        self.history.append({"role": "system", "content": system_prompt})

    def process_instruction(self, instruction: str, context_images: list | None = None) -> str:
        """Process a user instruction and return the agent's response.

        Args:
            instruction: The text instruction from the user.
            context_images: Optional list of images (e.g., from the robot's camera)
                to provide visual context for the instruction.

        Returns:
            The agent's response (descriptive text or trajectory data).
        """
        if not instruction.strip():
            return ""

        self.history.append({"role": "user", "content": instruction})

        response_text = self._mock_gemini_api_call(instruction, context_images)

        self.history.append({"role": "assistant", "content": response_text})

        return response_text

    def _mock_gemini_api_call(self, text: str, images: list | None = None) -> str:
        """Simulate a call to the gemini-robotics-er-2-streaming-preview model.

        Args:
            text: The user's input text.
            images: Optional visual context.

        Returns:
            A simulated text or trajectory response.
        """
        # Mock logic to simulate Live API response pattern
        response = (
            f"⚡ **[gemini-robotics-er]**: Processed action command *"
            f'"{text}"* in mode `{self.mode.upper()}`. '
        )

        if images and len(images) > 0:
            response += f"Analyzed {len(images)} context image(s). "

        response += "Simulated trajectory generation completed."

        return response

    def get_history(self) -> list[dict[str, str]]:
        """Retrieve the current conversation history.

        Returns:
            A list of message dictionaries with 'role' and 'content' keys.
        """
        return self.history

    def clear_history(self) -> None:
        """Clear the conversation history, keeping only the system prompt."""
        if self.history:
            system_prompt = self.history[0]
            self.history = [system_prompt]
