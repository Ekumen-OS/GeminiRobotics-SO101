"""Pure progressive response assembler for markdown chat streaming."""

from typing import Any


class ResponseAssembler:
    """Stateful progressive assembler that formats thought, text, status, and error items."""

    def __init__(self) -> None:
        """Initialize an empty response assembler state."""
        self._full_response = ""
        self._in_thought_block = False

    def feed(self, item: dict[str, Any]) -> str:
        """Process an item and return the cumulative response string so far.

        Args:
            item: Queue item dictionary with 'type' and 'content' keys.

        Returns:
            Current formatted response text.
        """
        item_type = item.get("type")
        content = item.get("content", "")

        if item_type == "thought":
            if not self._in_thought_block:
                if self._full_response and not self._full_response.endswith("\n"):
                    self._full_response += "\n\n"
                self._full_response += "> 💭 *"
                self._in_thought_block = True
            self._full_response += content

        elif item_type == "text":
            if self._in_thought_block:
                self._full_response += "*\n\n"
                self._in_thought_block = False
            if (
                self._full_response
                and not self._full_response.endswith("\n")
                and not self._full_response.endswith(" ")
            ):
                self._full_response += "\n\n"
            self._full_response += content

        elif item_type == "status":
            if self._in_thought_block:
                self._full_response += "*\n\n"
                self._in_thought_block = False
            if self._full_response and not self._full_response.endswith("\n"):
                self._full_response += "\n"
            self._full_response += f"{content}\n"

        elif item_type == "error":
            if self._in_thought_block:
                self._full_response += "*\n\n"
                self._in_thought_block = False
            if self._full_response and not self._full_response.endswith("\n"):
                self._full_response += "\n"
            self._full_response += f"\n**Error:** {content}\n"

        return self._full_response

    def finish(self) -> str:
        """Finalize assembly, closing any unclosed thought block.

        Returns:
            Final formatted response text.
        """
        if self._in_thought_block:
            self._full_response += "*\n\n"
            self._in_thought_block = False
        return self._full_response
