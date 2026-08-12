"""Live API WebSocket session manager for Gemini."""

import asyncio
import os
from typing import AsyncIterator

from dotenv import load_dotenv
from google import genai
from google.genai import types


def create_genai_client() -> genai.Client:
    """Create a genai Client with dual auth support.

    Prioritizes GOOGLE_API_KEY for Google AI Studio.
    Falls back to Vertex AI ADC if GOOGLE_CLOUD_PROJECT is set.

    Raises:
        ConnectionError: If neither authentication method is configured.
    """
    load_dotenv()
    if "GOOGLE_API_KEY" in os.environ:
        return genai.Client()

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    if project:
        return genai.Client(vertexai=True, project=project, location=location)

    raise ConnectionError(
        "Missing authentication credentials for Gemini Live API. "
        "Please set either GOOGLE_API_KEY or GOOGLE_CLOUD_PROJECT."
    )


def build_live_config(tool_declarations: list[dict], system_instruction: str) -> dict:
    """Build the LiveConnectConfig for the session."""
    return {
        "response_modalities": ["TEXT"],
        "tools": [{"function_declarations": tool_declarations}],
        "system_instruction": {"parts": [{"text": system_instruction}]},
    }


class GeminiLiveSession:
    """Manager for the Gemini Live API WebSocket session."""

    def __init__(self, model: str, tools: list[dict], system_instruction: str):
        """Initialize the Gemini Live session manager.

        Args:
            model: The Gemini model identifier.
            tools: List of tool declarations.
            system_instruction: System prompt text for the session.
        """
        self.model = model
        self.tools = tools
        self.system_instruction = system_instruction

        self.client = create_genai_client()
        self.config = build_live_config(self.tools, self.system_instruction)

        self._session = None
        self._session_cm = None
        self._lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        """Check if the session is currently active."""
        return self._session is not None

    async def connect(self) -> None:
        """Establish the live WebSocket connection."""
        async with self._lock:
            if self._session is not None:
                return

            print(f"[LiveSession] Connecting to {self.model}...")
            print(f"[LiveSession] Config: {self.config}")
            self._session_cm = self.client.aio.live.connect(model=self.model, config=self.config)
            self._session = await self._session_cm.__aenter__()
            print("[LiveSession] Connected successfully.")

    async def disconnect(self) -> None:
        """Close the live WebSocket connection."""
        print("[LiveSession] Disconnecting...")
        async with self._lock:
            if self._session_cm is not None:
                try:
                    await self._session_cm.__aexit__(None, None, None)
                except Exception as e:
                    print(f"[LiveSession] Error during disconnect: {e}")
                self._session_cm = None
            self._session = None
        print("[LiveSession] Disconnected.")

    async def send_instruction(self, text: str, frame_jpeg: bytes | None) -> None:
        """Send user instruction with an optional JPEG frame."""
        if not self._session:
            raise ConnectionError("Session is not connected.")

        if frame_jpeg:
            print(f"[LiveSession] Sending frame ({len(frame_jpeg)} bytes)")
            await self._session.send_realtime_input(
                video=types.Blob(mime_type="image/jpeg", data=frame_jpeg)
            )

        print(f"[LiveSession] Sending text: {text!r}")
        await self._session.send_realtime_input(text=text)

    async def send_frame(self, jpeg_bytes: bytes) -> None:
        """Send a JPEG frame to the model without instruction (proactive)."""
        if not self._session:
            return
        await self._session.send_realtime_input(
            video=types.Blob(mime_type="image/jpeg", data=jpeg_bytes)
        )

    async def send_tool_response(self, function_responses: list[types.FunctionResponse]) -> None:
        """Send tool execution results back to the model."""
        if not self._session:
            return

        print(f"[LiveSession] Sending tool responses: {function_responses}")
        await self._session.send_tool_response(function_responses=function_responses)

    async def receive(self) -> AsyncIterator[types.LiveServerMessage]:
        """Yield messages from the active session."""
        if not self._session:
            raise ConnectionError("Session is not connected.")

        async for message in self._session.receive():
            yield message
