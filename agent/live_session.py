"""Live API WebSocket session manager for Gemini."""

import asyncio
import logging
from typing import AsyncIterator

from google import genai
from google.genai import types

from robot.logger import log_with_tag
from robot.settings import get_settings

logger = logging.getLogger(__name__)


def create_genai_client() -> genai.Client:
    """Create a genai Client with dual auth support based on Settings.

    Prioritizes GOOGLE_API_KEY for Google AI Studio.
    Falls back to Vertex AI ADC if GOOGLE_CLOUD_PROJECT is set.

    Raises:
        ConnectionError: If neither authentication method is configured.
    """
    settings = get_settings()
    if settings.credentials_mode == "api_key" and settings.google_api_key:
        return genai.Client(api_key=settings.google_api_key)
    if settings.credentials_mode == "vertex" and settings.google_cloud_project:
        return genai.Client(
            vertexai=True,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
        )

    raise ConnectionError(
        "Missing authentication credentials for Gemini Live API. "
        "Please set either GOOGLE_API_KEY or GOOGLE_CLOUD_PROJECT."
    )


def build_live_config(tool_declarations: list[dict], system_instruction: str) -> dict:
    """Build the LiveConnectConfig for the session."""
    return {
        "response_modalities": ["TEXT"],
        "thinking_config": {"include_thoughts": True},
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
        self._ws_lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        """Check if the session is currently active."""
        return self._session is not None

    async def connect(self) -> None:
        """Establish the live WebSocket connection."""
        async with self._lock:
            if self._session is not None:
                return

            log_with_tag(logger, "INFO", f"Connecting to {self.model}...")
            log_with_tag(logger, "INFO", f"Config: {self.config}")
            self._session_cm = self.client.aio.live.connect(model=self.model, config=self.config)
            self._session = await self._session_cm.__aenter__()
            log_with_tag(logger, "INFO", "Connected successfully.")

    async def disconnect(self) -> None:
        """Close the live WebSocket connection."""
        log_with_tag(logger, "INFO", "Disconnecting...")
        async with self._lock:
            if self._session_cm is not None:
                try:
                    await self._session_cm.__aexit__(None, None, None)
                except Exception as e:
                    log_with_tag(logger, "ERROR", f"Error during disconnect: {e}")
                self._session_cm = None
            self._session = None
        log_with_tag(logger, "INFO", "Disconnected.")

    async def send_instruction(self, text: str, frame_jpeg: bytes | None) -> None:
        """Send user instruction with an optional JPEG frame."""
        if not self._session:
            raise ConnectionError("Session is not connected.")

        async with self._ws_lock:
            if frame_jpeg:
                log_with_tag(
                    logger,
                    "STREAM",
                    f"📸 Sent initial observation frame ({len(frame_jpeg)} bytes)",
                )
                await self._session.send_realtime_input(
                    video=types.Blob(mime_type="image/jpeg", data=frame_jpeg)
                )

            log_with_tag(logger, "INFO", f"Sending text: {text!r}")
            await self._session.send_realtime_input(text=text)

    async def send_text(self, text: str) -> None:
        """Send a realtime text prompt to the session."""
        if not self._session:
            return
        async with self._ws_lock:
            log_with_tag(logger, "INFO", f"Sending text: {text!r}")
            await self._session.send_realtime_input(text=text)

    async def send_frame(self, jpeg_bytes: bytes) -> None:
        """Send a JPEG frame to the model without instruction (proactive)."""
        if not self._session:
            return
        async with self._ws_lock:
            log_with_tag(logger, "STREAM", f"📸 Sent video frame ({len(jpeg_bytes)} bytes)")
            await self._session.send_realtime_input(
                video=types.Blob(mime_type="image/jpeg", data=jpeg_bytes)
            )

    async def send_tool_response(self, function_responses: list[types.FunctionResponse]) -> None:
        """Send tool execution results back to the model."""
        if not self._session:
            return

        async with self._ws_lock:
            log_with_tag(logger, "INFO", f"Sending tool responses: {function_responses}")
            await self._session.send_tool_response(function_responses=function_responses)

    async def receive(self) -> AsyncIterator[types.LiveServerMessage]:
        """Yield messages from the active session."""
        if not self._session:
            raise ConnectionError("Session is not connected.")

        async for message in self._session.receive():
            yield message
