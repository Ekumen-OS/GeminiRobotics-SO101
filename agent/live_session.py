"""Live API WebSocket session manager for Gemini."""

import asyncio
import datetime
import os
import sys
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from google import genai
from google.genai import types


def _log(module: str, tag: str, message: str, stream: Any = None) -> None:
    """Format and print an ISO-timestamped diagnostic log entry.

    Args:
        module: Component name (e.g. 'LiveSession').
        tag: Semantic tag (e.g. 'INFO', 'STREAM', 'ERROR', 'WARN').
        message: Log message content.
        stream: Output stream (defaults to sys.stderr for ERROR/WARN, sys.stdout otherwise).
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    if stream is None:
        stream = sys.stderr if tag in ("ERROR", "WARN") else sys.stdout
    print(f"[{now_str}] [{module}] [{tag}] {message}", file=stream, flush=True)


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

            _log("LiveSession", "INFO", f"Connecting to {self.model}...")
            _log("LiveSession", "INFO", f"Config: {self.config}")
            self._session_cm = self.client.aio.live.connect(model=self.model, config=self.config)
            self._session = await self._session_cm.__aenter__()
            _log("LiveSession", "INFO", "Connected successfully.")

    async def disconnect(self) -> None:
        """Close the live WebSocket connection."""
        _log("LiveSession", "INFO", "Disconnecting...")
        async with self._lock:
            if self._session_cm is not None:
                try:
                    await self._session_cm.__aexit__(None, None, None)
                except Exception as e:
                    _log("LiveSession", "ERROR", f"Error during disconnect: {e}")
                self._session_cm = None
            self._session = None
        _log("LiveSession", "INFO", "Disconnected.")

    async def send_instruction(self, text: str, frame_jpeg: bytes | None) -> None:
        """Send user instruction with an optional JPEG frame."""
        if not self._session:
            raise ConnectionError("Session is not connected.")

        async with self._ws_lock:
            if frame_jpeg:
                _log(
                    "LiveSession",
                    "STREAM",
                    f"📸 Sent initial observation frame ({len(frame_jpeg)} bytes)",
                )
                await self._session.send_realtime_input(
                    video=types.Blob(mime_type="image/jpeg", data=frame_jpeg)
                )

            _log("LiveSession", "INFO", f"Sending text: {text!r}")
            await self._session.send_realtime_input(text=text)

    async def send_text(self, text: str) -> None:
        """Send a realtime text prompt to the session."""
        if not self._session:
            return
        async with self._ws_lock:
            _log("LiveSession", "INFO", f"Sending text: {text!r}")
            await self._session.send_realtime_input(text=text)

    async def send_frame(self, jpeg_bytes: bytes) -> None:
        """Send a JPEG frame to the model without instruction (proactive)."""
        if not self._session:
            return
        async with self._ws_lock:
            _log("LiveSession", "STREAM", f"📸 Sent video frame ({len(jpeg_bytes)} bytes)")
            await self._session.send_realtime_input(
                video=types.Blob(mime_type="image/jpeg", data=jpeg_bytes)
            )

    async def send_tool_response(self, function_responses: list[types.FunctionResponse]) -> None:
        """Send tool execution results back to the model."""
        if not self._session:
            return

        async with self._ws_lock:
            _log("LiveSession", "INFO", f"Sending tool responses: {function_responses}")
            await self._session.send_tool_response(function_responses=function_responses)

    async def receive(self) -> AsyncIterator[types.LiveServerMessage]:
        """Yield messages from the active session."""
        if not self._session:
            raise ConnectionError("Session is not connected.")

        async for message in self._session.receive():
            yield message
