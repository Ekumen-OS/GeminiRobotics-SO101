"""Live API WebSocket session manager for Gemini."""

import asyncio
import base64
import json
import logging
from typing import Any, AsyncIterator

from google import genai
from google.genai import types

from robot.errors import RoboticsError
from robot.logger import log_with_tag
from robot.settings import get_settings

logger = logging.getLogger(__name__)


class SessionError(ConnectionError, RoboticsError):
    """Raised when Gemini Live API session connection, send, or receive fails."""


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


def _serialize_function_responses(
    function_responses: list[types.FunctionResponse],
) -> dict[str, Any]:
    """Serialize function responses with multimodal parts to Live API WebSocket wire format.

    Emits camelCase keys matching the Live API protocol: functionResponses, inlineData, mimeType.
    Rejects unsupported part fields to prevent silent payload loss.

    Args:
        function_responses: List of FunctionResponse objects.

    Returns:
        JSON-serializable dictionary with tool_response structure.

    Raises:
        ValueError: If a part contains unsupported fields.
    """
    responses: list[dict[str, Any]] = []
    for resp in function_responses:
        d = resp.model_dump(exclude_none=True)
        item: dict[str, Any] = {
            "id": d["id"],
            "name": d["name"],
            "response": d.get("response", {}),
        }
        if "parts" in d:
            parts_list = []
            for p in d["parts"]:
                part_dict: dict[str, Any] = {}
                if "inline_data" in p:
                    data_val = p["inline_data"]["data"]
                    if isinstance(data_val, (bytes, bytearray)):
                        data_val = base64.b64encode(data_val).decode("utf-8")
                    part_dict["inlineData"] = {
                        "mimeType": p["inline_data"]["mime_type"],
                        "data": data_val,
                    }
                elif "file_data" in p:
                    raise ValueError("file_data parts in tool responses are not supported")
                else:
                    raise ValueError(f"Unsupported FunctionResponsePart: {p}")
                parts_list.append(part_dict)
            item["parts"] = parts_list
        responses.append(item)
    return {"tool_response": {"functionResponses": responses}}


async def _send_tool_response_wire(
    session: Any, function_responses: list[types.FunctionResponse]
) -> None:
    """Send tool responses carrying multimodal parts directly over the WebSocket.

    Workaround for google-genai <= 2.32.0 bug: AsyncSession.send_tool_response calls
    _common.convert_to_dict which leaves FunctionResponseBlob.data as raw bytes, causing
    json.dumps to raise TypeError ('Object of type bytes is not JSON serializable').
    Delete this function once the SDK natively serializes FunctionResponseBlob.data.

    Args:
        session: Underlying live session object from client.aio.live.connect().
        function_responses: List of FunctionResponse objects containing parts.

    Raises:
        SessionError: If session._ws is not available.
    """
    ws = getattr(session, "_ws", None)
    if ws is None:
        raise SessionError(
            "Cannot send multimodal tool response with parts: session._ws is not available."
        )
    payload = _serialize_function_responses(function_responses)
    await ws.send(json.dumps(payload))


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
        self.config: Any = build_live_config(self.tools, self.system_instruction)

        self._session: Any = None
        self._session_cm: Any = None
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
            try:
                self._session_cm = self.client.aio.live.connect(
                    model=self.model, config=self.config
                )
                self._session = await self._session_cm.__aenter__()
                log_with_tag(logger, "INFO", "Connected successfully.")
            except Exception as e:
                raise SessionError(f"Failed to connect to Gemini Live session: {e}") from e

    async def disconnect(self) -> None:
        """Close the live WebSocket connection."""
        log_with_tag(logger, "INFO", "Disconnecting...")
        async with self._lock:
            if self._session_cm is not None:
                try:
                    await self._session_cm.__aexit__(None, None, None)
                except Exception as e:
                    log_with_tag(logger, "DEBUG", f"Error during disconnect: {e}")
                self._session_cm = None
            self._session = None
        log_with_tag(logger, "INFO", "Disconnected.")

    async def send_instruction(self, text: str, frame_jpeg: bytes | None) -> None:
        """Send user instruction with an optional JPEG frame."""
        if not self._session:
            raise SessionError("Session is not connected.")

        try:
            async with self._ws_lock:
                parts: list[types.Part] = []
                if frame_jpeg:
                    log_with_tag(
                        logger,
                        "STREAM",
                        f"📸 Sent initial observation frame ({len(frame_jpeg)} bytes)",
                    )
                    parts.append(
                        types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=frame_jpeg))
                    )

                parts.append(types.Part(text=text))
                log_with_tag(logger, "INFO", f"Sending instruction turn: {text!r}")
                await self._session.send_client_content(
                    turns=types.Content(role="user", parts=parts),
                    turn_complete=True,
                )
        except Exception as e:
            raise SessionError(f"Failed to send instruction: {e}") from e

    @staticmethod
    def _summarize_tool_responses(function_responses: list[types.FunctionResponse]) -> str:
        """Summarize tool responses without logging raw image bytes."""
        summaries = []
        for resp in function_responses:
            parts_info = ""
            if resp.parts:
                jpeg_bytes = sum(
                    len(p.inline_data.data)
                    for p in resp.parts
                    if p.inline_data and isinstance(p.inline_data.data, (bytes, bytearray))
                )
                parts_info = f" (parts={len(resp.parts)} jpeg={jpeg_bytes} bytes)"
            summaries.append(f"{resp.name}[{resp.id}]: {resp.response}{parts_info}")
        return "[" + ", ".join(summaries) + "]"

    async def send_tool_response(self, function_responses: list[types.FunctionResponse]) -> None:
        """Send tool execution results back to the model."""
        if not self._session:
            return

        try:
            async with self._ws_lock:
                log_with_tag(
                    logger,
                    "INFO",
                    f"Sending tool responses: {self._summarize_tool_responses(function_responses)}",
                )
                has_parts = any(getattr(r, "parts", None) for r in function_responses)
                if has_parts:
                    await _send_tool_response_wire(self._session, function_responses)
                else:
                    await self._session.send_tool_response(function_responses=function_responses)
        except Exception as e:
            raise SessionError(f"Failed to send tool response: {e}") from e

    async def receive(self) -> AsyncIterator[types.LiveServerMessage]:
        """Yield messages from the active session."""
        if not self._session:
            raise SessionError("Session is not connected.")

        try:
            async for message in self._session.receive():
                yield message
        except Exception as e:
            raise SessionError(f"Failed to receive from session: {e}") from e
