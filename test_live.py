import asyncio

from google.genai import types

from agent.live_session import GeminiLiveSession
from agent.tools import ROBOT_TOOL_DECLARATIONS


async def main():
    """Run live session test script."""
    system_prompt = "You are a robot."

    print("Initializing session...")
    session = GeminiLiveSession(
        model="gemini-robotics-er-2-streaming-preview",
        tools=ROBOT_TOOL_DECLARATIONS,
        system_instruction=system_prompt,
    )

    try:
        print("Connecting...")
        await session.connect()
        print("Connected!")

        print("Sending instruction...")
        # Create a dummy JPEG using cv2
        import cv2
        import numpy as np

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", img)
        dummy_frame = encoded.tobytes()

        await session._session.send_realtime_input(
            video=types.Blob(mime_type="image/jpeg", data=dummy_frame)
        )
        await session._session.send(input="Move to the home position")

        print("Receiving...")
        async for msg in session.receive():
            print(f"Received: {msg}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await session.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
