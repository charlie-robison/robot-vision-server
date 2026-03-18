"""Quick test client — sends a solid-colored test image to the WebSocket server."""

import asyncio
import base64
import json

import cv2
import numpy as np
import websockets


async def send_test_image():
    uri = "ws://localhost:8765"

    # Create a 200x200 solid blue test image (BGR)
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[:] = (255, 0, 0)  # blue in BGR

    # Encode as PNG
    _, encoded = cv2.imencode(".png", image)
    b64 = base64.b64encode(encoded.tobytes()).decode()

    async with websockets.connect(uri) as ws:
        payload = json.dumps({"width": 200, "height": 200, "image": b64})
        await ws.send(payload)
        response = json.loads(await ws.recv())
        print("Server response:", json.dumps(response, indent=2))


if __name__ == "__main__":
    asyncio.run(send_test_image())
