import asyncio
import json
import os
import struct
from datetime import datetime

import cv2
import numpy as np
import websockets

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "received_images")
os.makedirs(SAVE_DIR, exist_ok=True)

# HSV color ranges mapped to human-readable names
COLOR_RANGES = [
    ((0, 70, 50), (10, 255, 255), "red"),
    ((170, 70, 50), (180, 255, 255), "red"),
    ((11, 70, 50), (25, 255, 255), "orange"),
    ((26, 70, 50), (34, 255, 255), "yellow"),
    ((35, 70, 50), (85, 255, 255), "green"),
    ((86, 70, 50), (125, 255, 255), "blue"),
    ((126, 70, 50), (145, 255, 255), "purple"),
    ((146, 70, 50), (169, 255, 255), "pink"),
]


def detect_dominant_color(image: np.ndarray) -> dict:
    """Detect the dominant color of the main object in the image using OpenCV."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Build a mask to ignore near-black and near-white (background) pixels
    mask = cv2.inRange(hsv, (0, 30, 30), (180, 255, 255))

    # Count pixels for each named color range
    color_counts: dict[str, int] = {}
    for lower, upper, name in COLOR_RANGES:
        color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        combined = cv2.bitwise_and(color_mask, mask)
        count = int(cv2.countNonZero(combined))
        color_counts[name] = color_counts.get(name, 0) + count

    # Check for white / gray / black regions
    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
    color_counts["white"] = int(cv2.countNonZero(white_mask))

    gray_mask = cv2.inRange(hsv, (0, 0, 50), (180, 30, 199))
    color_counts["gray"] = int(cv2.countNonZero(gray_mask))

    black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 29))
    color_counts["black"] = int(cv2.countNonZero(black_mask))

    total_pixels = image.shape[0] * image.shape[1]
    dominant = max(color_counts, key=color_counts.get)
    dominant_pct = round(color_counts[dominant] / total_pixels * 100, 1)

    # Build a sorted breakdown of all detected colors
    breakdown = {
        name: round(count / total_pixels * 100, 1)
        for name, count in sorted(color_counts.items(), key=lambda x: -x[1])
        if count > 0
    }

    return {
        "dominant_color": dominant,
        "dominant_pct": dominant_pct,
        "breakdown": breakdown,
    }


async def handle_client(websocket):
    print(f"[+] Client connected: {websocket.remote_address}")
    try:
        async for message in websocket:
            # --- JSON text message ------------------------------------------
            if isinstance(message, str):
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"error": "Invalid JSON"}))
                    continue

                width = payload.get("width")
                height = payload.get("height")
                img_bytes = payload.get("image")  # base64 or list-of-ints

                if width is None or height is None or img_bytes is None:
                    await websocket.send(
                        json.dumps({"error": "JSON must include 'width', 'height', and 'image' (base64 string or int array)"})
                    )
                    continue

                # Decode image bytes
                if isinstance(img_bytes, str):
                    import base64
                    raw = base64.b64decode(img_bytes)
                elif isinstance(img_bytes, list):
                    raw = bytes(img_bytes)
                else:
                    await websocket.send(json.dumps({"error": "'image' must be base64 string or int array"}))
                    continue

            # --- Binary message: 4-byte width + 4-byte height + raw pixels --
            else:
                if len(message) < 8:
                    await websocket.send(json.dumps({"error": "Binary message too short (need 8-byte header)"}))
                    continue
                width, height = struct.unpack("!II", message[:8])
                raw = message[8:]

            # Try to decode as an encoded image (PNG/JPEG/etc.) first
            arr = np.frombuffer(raw, dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if image is None:
                # Fall back to treating raw bytes as BGR pixel data
                expected = width * height * 3
                if len(raw) == width * height * 4:
                    # BGRA → BGR
                    image = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                elif len(raw) == expected:
                    image = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                else:
                    await websocket.send(
                        json.dumps({
                            "error": f"Cannot decode image. Raw length {len(raw)} "
                                     f"doesn't match {width}x{height}x3={expected} or a valid encoded format."
                        })
                    )
                    continue

            # Save to disk
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_path = os.path.join(SAVE_DIR, f"image_{timestamp}.png")
            cv2.imwrite(save_path, image)
            print(f"[*] Saved image ({image.shape[1]}x{image.shape[0]}) → {save_path}")

            # Detect color
            result = detect_dominant_color(image)
            result["saved_to"] = save_path
            result["resolution"] = f"{image.shape[1]}x{image.shape[0]}"

            await websocket.send(json.dumps(result))
            print(f"[*] Dominant color: {result['dominant_color']} ({result['dominant_pct']}%)")

    except websockets.ConnectionClosed:
        print(f"[-] Client disconnected: {websocket.remote_address}")


async def main():
    host = "0.0.0.0"
    port = 8765
    print(f"Color detection WebSocket server starting on ws://{host}:{port}")
    async with websockets.serve(handle_client, host, port):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    main = asyncio.run(main())
