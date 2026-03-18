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

# HSV ranges that cover brown tones including dark wood
BROWN_RANGES = [
    ((8, 30, 10), (20, 255, 160)),
    ((20, 30, 10), (30, 255, 130)),
]

# Minimum percentage of non-background pixels that must be brown to return True
BROWN_THRESHOLD_PCT = 20.0

# Obstruction detection thresholds
OBSTRUCTION_BRIGHTNESS_DIFF = 40  # min difference in mean V between halves
OBSTRUCTION_DARK_THRESHOLD = 50   # absolute mean V below which a side is "dark"


def detect_obstruction(image: np.ndarray) -> str | None:
    """Return 'LEFT' or 'RIGHT' to steer away from an obstructed side, or None."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    mid_x = v_channel.shape[1] // 2
    left_mean = float(np.mean(v_channel[:, :mid_x]))
    right_mean = float(np.mean(v_channel[:, mid_x:]))

    diff = abs(left_mean - right_mean)

    # One side is much darker than the other → steer away from the dark side
    if diff >= OBSTRUCTION_BRIGHTNESS_DIFF:
        return "RIGHT" if left_mean < right_mean else "LEFT"

    # One side has very low absolute brightness → close dark object
    if left_mean < OBSTRUCTION_DARK_THRESHOLD and right_mean >= OBSTRUCTION_DARK_THRESHOLD:
        return "RIGHT"
    if right_mean < OBSTRUCTION_DARK_THRESHOLD and left_mean >= OBSTRUCTION_DARK_THRESHOLD:
        return "LEFT"

    return None


def detect_brown_direction(image: np.ndarray) -> str | None:
    """Return 'LEFT' or 'RIGHT' to avoid a brown object, or None if no brown detected."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Ignore near-black and near-white background pixels
    fg_mask = cv2.inRange(hsv, (0, 10, 10), (180, 255, 255))
    fg_pixels = int(cv2.countNonZero(fg_mask))
    if fg_pixels == 0:
        return None

    # Build combined brown mask
    brown_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in BROWN_RANGES:
        range_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        brown_mask = cv2.bitwise_or(brown_mask, range_mask)
    brown_mask = cv2.bitwise_and(brown_mask, fg_mask)

    brown_pixels = int(cv2.countNonZero(brown_mask))
    brown_pct = brown_pixels / fg_pixels * 100
    if brown_pct < BROWN_THRESHOLD_PCT:
        return None

    # Find the center of the brown region
    mid_x = image.shape[1] // 2
    left_brown = int(cv2.countNonZero(brown_mask[:, :mid_x]))
    right_brown = int(cv2.countNonZero(brown_mask[:, mid_x:]))

    # Steer away from the object
    return "LEFT" if right_brown >= left_brown else "RIGHT"


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

            # Detect obstruction first (close-up objects), then brown objects
            obstruction_dir = detect_obstruction(image)
            direction = obstruction_dir or detect_brown_direction(image)

            if direction:
                source = "obstruction" if obstruction_dir else "brown"
                await websocket.send(direction)
                print(f"[*] {source.capitalize()} detected → steer {direction}")
            else:
                await websocket.send("NONE")
                print("[*] No obstruction or brown object detected")

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
