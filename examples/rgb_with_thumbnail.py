#!/usr/bin/env python3
"""Working DepthAI v3 example: RGB MJPEG + OpenCV thumbnail preview."""

from __future__ import annotations

import signal
import time
from pathlib import Path
from threading import Event

import cv2
import depthai as dai
import numpy as np

from depthai_camera_stream import CameraStream

STOP = Event()
WINDOW_NAME = "RGB thumbnail"


def request_stop(*_args: object) -> None:
    STOP.set()


def write_jpeg_once(path: Path, packet: object) -> None:
    if path.exists():
        return
    data = packet.getData()
    path.write_bytes(bytes(data))
    print(f"wrote {path} ({len(data)} bytes)")


def decode_mjpeg(packet: object) -> np.ndarray | None:
    """Decode one DepthAI MJPEG packet into an OpenCV BGR image."""
    encoded = np.frombuffer(bytes(packet.getData()), dtype=np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def show_thumbnail(packet: object) -> None:
    frame = decode_mjpeg(packet)
    if frame is not None:
        cv2.imshow(WINDOW_NAME, frame)


def main() -> None:
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    rgb_path = Path("rgb.jpg")
    thumbnail_path = Path("thumbnail.jpg")

    try:
        with dai.Pipeline(dai.Device(dai.DeviceInfo("169.254.1.222"))) as pipeline:
            rgb = CameraStream(
                pipeline=pipeline,
                name="rgb",
                socket=dai.CameraBoardSocket.CAM_A,
                size=(1920, 1080),
                fps=30,
                input_type=dai.ImgFrame.Type.NV12,
                resize_mode=dai.ImgResizeMode.CROP,
                mjpeg_quality=90,
                queue_size=4,
                queue_blocking=False,
                thumbnail_size=(320, 180),
                thumbnail_fps=5,
                thumbnail_mjpeg_quality=70,
                thumbnail_queue_size=1,
                thumbnail_queue_blocking=False,
            ).build()

            rgb_calib = rgb.read_calibration_dict()
            print(rgb_calib)

            pipeline.start()
            print("Pipeline started. Press q, Esc, or Ctrl+C to stop.")

            main_count = 0
            thumbnail_count = 0
            last_report = time.monotonic()

            while pipeline.isRunning() and not STOP.is_set():
                main_packet = rgb.queue.tryGet()
                if main_packet is not None:
                    main_count += 1
                    write_jpeg_once(rgb_path, main_packet)

                thumbnail_packet = rgb.thumbnail_queue.tryGet()
                if thumbnail_packet is not None:
                    thumbnail_count += 1
                    write_jpeg_once(thumbnail_path, thumbnail_packet)
                    show_thumbnail(thumbnail_packet)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or Esc
                    request_stop()

                now = time.monotonic()
                if now - last_report >= 1.0:
                    print(
                        f"received in last interval: main={main_count}, "
                        f"thumbnail={thumbnail_count}"
                    )
                    main_count = 0
                    thumbnail_count = 0
                    last_report = now

                time.sleep(0.001)

            pipeline.stop()
            pipeline.wait()
            print("Pipeline stopped.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
