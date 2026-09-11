#!/usr/bin/env python3
"""Low-latency DepthAI v3 RGB + thumbnail + stereo MJPEG example."""

from __future__ import annotations

from dataclasses import dataclass
import signal
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import depthai as dai
import numpy as np

from depthai_camera_stream import CameraStream

STOP = threading.Event()
RGB_WINDOW = "RGB thumbnail"
STEREO_WINDOW = "Stereo (left | right)"

@dataclass
class DaiStereoCameraStream:
    device_ip: str = ""
    rgb_size: tuple[int, int] = (3872, 3008)
    stereo_size: tuple[int, int] = (1280, 800)
    mjpeg_quality: int = 95
    fps: float = 12.0
    input_type: str = "NV12"
    resize_mode: str = "CROP"
    max_exposure_us: int = 16667

    def build(self, pipeline: dai.Pipeline) -> dict[str, Any]:
        common = dict(
            pipeline=pipeline,
            fps=self.fps,
            max_exposure_us=self.max_exposure_us,
            input_type={"NV12": dai.ImgFrame.Type.NV12}[self.input_type],
            resize_mode={"CROP": dai.ImgResizeMode.CROP}[self.resize_mode],
            queue_size=1,
            queue_blocking=False,
            thumbnail_size=(192, 150),
            thumbnail_fps=self.fps,
            thumbnail_mjpeg_quality=70,
            thumbnail_queue_size=1,
            thumbnail_queue_blocking=False,
        )
        specs = {
            "rgb": (dai.CameraBoardSocket.CAM_A, self.rgb_size, self.mjpeg_quality),
            "left": (dai.CameraBoardSocket.CAM_B, self.stereo_size, self.mjpeg_quality - 5),
            "right": (dai.CameraBoardSocket.CAM_C, self.stereo_size, self.mjpeg_quality - 5),
        }
        return {
            name: CameraStream(
                name=name, socket=socket, size=size, mjpeg_quality=quality, **common
            ).build()
            for name, (socket, size, quality) in specs.items()
        }


JPEG_PATHS = {
    name: Path(f"{name}.jpg") for name in ("rgb", "left", "right", "thumbnail")
}


def request_stop(*_: object) -> None:
    STOP.set()


def write_jpeg_once(path: Path, packet: Any) -> None:
    if path.exists():
        return
    data = bytes(packet.getData())
    path.write_bytes(data)
    print(f"wrote {path} ({len(data)} bytes)")


def decode_mjpeg(packet: Any) -> np.ndarray | None:
    data = np.frombuffer(bytes(packet.getData()), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class PreviewWorker:
    """Decode/display only the newest submitted preview packets."""

    def __init__(self, stop_event: threading.Event) -> None:
        self.stop = stop_event
        self.shutdown = threading.Event()
        self.wake = threading.Event()
        self.lock = threading.Lock()
        self.packets: dict[str, Any | None] = {
            "thumbnail": None,
            "left": None,
            "right": None,
        }
        self.thread = threading.Thread(
            target=self._run, name="opencv-preview", daemon=True
        )

    def start(self) -> None:
        self.thread.start()

    def submit(self, **packets: Any | None) -> None:
        with self.lock:
            for name, packet in packets.items():
                if packet is not None:
                    self.packets[name] = packet
        self.wake.set()

    def close(self) -> None:
        self.shutdown.set()
        self.wake.set()
        self.thread.join(timeout=2.0)

    def _take_latest(self) -> dict[str, Any | None]:
        with self.lock:
            packets = self.packets.copy()
            self.packets = dict.fromkeys(self.packets)
        return packets

    def _run(self) -> None:
        stereo: dict[str, np.ndarray | None] = {"left": None, "right": None}
        try:
            while not self.shutdown.is_set() and not self.stop.is_set():
                self.wake.wait(0.05)
                self.wake.clear()
                packets = self._take_latest()

                if packets["thumbnail"] is not None:
                    frame = decode_mjpeg(packets["thumbnail"])
                    if frame is not None:
                        cv2.imshow(RGB_WINDOW, frame)

                changed = False
                for side in ("left", "right"):
                    packet = packets[side]
                    if packet is not None:
                        frame = decode_mjpeg(packet)
                        if frame is not None:
                            stereo[side] = frame
                            changed = True

                if changed and all(frame is not None for frame in stereo.values()):
                    cv2.imshow(
                        STEREO_WINDOW,
                        np.hstack((stereo["left"], stereo["right"])),
                    )

                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    self.stop.set()
        finally:
            cv2.destroyAllWindows()


def main() -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)

    preview = PreviewWorker(STOP)

    with dai.Pipeline() as pipeline:
        streams = DaiStereoCameraStream().build(pipeline)
        pipeline.start()
        preview.start()
        print("Pipeline started. Press q, Esc, or Ctrl+C to stop.")

        counts = dict.fromkeys(("rgb", "left", "right", "thumbnail"), 0)
        last_report = time.monotonic()

        try:
            while pipeline.isRunning() and not STOP.is_set():
                # Preserve the original blocking reads for full-resolution streams.
                for name in ("rgb", "left", "right"):
                    packet = streams[name].read_latest()
                    counts[name] += 1
                    write_jpeg_once(JPEG_PATHS[name], packet)

                # Preview thumbnails never block the capture path.
                thumbnails = {
                    name: streams[name].read_latest(thumbnail=True, block=False)
                    for name in ("rgb", "left", "right")
                }
                rgb_thumbnail = thumbnails["rgb"]
                if rgb_thumbnail is not None:
                    counts["thumbnail"] += 1
                    write_jpeg_once(JPEG_PATHS["thumbnail"], rgb_thumbnail)

                preview.submit(
                    thumbnail=rgb_thumbnail,
                    left=thumbnails["left"],
                    right=thumbnails["right"],
                )

                now = time.monotonic()
                elapsed = now - last_report
                if elapsed >= 1.0:
                    stats = ", ".join(
                        f"{name}={count / elapsed:.1f}"
                        for name, count in counts.items()
                    )
                    print(f"host FPS: {stats}")
                    counts = dict.fromkeys(counts, 0)
                    last_report = now
        finally:
            preview.close()
            pipeline.stop()
            pipeline.wait()
            print("Pipeline stopped.")


if __name__ == "__main__":
    main()
