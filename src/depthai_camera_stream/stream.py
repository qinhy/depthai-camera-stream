"""DepthAI camera stream abstraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import depthai as dai


@dataclass(kw_only=True, slots=True)
class CameraStream:
    """Build one MJPEG camera stream and an optional thumbnail stream.

    Both outputs are produced by the same DepthAI ``Camera`` node. The thumbnail
    is disabled when ``thumbnail_size`` is ``None``.

    The object intentionally contains both configuration and runtime state:
    call :meth:`build` once before accessing the runtime attributes.
    """

    pipeline: dai.Pipeline
    name: str
    socket: Any
    size: tuple[int, int]
    fps: float
    input_type: Any
    resize_mode: Any

    mjpeg_quality: int = 90
    queue_size: int = 4
    queue_blocking: bool = False
    max_exposure_us: int | None = None

    thumbnail_size: tuple[int, int] | None = None
    thumbnail_fps: float | None = None
    thumbnail_input_type: Any | None = None
    thumbnail_resize_mode: Any | None = None
    thumbnail_mjpeg_quality: int = 70
    thumbnail_queue_size: int = 1
    thumbnail_queue_blocking: bool = False

    camera: Any = field(init=False, default=None, repr=False)
    frame: Any = field(init=False, default=None, repr=False)
    encoder: Any = field(init=False, default=None, repr=False)
    queue: Any = field(init=False, default=None, repr=False)

    thumbnail_frame: Any = field(init=False, default=None, repr=False)
    thumbnail_encoder: Any = field(init=False, default=None, repr=False)
    thumbnail_queue: Any = field(init=False, default=None, repr=False)

    _built: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self._validate()

    @property
    def has_thumbnail(self) -> bool:
        """Return whether a thumbnail output is configured."""
        return self.thumbnail_size is not None

    @property
    def is_built(self) -> bool:
        """Return whether DepthAI nodes and queues have been created."""
        return self._built

    def build(self) -> CameraStream:
        """Create camera, encoders, and host queues in the configured pipeline.

        This method is one-shot because rebuilding the same object would add
        duplicate nodes and queues to the pipeline.
        """
        if self._built:
            raise RuntimeError(f"CameraStream {self.name!r} has already been built")

        self.camera = self.pipeline.create(dai.node.Camera).build(self.socket)
        self._configure_camera()

        self.frame, self.encoder, self.queue = self._create_mjpeg_output(
            name=self.name,
            size=self.size,
            fps=self.fps,
            input_type=self.input_type,
            resize_mode=self.resize_mode,
            quality=self.mjpeg_quality,
            queue_size=self.queue_size,
            queue_blocking=self.queue_blocking,
        )

        if self.has_thumbnail:
            self.thumbnail_frame, self.thumbnail_encoder, self.thumbnail_queue = (
                self._create_mjpeg_output(
                    name=f"{self.name}-thumbnail",
                    size=self.thumbnail_size,
                    fps=self.thumbnail_fps if self.thumbnail_fps is not None else self.fps,
                    input_type=(
                        self.thumbnail_input_type
                        if self.thumbnail_input_type is not None
                        else self.input_type
                    ),
                    resize_mode=(
                        self.thumbnail_resize_mode
                        if self.thumbnail_resize_mode is not None
                        else self.resize_mode
                    ),
                    quality=self.thumbnail_mjpeg_quality,
                    queue_size=self.thumbnail_queue_size,
                    queue_blocking=self.thumbnail_queue_blocking,
                )
            )

        self._built = True
        return self

    def read_latest(self, *, thumbnail: bool = False, block: bool = True) -> Any | None:
        """Return the newest host packet currently available.

        When ``block`` is true, wait for at least one packet with ``get()``.
        Afterwards, drain the queue with ``tryGet()`` and return only the newest
        packet.  This is useful for low-latency consumers that prefer dropping
        stale frames over processing a backlog.

        Set ``thumbnail=True`` to read from the optional thumbnail queue.
        With ``block=False``, return ``None`` when no packet is available.
        """
        if not self._built:
            raise RuntimeError(f"CameraStream {self.name!r} has not been built")

        if thumbnail:
            if not self.has_thumbnail or self.thumbnail_queue is None:
                raise RuntimeError(f"CameraStream {self.name!r} has no thumbnail output")
            queue = self.thumbnail_queue
        else:
            queue = self.queue

        packet = queue.get() if block else queue.tryGet()
        if packet is None:
            return None

        while True:
            newer = queue.tryGet()
            if newer is None:
                return packet
            packet = newer

    def _configure_camera(self) -> None:
        if self.max_exposure_us is not None:
            self.camera.initialControl.setAutoExposureLimit(self.max_exposure_us)

    def _create_mjpeg_output(
        self,
        *,
        name: str,
        size: tuple[int, int],
        fps: float,
        input_type: Any,
        resize_mode: Any,
        quality: int,
        queue_size: int,
        queue_blocking: bool,
    ) -> tuple[Any, Any, Any]:
        # Use positional arguments here intentionally.  DepthAI exposes this
        # pybind11 method with camelCase keyword names (for example
        # ``resizeMode``), while our public API uses Pythonic snake_case.
        # Positional arguments avoid coupling this wrapper to binding keyword
        # spelling and match the stable requestOutput(size, type, mode, fps)
        # signature.
        frame = self.camera.requestOutput(
            size,
            input_type,
            resize_mode,
            fps,
        )

        encoder = self.pipeline.create(dai.node.VideoEncoder).build(
            frame,
            frameRate=fps,
            profile=dai.VideoEncoderProperties.Profile.MJPEG,
            quality=quality,
        )

        # A node name is useful in pipeline diagnostics when supported.
        if hasattr(encoder, "setName"):
            encoder.setName(f"{name}-mjpeg")

        queue = encoder.out.createOutputQueue(
            maxSize=queue_size,
            blocking=queue_blocking,
        )

        return frame, encoder, queue

    def _validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")

        self._validate_size("size", self.size)
        self._validate_positive("fps", self.fps)
        self._validate_quality("mjpeg_quality", self.mjpeg_quality)
        self._validate_queue_size("queue_size", self.queue_size)

        if self.max_exposure_us is not None and self.max_exposure_us <= 0:
            raise ValueError("max_exposure_us must be > 0 when provided")

        if self.thumbnail_size is None:
            if self.thumbnail_fps is not None and self.thumbnail_fps <= 0:
                raise ValueError("thumbnail_fps must be > 0 when provided")
            return

        self._validate_size("thumbnail_size", self.thumbnail_size)
        if self.thumbnail_fps is not None:
            self._validate_positive("thumbnail_fps", self.thumbnail_fps)
        self._validate_quality("thumbnail_mjpeg_quality", self.thumbnail_mjpeg_quality)
        self._validate_queue_size("thumbnail_queue_size", self.thumbnail_queue_size)

    @staticmethod
    def _validate_size(name: str, size: tuple[int, int]) -> None:
        if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
            raise ValueError(f"{name} must be a positive (width, height) tuple")

    @staticmethod
    def _validate_positive(name: str, value: float) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be > 0")

    @staticmethod
    def _validate_quality(name: str, value: int) -> None:
        if not 0 <= value <= 100:
            raise ValueError(f"{name} must be between 0 and 100")

    @staticmethod
    def _validate_queue_size(name: str, value: int) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be > 0")
