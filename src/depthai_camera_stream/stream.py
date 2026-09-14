"""DepthAI camera stream abstraction with independent per-stream calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from threading import RLock
from typing import Any

import depthai as dai


Matrix = tuple[tuple[float, ...], ...]
Vector = tuple[float, ...]

DEPTHAI_DISTORTION_COEFF_NAMES: tuple[str, ...] = (
    "k1",
    "k2",
    "p1",
    "p2",
    "k3",
    "k4",
    "k5",
    "k6",
    "s1",
    "s2",
    "s3",
    "s4",
    "tau_x",
    "tau_y",
)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_matrix(value: Any) -> Matrix:
    return tuple(tuple(float(item) for item in row) for row in value)


def _float_list(value: Any) -> Vector:
    return tuple(float(item) for item in value)


@dataclass(frozen=True, slots=True)
class CameraCalibrationResult:
    """Normalized calibration for one DepthAI camera stream.

    The result contains only plain Python values and is safe to cache,
    serialize, and return through RPC/JSON boundaries.
    """

    resolution: tuple[int, int]
    intrinsics: Matrix
    distortion: Vector

    distortion_coeff_order: tuple[str, ...] = DEPTHAI_DISTORTION_COEFF_NAMES

    board_name: str | None = None
    product_name: str | None = None
    device_id: str | None = None
    fov_deg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary suitable for RPC/JSON serialization."""
        return asdict(self)


@dataclass(kw_only=True, slots=True)
class CameraCalibration:
    """Calibration reader/cache for one camera stream.

    Each CameraStream owns its own CameraCalibration instance. Nothing is
    shared between RGB, left, and right streams.

    EEPROM calibration is loaded lazily on the first calibration operation and
    then cached inside this object.
    """

    pipeline: dai.Pipeline
    socket: Any
    size: tuple[int, int]

    _device: Any = field(init=False, default=None, repr=False)
    _handler: Any = field(init=False, default=None, repr=False)
    _data: CameraCalibrationResult | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _lock: RLock = field(
        init=False,
        default_factory=RLock,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._validate_size("size", self.size)

    @property
    def is_loaded(self) -> bool:
        """Return whether EEPROM calibration has already been loaded."""
        with self._lock:
            return self._handler is not None

    def read_calibration(self) -> CameraCalibrationResult:
        """Read and cache calibration for this camera."""
        with self._lock:
            if self._data is None:
                self._load_once_locked()
                self._data = self._build_result_locked()

            return self._data

    def read_calibration_dict(self) -> dict[str, Any]:
        """Return this camera calibration as a plain dictionary."""
        return self.read_calibration().to_dict()

    def read_extrinsics(self, destination_socket: Any) -> Matrix:
        """Return the transform from this camera to another camera socket."""
        with self._lock:
            self._load_once_locked()
            return _float_matrix(
                self._handler.getCameraExtrinsics(
                    self.socket,
                    destination_socket,
                )
            )

    def read_baseline_cm(self, destination_socket: Any) -> float | None:
        """Return baseline distance to another camera when supported."""
        with self._lock:
            self._load_once_locked()

            if not hasattr(self._handler, "getBaselineDistance"):
                return None

            try:
                return _safe_float(
                    self._handler.getBaselineDistance(
                        self.socket,
                        destination_socket,
                    )
                )
            except Exception:
                return None

    def _load_once_locked(self) -> None:
        if self._handler is not None:
            return

        device = self.pipeline.getDefaultDevice()
        if device is None:
            raise RuntimeError("DepthAI default device is not available")

        self._device = device
        self._handler = device.readCalibration()

    def _build_result_locked(self) -> CameraCalibrationResult:
        if self._device is None or self._handler is None:
            raise RuntimeError("DepthAI calibration has not been loaded")

        width, height = self.size
        calib = self._handler

        board_name: str | None = None
        product_name: str | None = None
        device_id: str | None = None
        fov_deg: float | None = None

        try:
            eeprom = calib.getEepromData()
            board_name = getattr(eeprom, "boardName", None)
            product_name = getattr(eeprom, "productName", None)
        except Exception:
            pass

        try:
            device_id = self._device.getDeviceInfo().getDeviceId()
        except Exception:
            pass

        try:
            if hasattr(calib, "getFov"):
                fov_deg = _safe_float(calib.getFov(self.socket))
        except Exception:
            pass

        return CameraCalibrationResult(
            resolution=self.size,
            intrinsics=_float_matrix(
                calib.getCameraIntrinsics(
                    self.socket,
                    width,
                    height,
                )
            ),
            distortion=_float_list(
                calib.getDistortionCoefficients(
                    self.socket,
                )
            ),
            board_name=board_name,
            product_name=product_name,
            device_id=device_id,
            fov_deg=fov_deg,
        )

    @staticmethod
    def _validate_size(name: str, size: tuple[int, int]) -> None:
        if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
            raise ValueError(
                f"{name} must be a positive (width, height) tuple"
            )


@dataclass(kw_only=True, slots=True)
class CameraStream:
    """Build one MJPEG camera stream and an optional thumbnail stream.

    Every CameraStream automatically owns an independent CameraCalibration
    object configured for the same pipeline, socket, and output size.

    No shared calibration object and no calibration role are required.
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

    calibration: CameraCalibration = field(
        init=False,
        repr=False,
    )

    camera: Any = field(
        init=False,
        default=None,
        repr=False,
    )
    frame: Any = field(
        init=False,
        default=None,
        repr=False,
    )
    encoder: Any = field(
        init=False,
        default=None,
        repr=False,
    )
    queue: Any = field(
        init=False,
        default=None,
        repr=False,
    )

    thumbnail_frame: Any = field(
        init=False,
        default=None,
        repr=False,
    )
    thumbnail_encoder: Any = field(
        init=False,
        default=None,
        repr=False,
    )
    thumbnail_queue: Any = field(
        init=False,
        default=None,
        repr=False,
    )

    _built: bool = field(
        init=False,
        default=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._validate()

        self.calibration = CameraCalibration(
            pipeline=self.pipeline,
            socket=self.socket,
            size=self.size,
        )

    @property
    def has_thumbnail(self) -> bool:
        """Return whether a thumbnail output is configured."""
        return self.thumbnail_size is not None

    @property
    def is_built(self) -> bool:
        """Return whether DepthAI nodes and queues have been created."""
        return self._built

    def build(self) -> CameraStream:
        """Create the camera, encoders, and output queues."""
        if self._built:
            raise RuntimeError(
                f"CameraStream {self.name!r} has already been built"
            )

        self.camera = self.pipeline.create(dai.node.Camera).build(
            self.socket
        )
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
            (
                self.thumbnail_frame,
                self.thumbnail_encoder,
                self.thumbnail_queue,
            ) = self._create_mjpeg_output(
                name=f"{self.name}-thumbnail",
                size=self.thumbnail_size,
                fps=(
                    self.thumbnail_fps
                    if self.thumbnail_fps is not None
                    else self.fps
                ),
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

        self._built = True
        return self

    def read_calibration(self) -> CameraCalibrationResult:
        """Return calibration for this camera stream."""
        return self.calibration.read_calibration()

    def read_calibration_dict(self) -> dict[str, Any]:
        """Return calibration for this stream as a plain dictionary."""
        return self.calibration.read_calibration_dict()

    def read_extrinsics(self, destination_socket: Any) -> Matrix:
        """Return transform from this camera to another camera socket."""
        return self.calibration.read_extrinsics(destination_socket)

    def read_baseline_cm(
        self,
        destination_socket: Any,
    ) -> float | None:
        """Return baseline distance to another camera when supported."""
        return self.calibration.read_baseline_cm(destination_socket)

    def read_latest(
        self,
        *,
        thumbnail: bool = False,
        block: bool = True,
    ) -> Any | None:
        """Return the newest host packet currently available.

        When block=True, wait for at least one packet with get(). Then drain
        the queue with tryGet() so stale frames are discarded.

        When block=False, return None immediately if no packet is available.
        """
        if not self._built:
            raise RuntimeError(
                f"CameraStream {self.name!r} has not been built"
            )

        if thumbnail:
            if not self.has_thumbnail or self.thumbnail_queue is None:
                raise RuntimeError(
                    f"CameraStream {self.name!r} "
                    "has no thumbnail output"
                )
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
            self.camera.initialControl.setAutoExposureLimit(
                self.max_exposure_us
            )

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
        # Positional arguments are intentional here. DepthAI exposes this
        # pybind11 API with camelCase keyword names, while this wrapper uses
        # Python-style snake_case names.
        frame = self.camera.requestOutput(
            size,
            input_type,
            resize_mode,
            fps,
        )

        encoder = self.pipeline.create(
            dai.node.VideoEncoder
        ).build(
            frame,
            frameRate=fps,
            profile=dai.VideoEncoderProperties.Profile.MJPEG,
            quality=quality,
        )

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
        self._validate_quality(
            "mjpeg_quality",
            self.mjpeg_quality,
        )
        self._validate_queue_size(
            "queue_size",
            self.queue_size,
        )

        if (
            self.max_exposure_us is not None
            and self.max_exposure_us <= 0
        ):
            raise ValueError(
                "max_exposure_us must be > 0 when provided"
            )

        if self.thumbnail_size is None:
            if (
                self.thumbnail_fps is not None
                and self.thumbnail_fps <= 0
            ):
                raise ValueError(
                    "thumbnail_fps must be > 0 when provided"
                )
            return

        self._validate_size(
            "thumbnail_size",
            self.thumbnail_size,
        )

        if self.thumbnail_fps is not None:
            self._validate_positive(
                "thumbnail_fps",
                self.thumbnail_fps,
            )

        self._validate_quality(
            "thumbnail_mjpeg_quality",
            self.thumbnail_mjpeg_quality,
        )
        self._validate_queue_size(
            "thumbnail_queue_size",
            self.thumbnail_queue_size,
        )

    @staticmethod
    def _validate_size(
        name: str,
        size: tuple[int, int],
    ) -> None:
        if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
            raise ValueError(
                f"{name} must be a positive "
                "(width, height) tuple"
            )

    @staticmethod
    def _validate_positive(
        name: str,
        value: float,
    ) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be > 0")

    @staticmethod
    def _validate_quality(
        name: str,
        value: int,
    ) -> None:
        if not 0 <= value <= 100:
            raise ValueError(
                f"{name} must be between 0 and 100"
            )

    @staticmethod
    def _validate_queue_size(
        name: str,
        value: int,
    ) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be > 0")
