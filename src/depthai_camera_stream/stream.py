"""DepthAI camera stream abstraction with cached RGB-stereo calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from threading import RLock
from typing import Any, Literal

import depthai as dai


CalibrationRole = Literal["rgb", "left", "right"]
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

_REQUIRED_CALIBRATION_ROLES: tuple[CalibrationRole, ...] = (
    "rgb",
    "left",
    "right",
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
    """Normalized RGB-stereo calibration captured from one DepthAI device.

    The result is immutable and contains only plain Python values, so it can be
    safely retained for the lifetime of the camera service. ``to_dict()`` is
    provided for RPC/JSON boundaries that still expect a mapping.
    """

    rgb_resolution: tuple[int, int]
    left_resolution: tuple[int, int]
    right_resolution: tuple[int, int]

    rgb_intrinsics: Matrix
    left_intrinsics: Matrix
    right_intrinsics: Matrix

    left_to_right_extrinsics: Matrix
    left_to_rgb_extrinsics: Matrix

    rgb_distortion: Vector
    left_distortion: Vector
    right_distortion: Vector

    distortion_coeff_order: tuple[str, ...] = DEPTHAI_DISTORTION_COEFF_NAMES
    stereo_translation_units: str = "cm"

    board_name: str | None = None
    product_name: str | None = None
    device_id: str | None = None
    stereo_baseline_cm: float | None = None
    rgb_fov_deg: float | None = None
    left_fov_deg: float | None = None
    right_fov_deg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain mapping suitable for RPC/JSON serialization."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _CalibrationCamera:
    socket: Any
    size: tuple[int, int]


@dataclass(kw_only=True, slots=True)
class CameraCalibration:
    """Cache calibration for one physical RGB-stereo DepthAI device.

    The same instance should be shared by the RGB, left, and right
    :class:`CameraStream` objects. Each stream registers its camera role when
    ``build()`` succeeds.

    EEPROM calibration is read only once. After all three roles are registered,
    a normalized :class:`CameraCalibrationResult` is cached for external/RPC
    access.
    """

    pipeline: dai.Pipeline

    _device: Any = field(init=False, default=None, repr=False)
    _handler: Any = field(init=False, default=None, repr=False)
    _cameras: dict[CalibrationRole, _CalibrationCamera] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )
    _data: CameraCalibrationResult | None = field(init=False, default=None, repr=False)
    _lock: RLock = field(init=False, default_factory=RLock, repr=False)

    @property
    def is_loaded(self) -> bool:
        """Return whether the device EEPROM calibration has been read."""
        with self._lock:
            return self._handler is not None

    @property
    def is_ready(self) -> bool:
        """Return whether RGB, left, and right calibration data is cached."""
        with self._lock:
            return self._data is not None

    @property
    def device(self) -> Any:
        """Return the DepthAI device after calibration loading."""
        with self._lock:
            if self._device is None:
                raise RuntimeError("DepthAI device calibration has not been loaded")
            return self._device

    @property
    def handler(self) -> Any:
        """Return the cached DepthAI CalibrationHandler."""
        with self._lock:
            if self._handler is None:
                raise RuntimeError("DepthAI calibration has not been loaded")
            return self._handler

    def register_camera(
        self,
        *,
        role: CalibrationRole,
        socket: Any,
        size: tuple[int, int],
    ) -> None:
        """Register one camera participating in the RGB-stereo rig.

        The first registration reads the EEPROM calibration exactly once.
        When all three roles are present, the normalized calibration dictionary
        is generated and cached.
        """
        if role not in _REQUIRED_CALIBRATION_ROLES:
            raise ValueError(f"unsupported calibration role: {role!r}")

        CameraStream._validate_size(f"{role}_calibration_size", size)
        camera = _CalibrationCamera(socket=socket, size=size)

        with self._lock:
            existing = self._cameras.get(role)
            if existing is not None:
                if existing.socket != socket or existing.size != size:
                    raise RuntimeError(
                        f"Calibration role {role!r} is already registered "
                        "with a different socket or size"
                    )
            else:
                self._cameras[role] = camera

            self._load_once_locked()

            if all(role_name in self._cameras for role_name in _REQUIRED_CALIBRATION_ROLES):
                self._data = self._build_calibration_result_locked()

    def read_calibration(self) -> CameraCalibrationResult:
        """Return the cached immutable RGB-stereo calibration result.

        This method never reads the device EEPROM. EEPROM calibration is read
        during stream construction/registration and retained in memory.
        """
        with self._lock:
            if self._data is None:
                missing = [
                    role
                    for role in _REQUIRED_CALIBRATION_ROLES
                    if role not in self._cameras
                ]
                raise RuntimeError(
                    "RGB-stereo calibration is not ready; missing camera roles: "
                    + ", ".join(missing)
                )

            return self._data

    def read_calibration_dict(self) -> dict[str, Any]:
        """Return the cached calibration as a plain dictionary.

        This compatibility helper is useful at RPC/JSON boundaries. New Python
        callers should prefer :meth:`read_calibration`.
        """
        return self.read_calibration().to_dict()

    def _load_once_locked(self) -> None:
        if self._handler is not None:
            return

        device = self.pipeline.getDefaultDevice()
        if device is None:
            raise RuntimeError("DepthAI default device is not available")

        self._device = device
        self._handler = device.readCalibration()

    def _build_calibration_result_locked(self) -> CameraCalibrationResult:
        if self._device is None or self._handler is None:
            raise RuntimeError("DepthAI calibration has not been loaded")

        calib = self._handler

        rgb = self._cameras["rgb"]
        left = self._cameras["left"]
        right = self._cameras["right"]

        rgb_width, rgb_height = rgb.size
        left_width, left_height = left.size
        right_width, right_height = right.size

        board_name: str | None = None
        product_name: str | None = None
        device_id: str | None = None
        stereo_baseline_cm: float | None = None
        rgb_fov_deg: float | None = None
        left_fov_deg: float | None = None
        right_fov_deg: float | None = None

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
            if hasattr(calib, "getBaselineDistance"):
                stereo_baseline_cm = _safe_float(
                    calib.getBaselineDistance(
                        left.socket,
                        right.socket,
                    )
                )
        except Exception:
            pass

        try:
            if hasattr(calib, "getFov"):
                rgb_fov_deg = _safe_float(calib.getFov(rgb.socket))
                left_fov_deg = _safe_float(calib.getFov(left.socket))
                right_fov_deg = _safe_float(calib.getFov(right.socket))
        except Exception:
            pass

        return CameraCalibrationResult(
            rgb_resolution=rgb.size,
            left_resolution=left.size,
            right_resolution=right.size,
            rgb_intrinsics=self._read_intrinsics(
                calib,
                rgb.socket,
                rgb_width,
                rgb_height,
            ),
            left_intrinsics=self._read_intrinsics(
                calib,
                left.socket,
                left_width,
                left_height,
            ),
            right_intrinsics=self._read_intrinsics(
                calib,
                right.socket,
                right_width,
                right_height,
            ),
            left_to_right_extrinsics=self._read_extrinsics(
                calib,
                left.socket,
                right.socket,
            ),
            left_to_rgb_extrinsics=self._read_extrinsics(
                calib,
                left.socket,
                rgb.socket,
            ),
            rgb_distortion=self._read_distortion(calib, rgb.socket),
            left_distortion=self._read_distortion(calib, left.socket),
            right_distortion=self._read_distortion(calib, right.socket),
            board_name=board_name,
            product_name=product_name,
            device_id=device_id,
            stereo_baseline_cm=stereo_baseline_cm,
            rgb_fov_deg=rgb_fov_deg,
            left_fov_deg=left_fov_deg,
            right_fov_deg=right_fov_deg,
        )

    @staticmethod
    def _read_intrinsics(
        calib: Any,
        socket: Any,
        width: int,
        height: int,
    ) -> Matrix:
        return _float_matrix(
            calib.getCameraIntrinsics(
                socket,
                width,
                height,
            )
        )

    @staticmethod
    def _read_extrinsics(
        calib: Any,
        source_socket: Any,
        destination_socket: Any,
    ) -> Matrix:
        return _float_matrix(
            calib.getCameraExtrinsics(
                source_socket,
                destination_socket,
            )
        )

    @staticmethod
    def _read_distortion(
        calib: Any,
        socket: Any,
    ) -> Vector:
        return _float_list(calib.getDistortionCoefficients(socket))


@dataclass(kw_only=True, slots=True)
class CameraStream:
    """Build one MJPEG camera stream and an optional thumbnail stream.

    Both outputs are produced by the same DepthAI ``Camera`` node. The thumbnail
    is disabled when ``thumbnail_size`` is ``None``.

    The object intentionally contains both configuration and runtime state:
    call :meth:`build` once before accessing the runtime attributes.

    To cache RGB-stereo calibration, share one :class:`CameraCalibration`
    instance among the RGB, left, and right streams and give each stream its
    corresponding ``calibration_role``.
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

    calibration: CameraCalibration | None = None
    calibration_role: CalibrationRole | None = None

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
        """Create camera, encoders, queues, and register calibration.

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
            )

        # Register only after the stream has been built successfully. The shared
        # CameraCalibration object reads EEPROM once and finalizes its cached
        # dictionary after rgb/left/right have all registered.
        if self.calibration is not None:
            assert self.calibration_role is not None
            self.calibration.register_camera(
                role=self.calibration_role,
                socket=self.socket,
                size=self.size,
            )

        self._built = True
        return self

    def read_calibration(self) -> CameraCalibrationResult:
        """Return the shared cached RGB-stereo calibration dataclass."""
        if self.calibration is None:
            raise RuntimeError(
                f"CameraStream {self.name!r} has no calibration store configured"
            )
        return self.calibration.read_calibration()

    def read_calibration_dict(self) -> dict[str, Any]:
        """Return the shared cached calibration as a plain dictionary."""
        return self.read_calibration().to_dict()

    def read_latest(self, *, thumbnail: bool = False, block: bool = True) -> Any | None:
        """Return the newest host packet currently available.

        When ``block`` is true, wait for at least one packet with ``get()``.
        Afterwards, drain the queue with ``tryGet()`` and return only the newest
        packet. This is useful for low-latency consumers that prefer dropping
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
        # Use positional arguments here intentionally. DepthAI exposes this
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

        if (self.calibration is None) != (self.calibration_role is None):
            raise ValueError(
                "calibration and calibration_role must either both be provided "
                "or both be None"
            )

        if self.calibration is not None and self.calibration.pipeline is not self.pipeline:
            raise ValueError("calibration must use the same pipeline as the CameraStream")

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
