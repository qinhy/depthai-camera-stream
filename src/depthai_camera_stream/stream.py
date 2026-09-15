"""DepthAI camera stream abstraction with independent full-rig calibration."""

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
    """Complete RGB + left + right calibration for one DepthAI device.

    Every CameraStream owns an independent CameraCalibration object, but every
    object reads and returns the complete calibration of the physical camera rig.

    The resolutions below are the resolutions stored in EEPROM calibration,
    returned by CalibrationHandler.getDefaultIntrinsics().
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


@dataclass(kw_only=True, slots=True)
class CameraCalibration:
    """Independent complete calibration reader for one RGB-stereo DepthAI device.

    This object is NOT shared between CameraStream objects.

    Each instance independently:
      1. obtains the DepthAI device,
      2. reads EEPROM calibration once,
      3. reads RGB + left + right calibration,
      4. caches the complete CameraCalibrationResult.

    No CameraStream registration and no calibration role are required.
    """

    pipeline: dai.Pipeline | None

    # Standard OAK RGB-stereo socket layout. Override if the hardware differs.
    rgb_socket: Any = dai.CameraBoardSocket.CAM_A
    left_socket: Any = dai.CameraBoardSocket.CAM_B
    right_socket: Any = dai.CameraBoardSocket.CAM_C

    # IMPORTANT: do not keep a strong reference to dai.Device or
    # CalibrationHandler here. The Pipeline owns the device lifecycle in
    # DepthAI v3. Keeping those pybind handles alive can delay native device
    # destruction/reconnect after the pipeline has been stopped.
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

    @property
    def is_loaded(self) -> bool:
        """Return whether EEPROM calibration has been read and cached."""
        with self._lock:
            return self._data is not None

    @property
    def is_ready(self) -> bool:
        """Return whether the complete calibration result is cached."""
        with self._lock:
            return self._data is not None

    def read_calibration(self) -> CameraCalibrationResult:
        """Return complete RGB + left + right calibration.

        The native Device and CalibrationHandler are deliberately kept only as
        local variables while the EEPROM data is copied into immutable Python
        values. This prevents CameraCalibration from extending the native
        device lifetime.
        """
        with self._lock:
            if self._data is None:
                pipeline = self.pipeline
                if pipeline is None:
                    raise RuntimeError(
                        "CameraCalibration has been released before calibration was read"
                    )

                device = pipeline.getDefaultDevice()
                if device is None:
                    raise RuntimeError("DepthAI default device is not available")

                calib = device.readCalibration()
                self._data = self._build_calibration_result_locked(
                    device=device,
                    calib=calib,
                )

                # Do not assign device/calib to self. Their native handles are
                # released as soon as this method returns.

            return self._data

    def read_calibration_dict(self) -> dict[str, Any]:
        """Return complete calibration as a plain dictionary."""
        return self.read_calibration().to_dict()

    def release(self, *, clear_cache: bool = False) -> None:
        """Release calibration-side state without closing the shared device.

        CameraCalibration does not own the DepthAI device. The containing
        Pipeline owns that lifecycle, so this method never calls device.close().
        """
        with self._lock:
            # Detach from Pipeline so a long-lived CameraCalibration object does
            # not keep an old DepthAI device generation alive. Cached plain
            # Python calibration values remain usable unless explicitly cleared.
            self.pipeline = None
            if clear_cache:
                self._data = None

    def __del__(self) -> None:
        """Best-effort fallback that detaches this object from its Pipeline.

        ``__del__`` must never be the primary lifecycle mechanism: its timing is
        controlled by Python's garbage collector and interpreter shutdown can
        leave objects only partially available.  Explicit ``release()`` remains
        preferred, but this prevents an abandoned CameraCalibration from keeping
        an old Pipeline/device generation alive.
        """
        try:
            self.release()
        except BaseException:
            # Exceptions escaping __del__ are ignored by Python but normally get
            # printed to stderr.  Cleanup during GC/interpreter shutdown must be
            # silent and best-effort.
            pass

    def _build_calibration_result_locked(
        self,
        *,
        device: Any,
        calib: Any,
    ) -> CameraCalibrationResult:

        rgb_intrinsics, rgb_resolution = self._read_default_intrinsics(
            calib,
            self.rgb_socket,
        )
        left_intrinsics, left_resolution = self._read_default_intrinsics(
            calib,
            self.left_socket,
        )
        right_intrinsics, right_resolution = self._read_default_intrinsics(
            calib,
            self.right_socket,
        )

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
            device_id = device.getDeviceInfo().getDeviceId()
        except Exception:
            pass

        try:
            if hasattr(calib, "getBaselineDistance"):
                stereo_baseline_cm = _safe_float(
                    calib.getBaselineDistance(
                        self.left_socket,
                        self.right_socket,
                    )
                )
        except Exception:
            pass

        try:
            if hasattr(calib, "getFov"):
                rgb_fov_deg = _safe_float(
                    calib.getFov(self.rgb_socket)
                )
                left_fov_deg = _safe_float(
                    calib.getFov(self.left_socket)
                )
                right_fov_deg = _safe_float(
                    calib.getFov(self.right_socket)
                )
        except Exception:
            pass

        return CameraCalibrationResult(
            rgb_resolution=rgb_resolution,
            left_resolution=left_resolution,
            right_resolution=right_resolution,
            rgb_intrinsics=rgb_intrinsics,
            left_intrinsics=left_intrinsics,
            right_intrinsics=right_intrinsics,
            left_to_right_extrinsics=self._read_extrinsics(
                calib,
                self.left_socket,
                self.right_socket,
            ),
            left_to_rgb_extrinsics=self._read_extrinsics(
                calib,
                self.left_socket,
                self.rgb_socket,
            ),
            rgb_distortion=self._read_distortion(
                calib,
                self.rgb_socket,
            ),
            left_distortion=self._read_distortion(
                calib,
                self.left_socket,
            ),
            right_distortion=self._read_distortion(
                calib,
                self.right_socket,
            ),
            board_name=board_name,
            product_name=product_name,
            device_id=device_id,
            stereo_baseline_cm=stereo_baseline_cm,
            rgb_fov_deg=rgb_fov_deg,
            left_fov_deg=left_fov_deg,
            right_fov_deg=right_fov_deg,
        )

    @staticmethod
    def _read_default_intrinsics(
        calib: Any,
        socket: Any,
    ) -> tuple[Matrix, tuple[int, int]]:
        """Read EEPROM intrinsics and their native calibration resolution."""
        intrinsics, width, height = calib.getDefaultIntrinsics(socket)

        return (
            _float_matrix(intrinsics),
            (int(width), int(height)),
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
        return _float_list(
            calib.getDistortionCoefficients(socket)
        )


@dataclass(kw_only=True, slots=True)
class CameraStream:
    """Build one MJPEG camera stream and optional thumbnail stream.

    Every CameraStream automatically creates its OWN CameraCalibration object.

    However, read_calibration() always returns the COMPLETE device calibration:
      - RGB intrinsics / distortion / FOV
      - left intrinsics / distortion / FOV
      - right intrinsics / distortion / FOV
      - left -> right extrinsics
      - left -> RGB extrinsics
      - stereo baseline
      - EEPROM/device information

    Therefore rgb.read_calibration(), left.read_calibration(), and
    right.read_calibration() all return equivalent complete rig information,
    while their CameraCalibration objects remain independent.
    """

    pipeline: dai.Pipeline | None
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

    # Full-rig calibration socket mapping.
    calibration_rgb_socket: Any = dai.CameraBoardSocket.CAM_A
    calibration_left_socket: Any = dai.CameraBoardSocket.CAM_B
    calibration_right_socket: Any = dai.CameraBoardSocket.CAM_C

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
    _closed: bool = field(
        init=False,
        default=False,
        repr=False,
    )
    _lock: RLock = field(
        init=False,
        default_factory=RLock,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._validate()

        # IMPORTANT:
        # Every CameraStream gets a NEW calibration object.
        # Nothing is shared between rgb/left/right CameraStream instances.
        self.calibration = CameraCalibration(
            pipeline=self.pipeline,
            rgb_socket=self.calibration_rgb_socket,
            left_socket=self.calibration_left_socket,
            right_socket=self.calibration_right_socket,
        )

    @property
    def has_thumbnail(self) -> bool:
        """Return whether a thumbnail output is configured."""
        return self.thumbnail_size is not None

    @property
    def is_built(self) -> bool:
        """Return whether DepthAI nodes and queues have been created."""
        with self._lock:
            return self._built

    @property
    def is_closed(self) -> bool:
        """Return whether this stream has released its host-side handles."""
        with self._lock:
            return self._closed

    def build(self) -> CameraStream:
        """Create camera, encoder, and output queues."""
        with self._lock:
            if self._closed:
                raise RuntimeError(
                    f"CameraStream {self.name!r} has already been closed"
                )
            if self._built:
                raise RuntimeError(
                    f"CameraStream {self.name!r} has already been built"
                )

        pipeline = self.pipeline
        if pipeline is None:
            raise RuntimeError(
                f"CameraStream {self.name!r} has no pipeline"
            )

        self.camera = pipeline.create(
            dai.node.Camera
        ).build(self.socket)

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

    def close(self) -> None:
        """Release this stream's host-side DepthAI handles.

        This method is intentionally *not* the owner of the shared Pipeline or
        Device, so it never calls pipeline.stop() or device.close(). Stop the
        shared pipeline once in the service/controller, then close every stream.
        Calling close() multiple times is safe.
        """
        with self._lock:
            if self._closed:
                return

            # Drop host queue references first, then node/output references.
            # The Pipeline itself still owns the graph until it is stopped and
            # released by the controller.
            self.thumbnail_queue = None
            self.queue = None

            self.thumbnail_encoder = None
            self.thumbnail_frame = None
            self.encoder = None
            self.frame = None
            self.camera = None

            self.calibration.release()

            # This is the important final detach. A server may keep old
            # CameraStream objects around for status/debugging; they must not
            # keep the old Pipeline (and therefore the device) alive.
            self.pipeline = None

            self._built = False
            self._closed = True

    def __del__(self) -> None:
        """Best-effort automatic cleanup when the stream object is collected.

        The destructor intentionally delegates to ``close()`` only.  It does NOT
        stop or close the shared Pipeline/Device because RGB/left/right streams
        may all use the same Pipeline.  When the final owner releases that
        Pipeline, DepthAI can tear down the device normally.
        """
        try:
            self.close()
        except BaseException:
            # The object may be only partially initialized, or Python may be in
            # interpreter shutdown.  Never allow a destructor exception to leak.
            pass

    def read_calibration(self) -> CameraCalibrationResult:
        """Return COMPLETE RGB + left + right device calibration."""
        return self.calibration.read_calibration()

    def read_calibration_dict(self) -> dict[str, Any]:
        """Return COMPLETE device calibration as a plain dictionary."""
        return self.calibration.read_calibration_dict()

    def read_latest(
        self,
        *,
        thumbnail: bool = False,
        block: bool = True,
    ) -> Any | None:
        """Return the newest host packet currently available.

        When block=True, wait for at least one packet with get(). Then drain
        the queue with tryGet() and return only the newest packet.

        When block=False, return None immediately when no packet is available.
        """
        if self._closed:
            raise RuntimeError(
                f"CameraStream {self.name!r} has been closed"
            )
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
        # Positional arguments are intentional because DepthAI's pybind11 API
        # uses camelCase keyword names.
        frame = self.camera.requestOutput(
            size,
            input_type,
            resize_mode,
            fps,
        )

        pipeline = self.pipeline
        if pipeline is None:
            raise RuntimeError(
                f"CameraStream {self.name!r} has no pipeline"
            )

        encoder = pipeline.create(
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


def shutdown_camera_pipeline(
    pipeline: dai.Pipeline,
    *streams: CameraStream,
) -> None:
    """Stop one shared DepthAI pipeline and release stream references.

    DepthAI v3 makes Pipeline the lifecycle owner. Do not call device.close()
    from individual CameraStream/CameraCalibration objects because several
    streams can share the same default device.

    For the strongest deterministic cleanup, create the pipeline with a
    context manager (``with dai.Pipeline() as pipeline:``) and call this helper
    from the ``finally`` block before leaving that context.
    """
    try:
        is_running = getattr(pipeline, "isRunning", None)
        if callable(is_running) and is_running():
            pipeline.stop()

        wait = getattr(pipeline, "wait", None)
        if callable(wait):
            wait()
    finally:
        for stream in streams:
            stream.close()
