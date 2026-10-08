"""Compact DepthAI camera stream abstraction with full-rig calibration."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from threading import RLock
from typing import cast

import depthai as dai


Size = tuple[int, int]
Matrix = tuple[tuple[float, ...], ...]
Vector = tuple[float, ...]
MjpegOutput = tuple[dai.Node.Output, dai.node.VideoEncoder, dai.MessageQueue]

DEPTHAI_DISTORTION_COEFF_NAMES = (
    "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6",
    "s1", "s2", "s3", "s4", "tau_x", "tau_y",
)


def _safe_float(value: float | int | str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_matrix(value: Sequence[Sequence[float]]) -> Matrix:
    return tuple(tuple(map(float, row)) for row in value)


def _float_vector(value: Sequence[float]) -> Vector:
    return tuple(map(float, value))


@dataclass(frozen=True, slots=True)
class CameraCalibrationResult:
    """Complete RGB + stereo calibration copied from one DepthAI device."""

    rgb_resolution: Size
    left_resolution: Size
    right_resolution: Size
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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(kw_only=True, slots=True)
class CameraCalibration:
    """Read and cache complete calibration without owning the device lifetime."""

    pipeline: dai.Pipeline | None
    rgb_socket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_A
    left_socket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_B
    right_socket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_C

    _data: CameraCalibrationResult | None = field(init=False, default=None, repr=False)
    _lock: RLock = field(init=False, default_factory=RLock, repr=False)

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._data is not None

    @property
    def is_ready(self) -> bool:
        return self.is_loaded

    def read_calibration(self,factory=True) -> CameraCalibrationResult:
        with self._lock:
            if self._data is not None:
                return self._data

            pipeline = self.pipeline
            if pipeline is None:
                raise RuntimeError("CameraCalibration has been released before calibration was read")

            device = pipeline.getDefaultDevice()
            if device is None:
                raise RuntimeError("DepthAI default device is not available")

            if not factory:
                # this sometime got wrong data
                data = self._build_calibration_result_locked(device=device, calib=device.readCalibration())
            else:
                data = self._build_calibration_result_locked(device=device, calib=device.readFactoryCalibration())
            self._data = data
            return data

    def read_calibration_dict(self,factory=True) -> dict[str, object]:
        return self.read_calibration(factory=factory).to_dict()

    def release(self, *, clear_cache: bool = False) -> None:
        with self._lock:
            self.pipeline = None
            if clear_cache:
                self._data = None

    def __del__(self) -> None:
        with suppress(BaseException):
            self.release()

    def _build_calibration_result_locked(
        self, *, device: dai.Device, calib: dai.CalibrationHandler
    ) -> CameraCalibrationResult:
        intrinsics = self._read_default_intrinsics
        rgb_k, rgb_res = intrinsics(calib, self.rgb_socket)
        left_k, left_res = intrinsics(calib, self.left_socket)
        right_k, right_res = intrinsics(calib, self.right_socket)

        board_name = product_name = device_id = None
        baseline = rgb_fov = left_fov = right_fov = None

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
            baseline = _safe_float(calib.getBaselineDistance(self.left_socket, self.right_socket))
        except Exception:
            pass

        try:
            rgb_fov = _safe_float(calib.getFov(self.rgb_socket))
            left_fov = _safe_float(calib.getFov(self.left_socket))
            right_fov = _safe_float(calib.getFov(self.right_socket))
        except Exception:
            pass

        extrinsics, distortion = self._read_extrinsics, self._read_distortion
        return CameraCalibrationResult(
            rgb_resolution=rgb_res,
            left_resolution=left_res,
            right_resolution=right_res,
            rgb_intrinsics=rgb_k,
            left_intrinsics=left_k,
            right_intrinsics=right_k,
            left_to_right_extrinsics=extrinsics(calib, self.left_socket, self.right_socket),
            left_to_rgb_extrinsics=extrinsics(calib, self.left_socket, self.rgb_socket),
            rgb_distortion=distortion(calib, self.rgb_socket),
            left_distortion=distortion(calib, self.left_socket),
            right_distortion=distortion(calib, self.right_socket),
            board_name=board_name,
            product_name=product_name,
            device_id=device_id,
            stereo_baseline_cm=baseline,
            rgb_fov_deg=rgb_fov,
            left_fov_deg=left_fov,
            right_fov_deg=right_fov,
        )

    @staticmethod
    def _read_default_intrinsics(calib: dai.CalibrationHandler, socket: dai.CameraBoardSocket) -> tuple[Matrix, Size]:
        matrix, width, height = calib.getDefaultIntrinsics(socket)
        return _float_matrix(matrix), (int(width), int(height))

    @staticmethod
    def _read_extrinsics(
        calib: dai.CalibrationHandler,
        source: dai.CameraBoardSocket,
        destination: dai.CameraBoardSocket,
    ) -> Matrix:
        return _float_matrix(calib.getCameraExtrinsics(source, destination))

    @staticmethod
    def _read_distortion(calib: dai.CalibrationHandler, socket: dai.CameraBoardSocket) -> Vector:
        return _float_vector(calib.getDistortionCoefficients(socket))


@dataclass(kw_only=True, slots=True)
class CameraStream:
    """Build one MJPEG camera stream with optional thumbnail and calibration."""

    pipeline: dai.Pipeline | None
    name: str
    socket: dai.CameraBoardSocket
    size: Size
    fps: float
    input_type: dai.ImgFrame.Type
    resize_mode: dai.ImgResizeMode

    mjpeg_quality: int = 90
    queue_size: int = 4
    queue_blocking: bool = False
    max_exposure_us: int | None = None

    thumbnail_size: Size | None = None
    thumbnail_fps: float | None = None
    thumbnail_input_type: dai.ImgFrame.Type | None = None
    thumbnail_resize_mode: dai.ImgResizeMode | None = None
    thumbnail_mjpeg_quality: int = 70
    thumbnail_queue_size: int = 1
    thumbnail_queue_blocking: bool = False

    calibration_rgb_socket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_A
    calibration_left_socket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_B
    calibration_right_socket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_C

    calibration: CameraCalibration = field(init=False, repr=False)
    camera: dai.node.Camera | None = field(init=False, default=None, repr=False)
    frame: dai.Node.Output | None = field(init=False, default=None, repr=False)
    encoder: dai.node.VideoEncoder | None = field(init=False, default=None, repr=False)
    queue: dai.MessageQueue | None = field(init=False, default=None, repr=False)
    thumbnail_frame: dai.Node.Output | None = field(init=False, default=None, repr=False)
    thumbnail_encoder: dai.node.VideoEncoder | None = field(init=False, default=None, repr=False)
    thumbnail_queue: dai.MessageQueue | None = field(init=False, default=None, repr=False)
    _built: bool = field(init=False, default=False, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)
    _lock: RLock = field(init=False, default_factory=RLock, repr=False)

    def __post_init__(self) -> None:
        self._validate()
        self.calibration = CameraCalibration(
            pipeline=self.pipeline,
            rgb_socket=self.calibration_rgb_socket,
            left_socket=self.calibration_left_socket,
            right_socket=self.calibration_right_socket,
        )

    @property
    def has_thumbnail(self) -> bool:
        return self.thumbnail_size is not None

    @property
    def is_built(self) -> bool:
        with self._lock:
            return self._built

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    def build(self) -> CameraStream:
        with self._lock:
            if self._closed:
                raise RuntimeError(f"CameraStream {self.name!r} has already been closed")
            if self._built:
                raise RuntimeError(f"CameraStream {self.name!r} has already been built")

        pipeline = self._require_pipeline()
        self.camera = pipeline.create(dai.node.Camera).build(self.socket)
        self._configure_camera()

        self.frame, self.encoder, self.queue = self._create_mjpeg_output(
            self.name, self.size, self.fps, self.input_type, self.resize_mode,
            self.mjpeg_quality, self.queue_size, self.queue_blocking,
        )

        thumbnail_size = self.thumbnail_size
        if thumbnail_size is not None:
            self.thumbnail_frame, self.thumbnail_encoder, self.thumbnail_queue = self._create_mjpeg_output(
                f"{self.name}-thumbnail",
                thumbnail_size,
                self.thumbnail_fps if self.thumbnail_fps is not None else self.fps,
                self.thumbnail_input_type if self.thumbnail_input_type is not None else self.input_type,
                self.thumbnail_resize_mode if self.thumbnail_resize_mode is not None else self.resize_mode,
                self.thumbnail_mjpeg_quality,
                self.thumbnail_queue_size,
                self.thumbnail_queue_blocking,
            )

        self._built = True
        return self

    def close(self) -> None:
        """Release stream handles; the shared Pipeline remains controller-owned."""
        with self._lock:
            if self._closed:
                return

            self.thumbnail_queue = self.queue = None
            self.thumbnail_encoder = self.encoder = None
            self.thumbnail_frame = self.frame = None
            self.camera = None
            self.calibration.release()
            self.pipeline = None
            self._built, self._closed = False, True

    def __del__(self) -> None:
        with suppress(BaseException):
            self.close()

    def read_calibration(self,factory=True) -> CameraCalibrationResult:
        return self.calibration.read_calibration(factory=factory)

    def read_calibration_dict(self) -> dict[str, object]:
        return self.calibration.read_calibration_dict()

    def read_latest(self, *, thumbnail: bool = False, block: bool = True) -> dai.EncodedFrame | None:
        if self._closed:
            raise RuntimeError(f"CameraStream {self.name!r} has been closed")
        if not self._built:
            raise RuntimeError(f"CameraStream {self.name!r} has not been built")

        if thumbnail:
            if not self.has_thumbnail or self.thumbnail_queue is None:
                raise RuntimeError(f"CameraStream {self.name!r} has no thumbnail output")
            queue = self.thumbnail_queue
        else:
            if self.queue is None:
                raise RuntimeError(f"CameraStream {self.name!r} has no output queue")
            queue = self.queue

        packet = cast(dai.EncodedFrame | None, queue.get() if block else queue.tryGet())
        if packet is None:
            return None

        while (newer := cast(dai.EncodedFrame | None, queue.tryGet())) is not None:
            packet = newer
        return packet

    def _require_pipeline(self) -> dai.Pipeline:
        if self.pipeline is None:
            raise RuntimeError(f"CameraStream {self.name!r} has no pipeline")
        return self.pipeline

    def _require_camera(self) -> dai.node.Camera:
        if self.camera is None:
            raise RuntimeError(f"CameraStream {self.name!r} camera has not been created")
        return self.camera

    def _configure_camera(self) -> None:
        if self.max_exposure_us is not None:
            self._require_camera().initialControl.setAutoExposureLimit(self.max_exposure_us)

    def _create_mjpeg_output(
        self,
        name: str,
        size: Size,
        fps: float,
        input_type: dai.ImgFrame.Type,
        resize_mode: dai.ImgResizeMode,
        quality: int,
        queue_size: int,
        queue_blocking: bool,
    ) -> MjpegOutput:
        frame = self._require_camera().requestOutput(size, input_type, resize_mode, fps)
        encoder = self._require_pipeline().create(dai.node.VideoEncoder).build(
            frame,
            frameRate=fps,
            profile=dai.VideoEncoderProperties.Profile.MJPEG,
            quality=quality,
        )
        if hasattr(encoder, "setName"):
            encoder.setName(f"{name}-mjpeg")
        queue = encoder.out.createOutputQueue(maxSize=queue_size, blocking=queue_blocking)
        return frame, encoder, queue

    def _validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")

        self._validate_size("size", self.size)
        self._validate_positive("fps", self.fps)
        self._validate_quality("mjpeg_quality", self.mjpeg_quality)
        self._validate_positive("queue_size", self.queue_size)

        if self.max_exposure_us is not None:
            self._validate_positive("max_exposure_us", self.max_exposure_us)
        if self.thumbnail_fps is not None:
            self._validate_positive("thumbnail_fps", self.thumbnail_fps)

        if self.thumbnail_size is not None:
            self._validate_size("thumbnail_size", self.thumbnail_size)
            self._validate_quality("thumbnail_mjpeg_quality", self.thumbnail_mjpeg_quality)
            self._validate_positive("thumbnail_queue_size", self.thumbnail_queue_size)

    @staticmethod
    def _validate_size(name: str, size: Size) -> None:
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


def shutdown_camera_pipeline(pipeline: dai.Pipeline, *streams: CameraStream) -> None:
    """Stop the shared pipeline, then release all stream-side references."""
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
