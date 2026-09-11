from __future__ import annotations

import importlib
import sys
import types
from collections import deque
from dataclasses import dataclass

import pytest


class FakeInitialControl:
    def __init__(self) -> None:
        self.max_exposure_us = None

    def setAutoExposureLimit(self, value: int) -> None:
        self.max_exposure_us = value


class FakeQueue:
    def __init__(self, max_size: int, blocking: bool) -> None:
        self.max_size = max_size
        self.blocking = blocking
        self.packets = deque()

    def push(self, *packets: object) -> None:
        self.packets.extend(packets)

    def get(self):
        if not self.packets:
            raise AssertionError("fake blocking get() called with no packet")
        return self.packets.popleft()

    def tryGet(self):
        if not self.packets:
            return None
        return self.packets.popleft()


class FakeEncoderOutput:
    def __init__(self) -> None:
        self.queue = None

    def createOutputQueue(self, *, maxSize: int, blocking: bool) -> FakeQueue:
        self.queue = FakeQueue(maxSize, blocking)
        return self.queue


class FakeFrame:
    pass


class FakeCamera:
    def __init__(self, socket: object) -> None:
        self.socket = socket
        self.initialControl = FakeInitialControl()
        self.requests = []

    def requestOutput(self, *args: object, **kwargs: object) -> FakeFrame:
        self.requests.append((args, kwargs))
        return FakeFrame()


class FakeCameraBuilder:
    def build(self, socket: object) -> FakeCamera:
        return FakeCamera(socket)


class FakeEncoder:
    def __init__(self) -> None:
        self.out = FakeEncoderOutput()
        self.build_args = None
        self.build_kwargs = None
        self.name = None

    def build(self, *args: object, **kwargs: object) -> "FakeEncoder":
        self.build_args = args
        self.build_kwargs = kwargs
        return self

    def setName(self, name: str) -> None:
        self.name = name


@dataclass
class FakePipeline:
    cameras: list
    encoders: list

    def __init__(self) -> None:
        self.cameras = []
        self.encoders = []

    def create(self, node_type: object):
        if node_type is FakeCameraNode:
            builder = FakeCameraBuilderWithTracking(self)
            return builder
        if node_type is FakeVideoEncoderNode:
            encoder = FakeEncoder()
            self.encoders.append(encoder)
            return encoder
        raise AssertionError(f"unexpected node type: {node_type!r}")


class FakeCameraBuilderWithTracking:
    def __init__(self, pipeline: FakePipeline) -> None:
        self.pipeline = pipeline

    def build(self, socket: object) -> FakeCamera:
        camera = FakeCamera(socket)
        self.pipeline.cameras.append(camera)
        return camera


class FakeCameraNode:
    pass


class FakeVideoEncoderNode:
    pass


def install_fake_depthai() -> types.ModuleType:
    fake = types.ModuleType("depthai")
    fake.Pipeline = FakePipeline
    fake.node = types.SimpleNamespace(Camera=FakeCameraNode, VideoEncoder=FakeVideoEncoderNode)
    fake.VideoEncoderProperties = types.SimpleNamespace(
        Profile=types.SimpleNamespace(MJPEG="MJPEG")
    )
    sys.modules["depthai"] = fake
    return fake


@pytest.fixture()
def CameraStream(monkeypatch):
    monkeypatch.syspath_prepend("src")
    install_fake_depthai()

    for name in ["depthai_camera_stream", "depthai_camera_stream.stream"]:
        sys.modules.pop(name, None)

    module = importlib.import_module("depthai_camera_stream")
    return module.CameraStream


def make_stream(CameraStream, **overrides):
    kwargs = dict(
        pipeline=FakePipeline(),
        name="rgb",
        socket="CAM_A",
        size=(1920, 1080),
        fps=30,
        input_type="NV12",
        resize_mode="CROP",
        mjpeg_quality=90,
        queue_size=4,
        queue_blocking=False,
    )
    kwargs.update(overrides)
    return CameraStream(**kwargs)


def test_builds_primary_stream(CameraStream):
    stream = make_stream(CameraStream).build()

    assert stream.is_built is True
    assert stream.has_thumbnail is False
    assert len(stream.pipeline.cameras) == 1
    assert len(stream.pipeline.encoders) == 1

    assert stream.camera.socket == "CAM_A"
    assert stream.camera.requests == [
        (((1920, 1080), "NV12", "CROP", 30), {})
    ]

    encoder = stream.encoder
    assert encoder.build_args == (stream.frame,)
    assert encoder.build_kwargs == {
        "frameRate": 30,
        "profile": "MJPEG",
        "quality": 90,
    }
    assert encoder.name == "rgb-mjpeg"
    assert stream.queue.max_size == 4
    assert stream.queue.blocking is False

    assert stream.thumbnail_frame is None
    assert stream.thumbnail_encoder is None
    assert stream.thumbnail_queue is None


def test_builds_thumbnail_from_same_camera(CameraStream):
    stream = make_stream(
        CameraStream,
        thumbnail_size=(320, 180),
        thumbnail_fps=5,
        thumbnail_mjpeg_quality=65,
        thumbnail_queue_size=2,
        thumbnail_queue_blocking=True,
    ).build()

    assert len(stream.pipeline.cameras) == 1
    assert len(stream.pipeline.encoders) == 2
    assert len(stream.camera.requests) == 2

    assert stream.camera.requests[1] == (
        ((320, 180), "NV12", "CROP", 5),
        {},
    )

    assert stream.thumbnail_encoder.build_args == (stream.thumbnail_frame,)
    assert stream.thumbnail_encoder.build_kwargs == {
        "frameRate": 5,
        "profile": "MJPEG",
        "quality": 65,
    }
    assert stream.thumbnail_encoder.name == "rgb-thumbnail-mjpeg"
    assert stream.thumbnail_queue.max_size == 2
    assert stream.thumbnail_queue.blocking is True


def test_thumbnail_inherits_primary_output_settings(CameraStream):
    stream = make_stream(
        CameraStream,
        input_type="GRAY8",
        resize_mode="LETTERBOX",
        fps=24,
        thumbnail_size=(160, 90),
    ).build()

    assert stream.camera.requests[1] == (
        ((160, 90), "GRAY8", "LETTERBOX", 24),
        {},
    )


def test_thumbnail_can_override_input_and_resize(CameraStream):
    stream = make_stream(
        CameraStream,
        thumbnail_size=(320, 180),
        thumbnail_input_type="THUMB_TYPE",
        thumbnail_resize_mode="STRETCH",
    ).build()

    args, kwargs = stream.camera.requests[1]
    assert kwargs == {}
    assert args[1] == "THUMB_TYPE"
    assert args[2] == "STRETCH"


def test_applies_exposure_limit(CameraStream):
    stream = make_stream(CameraStream, max_exposure_us=12_000).build()
    assert stream.camera.initialControl.max_exposure_us == 12_000


def test_build_is_one_shot(CameraStream):
    stream = make_stream(CameraStream).build()
    with pytest.raises(RuntimeError, match="already been built"):
        stream.build()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"name": ""}, "name must not be empty"),
        ({"size": (0, 1080)}, "size must be"),
        ({"fps": 0}, "fps must be > 0"),
        ({"mjpeg_quality": 101}, "mjpeg_quality must be between"),
        ({"queue_size": 0}, "queue_size must be > 0"),
        ({"max_exposure_us": 0}, "max_exposure_us must be > 0"),
        ({"thumbnail_size": (0, 180)}, "thumbnail_size must be"),
        (
            {"thumbnail_size": (320, 180), "thumbnail_fps": 0},
            "thumbnail_fps must be > 0",
        ),
        (
            {"thumbnail_size": (320, 180), "thumbnail_mjpeg_quality": -1},
            "thumbnail_mjpeg_quality must be between",
        ),
        (
            {"thumbnail_size": (320, 180), "thumbnail_queue_size": 0},
            "thumbnail_queue_size must be > 0",
        ),
    ],
)
def test_validation(CameraStream, overrides, message):
    with pytest.raises(ValueError, match=message):
        make_stream(CameraStream, **overrides)


def test_read_latest_blocks_then_drains_to_newest(CameraStream):
    stream = make_stream(CameraStream, queue_size=1).build()
    stream.queue.push("old", "middle", "newest")

    assert stream.read_latest() == "newest"
    assert stream.queue.tryGet() is None


def test_read_latest_nonblocking_returns_none_when_empty(CameraStream):
    stream = make_stream(CameraStream).build()
    assert stream.read_latest(block=False) is None


def test_read_latest_thumbnail_uses_thumbnail_queue(CameraStream):
    stream = make_stream(CameraStream, thumbnail_size=(320, 180)).build()
    stream.thumbnail_queue.push("thumb-old", "thumb-new")

    assert stream.read_latest(thumbnail=True, block=False) == "thumb-new"


def test_read_latest_requires_built_stream(CameraStream):
    stream = make_stream(CameraStream)
    with pytest.raises(RuntimeError, match="has not been built"):
        stream.read_latest()


def test_read_latest_thumbnail_requires_thumbnail(CameraStream):
    stream = make_stream(CameraStream).build()
    with pytest.raises(RuntimeError, match="has no thumbnail output"):
        stream.read_latest(thumbnail=True)
