# depthai-camera-stream

A compact, production-oriented wrapper around the DepthAI v3 `Camera` + `VideoEncoder` API.

The main use case is one physical camera with:

- one primary MJPEG stream;
- an optional thumbnail MJPEG stream from the same camera;
- independent size, FPS, quality, and host queue settings for the thumbnail;
- optional camera exposure limiting;
- no dependency on an external `owner` object or parent class;
- low-latency `read_latest()` queue draining for consumers that prefer freshness over backlog.

## Requirements

- Python 3.10+
- DepthAI 3.10+
- An OAK/DepthAI device for the hardware example

## Install

```bash
python -m pip install -e .
```

For the OpenCV hardware example:

```bash
python -m pip install -e '.[example]'
```

For development:

```bash
python -m pip install -e '.[dev]'
pytest
```

## Minimal example

```python
import depthai as dai

from depthai_camera_stream import CameraStream

with dai.Pipeline() as pipeline:
    rgb = CameraStream(
        pipeline=pipeline,
        name="rgb",
        socket=dai.CameraBoardSocket.CAM_A,
        size=(1920, 1080),
        fps=30,
        input_type=dai.ImgFrame.Type.NV12,
        resize_mode=dai.ImgResizeMode.CROP,
        mjpeg_quality=90,
        queue_size=1,
        queue_blocking=False,
        thumbnail_size=(320, 180),
        thumbnail_fps=5,
        thumbnail_mjpeg_quality=70,
    ).build()

    pipeline.start()

    while pipeline.isRunning():
        # Wait for one main packet, drain any backlog, and return the newest.
        frame = rgb.read_latest()
        print("main JPEG bytes:", len(frame.getData()))

        # The thumbnail may run at a lower FPS, so do not block for it.
        thumbnail = rgb.read_latest(thumbnail=True, block=False)
        if thumbnail is not None:
            print("thumbnail JPEG bytes:", len(thumbnail.getData()))
```

The thumbnail is created with a second `Camera.requestOutput(...)` call on the same camera node. No second physical camera node is created.

## API

### `CameraStream`

Required configuration:

- `pipeline`: existing `dai.Pipeline`
- `name`: human-readable stream name
- `socket`: camera socket, for example `dai.CameraBoardSocket.CAM_A`
- `size`: primary `(width, height)`
- `fps`: primary FPS
- `input_type`: encoder-compatible input, normally `dai.ImgFrame.Type.NV12`
- `resize_mode`: `CROP`, `STRETCH`, or `LETTERBOX`

Primary MJPEG options:

- `mjpeg_quality=90`
- `queue_size=4` (use `1` for the lowest-latency latest-frame pattern)
- `queue_blocking=False`

Optional camera option:

- `max_exposure_us=None`

Optional thumbnail options:

- `thumbnail_size=None` disables the thumbnail
- `thumbnail_fps=None` inherits primary FPS
- `thumbnail_input_type=None` inherits primary input type
- `thumbnail_resize_mode=None` inherits primary resize mode
- `thumbnail_mjpeg_quality=70`
- `thumbnail_queue_size=1`
- `thumbnail_queue_blocking=False`

After `build()`:

- `camera` is the built DepthAI camera node
- `frame` is the primary camera output
- `encoder` is the primary MJPEG encoder
- `queue` is the primary host output queue
- `thumbnail_frame`, `thumbnail_encoder`, and `thumbnail_queue` are populated only when `thumbnail_size` is set
- `read_latest()` waits for (or polls for) a packet, drains stale queued packets, and returns the newest packet

### Low-latency reads

Use `read_latest()` when live latency matters more than processing every frame:

```python
packet = stream.read_latest()
```

The default `block=True` waits for at least one packet and then drains any newer packets already queued. For a lower-FPS optional output such as the RGB thumbnail, use non-blocking mode:

```python
thumbnail = rgb.read_latest(thumbnail=True, block=False)
```

For this pattern, `queue_size=1` and `queue_blocking=False` are recommended so stale host frames cannot accumulate.

## Design notes

`CameraStream` combines configuration and runtime state intentionally. Before `build()`, it is a declarative stream definition. After `build()`, the same object exposes the corresponding DepthAI nodes and queues.

`build()` is deliberately one-shot. Calling it twice raises `RuntimeError`, which protects against accidentally adding duplicate nodes to the pipeline.

## Run the hardware examples

RGB + thumbnail:

```bash
python examples/rgb_with_thumbnail.py
```

RGB + thumbnail + left/right stereo cameras:

```bash
python examples/rgb_with_thumbnail_and_stereo.py
```

The stereo example uses the typical OAK-D socket mapping `CAM_A=RGB`, `CAM_B=LEFT`, and `CAM_C=RIGHT`. Its capture loop only reads encoded MJPEG packets. OpenCV decode, `imshow()`, and `waitKey()` run in a dedicated preview thread with a latest-frame mailbox, so a slow preview drops preview frames instead of building capture latency. The preview is not strict frame synchronization; use DepthAI `Sync` or `StereoDepth` when synchronized stereo pairs are required downstream.

The examples write the first encoded packet from each stream to JPEG files so the MJPEG outputs can be inspected directly.

Stop with `q`, `Esc`, or `Ctrl+C`.

## Testing

The unit tests use a small fake DepthAI API and do not require hardware. They verify:

- main stream construction;
- optional thumbnail construction;
- thumbnail defaults inherited from the primary stream;
- exposure configuration;
- encoder and queue parameters;
- validation failures;
- one-shot build behavior;
- blocking and non-blocking `read_latest()` behavior;
- stale-packet draining and optional-thumbnail reads.

## Notes on MJPEG input

For this package's MJPEG path, use `NV12` for RGB and `YUV400p` for mono/stereo streams. Although DepthAI documentation mentions `GRAY8` support, some device/firmware combinations reject `GRAY8` at the encoder with `Arrived frame type ... is not either NV12 or YUV400p`; `YUV400p` avoids that mismatch.
