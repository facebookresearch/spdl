import logging
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Tuple, Union

import spdl.utils

from . import _common

__all__ = [
    "convert_frames_cpu",
    "convert_frames",
    "decode_packets",
    "decode_packets_nvdec",
    "demux_media",
    "streaming_demux",
    "load_media",
    "batch_load_image",
    "transfer_buffer_to_cuda",
]

_LG = logging.getLogger(__name__)


def streaming_demux(
    media_type: str,
    src: Union[str, bytes, memoryview],
    timestamps: List[Tuple[float, float]],
    **kwargs,
) -> List[Future]:
    """Demux the given time windows from the source.

    Args:
        media_type: ``"audio"`` or ``"video"``.
        src: Source identifier. If ``str`` type, it is interpreted as a source location,
            such as local file path or URL. If ``bytes`` or ``memoryview`` type, then
            they are interpreted as in-memory data.
        timestamps: List of timestamps.

    Other args:
        format (str): *Optional:* The format detection. Optional.
            Can be used to demux headerless format.
        format_options (Dict[str, str]): *Optional:* Format options.
        buffer_size (int, optional): *Optional:* Change the internal buffer size used to process
            the data at a time.
        adaptor (SourceAdaptor, optional): *Optional:* Adaptor to apply to the `src`.
        executor (ThreadPoolExecutor, optional):
            *Optional:* Custom executor to in which the task is performed.
            By default the task is peformed in demuxer thread pool.

    Returns:
        (List[Future[Packets]]): Futures that wrap audio/video Packets.
    """
    func = _common._get_demux_func(media_type, src)
    return _common._futurize_generator(func, len(timestamps), src, timestamps, **kwargs)


def demux_media(
    media_type: str,
    src: Union[str, bytes, memoryview],
    timestamp: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> Future:
    """Demux image or one chunk of audio/video region from the source.

    Args:
        media_type: ``"audio"``, ``"video"`` or ``"image"``.
        src: Source identifier. If ``str`` type, it is interpreted as a source location,
            such as local file path or URL. If ``bytes`` or ``memoryview`` type, then
            they are interpreted as in-memory data.
        timestamp (Tuple[float, float]): *Audio/video only* Demux the given time window.

    Other args:
        format (str): *Optional:* Overwrite the format detection.
            Can be used to demux headerless format.
        format_options (Dict[str, str]): *Optional:* Format options.
        buffer_size (int): *Optional:* Buffer size in bytes.
        adaptor (SourceAdaptor): *Optional:* Adaptor to apply to the `src`.
        executor (ThreadPoolExecutor): *Optional:* Executor to perform the job.
            By default the job is peformed in demuxer thread pool.

    Returns:
        (Future[Packets]): Future which wraps an audio/video/image Packets object.
    """
    if media_type == "image":
        func = _common._get_demux_func(media_type, src)
        return _common._futurize_task(func, src, **kwargs)

    timestamps = [(0.0, float("inf")) if timestamp is None else timestamp]
    return streaming_demux(media_type, src, timestamps, **kwargs)[0]


def decode_packets(packets, **kwargs) -> Future:
    """Decode packets.

    Args:
        packets (Packets): Packets object.

    Other args:
        decoder (str): *Optional:* Overwrite the decoder.
        decoder_options (Dict[str, str]): *Optional:* Decoder options.
        sample_rate (int): *Optional, audio only:* Change the sample rate.
        num_channels (int): *Optional, audio only:* Change the number of channels.
        sample_fmt (str): *Optional, audio only:* Change the format of sample.
            Valid values are (``"u8"``, ``"u8p"``, ``s16``, ``s16p``,
            ``"s32"``, ``"s32p"``, ``"flt"``, ``"fltp"``, ``"s64"``,
            ``"s64p"``, ``"dbl"``, ``"dblp"``).
        frame_rate (int): *Optional, video only:* Change the frame rate.
        width,height (int): *Optional, video/image only:* Change the resolution of the frame.
        pix_fmt (str): *Optional, video/image only:* Change the pixel format.
            Valid values are ().
        num_frames (int): *Optional, audio/video only:* Fix the number of output frames by
            dropping the exceeding frames or padding.
            For audio, silence is added. For video, by default the last frame is
            repeated.
        pad_mode (str): *Optional, video only:* Change the padding frames to the given color.
        executor (ThreadPoolExecutor): *Optional:* Executor to perform the job.
            By default the job is peformed in decode thread pool.

    Returns:
        (Future[FFmpegFrames]): Future which wraps a Frames object.
            The type of the returned object corresponds to the input Packets type.

            - ``AudioPackets`` -> ``AudioFFmpegFrames``

            - ``VideoPackets`` -> ``VideoFFmpegFrames``

            - ``ImagePackets`` -> ``ImageFFmpegFrames``
    """
    func = _common._get_decoding_func(packets)
    return _common._futurize_task(func, packets, **kwargs)


def decode_packets_nvdec(packets, cuda_device_index, **kwargs) -> Future:
    """Decode packets with NVDEC.

    Args:
        packets (Packet): Packets object.
        cuda_device_index (int): The CUDA device to use for decoding.

    Other args:
        crop_left,crop_top,crop_right,crop_bottom (int):
            *Optional:* Crop the given number of pixels from each side.
        width,height (int): *Optional:* Resize the frame. Resizing is done after
            cropping.
        pix_fmt (str or ``None``): *Optional:* Change the format of the pixel.
            Supported value is ``"rgba"``. Default: ``"rgba"``.
        executor (ThreadPoolExecutor): *Optional:* Executor to perform the job.
            By default the job is peformed in decode thread pool.

    Returns:
        (Future[NvDecFrames]): Future that wraps a Frame object.
            The type of the returned object corresponds to the input Packets type.

            - ``VideoPackets`` -> ``VideoNvDecFrames``

            - ``ImagePackets`` -> ``ImageNvDecFrames``
    """
    func = _common._get_nvdec_decoding_func(packets)
    return _common._futurize_task(
        func, packets, cuda_device_index=cuda_device_index, **kwargs
    )


def convert_frames_cpu(frames, executor=None) -> Future:
    """Convert the frames to buffer.

    Args:
        frames (CPUFrames): Frames object.
            If the frame data are not CPU, then the conversion will fail.

    Other args:
        executor (ThreadPoolExecutor):
            *Optional:* Executor to run the conversion.
            By default, the conversion is performed on
            demuxer thread pool with higher priority than demuxing.

    Returns:
        (Future[Buffer]): Future which wraps a Buffer object.
            The type of the returned object corresponds to the input Packets type.

            - ``FFmpegAudioFrames`` -> ``CPUBuffer``

            - ``FFmpegVideoFrames`` -> ``CPUBuffer``

            - ``FFmpegImageFrames`` -> ``CPUBuffer``

            - ``List[FFmpegImageFrames]`` -> ``CPUBuffer``


    """
    func = _common._get_cpu_conversion_func(frames)
    return _common._futurize_task(func, frames, index=None, executor=executor)


def convert_frames(
    frames,
    executor=None,
) -> Future:
    """Convert the frames to buffer.

    Args:
        frames (Frames): Frames object.

    Other args:
        executor (ThreadPoolExecutor):
            *Optional:* Executor to run the conversion. By default, the conversion is performed on
            demuxer thread pool with higher priority than demuxing.

    Returns:
        (Future[Buffer]): Future what wraps a Buffer object.

            The buffer will be created on the device where the frame data are.

            - ``FFmpegAudioFrames`` -> ``CPUBuffer``

            - ``FFmpegVideoFrames`` -> ``CPUBuffer`` or ``CUDABuffer``

            - ``FFmpegImageFrames`` -> ``CPUBuffer`` or ``CUDABuffer``

            - ``NvDecVideoFrames`` -> ``CUDABuffer``

            - ``NvDecImageFrames`` -> ``CUDABuffer``

            - ``List[FFmpegImageFrames]`` -> ``CPUBuffer``

            - ``List[NvDecImageFrames]`` -> ``CUDABuffer``
    """
    func = _common._get_conversion_func(frames)
    return _common._futurize_task(func, frames, index=None, executor=executor)


def transfer_buffer_to_cuda(buffer, cuda_device_index: int):
    """Move the buffer data from CPU to CUDA.

    Args:
        buffer (Buffer): Buffer object.
        cuda_device_index (int): The CUDA device to move the data to.

    Returns:
        (Future[Buffer]): Future which wraps a CUDABuffer object.
    """
    func = _common._get_convert_to_cuda_func()
    return _common._futurize_task(func, buffer, cuda_device_index=cuda_device_index)


################################################################################
# High-level APIs
################################################################################


@spdl.utils.chain_futures
def load_media(
    media_type: str,
    src: Union[str, bytes, memoryview],
    demux_options: Optional[Dict[str, Any]] = None,
    decode_options: Optional[Dict[str, Any]] = None,
    convert_options: Optional[Dict[str, Any]] = None,
    use_nvdec: bool = False,
):
    """Load the given media into buffer.

    This function combines ``demux_media``, ``decode_packets`` (or
    ``decode_packets_nvdec``) and ``convert_frames`` and load media
    into buffer.

    ??? example
        ```python
        future = load_media(
            "image",
            "test.jpg",
            deocde_options={
                "width": 124,
                "height": 96,
                "pix_fmt": "rgb24",
            })
        buffer = future.result()  # blocking wait
        array = spdl.io.to_numpy(buffer)
        # An array with shape HWC==[96, 124, 3]
        ```

    Args:
        media_type: ``"audio"``, ``"video"`` or ``"image"``.

        src: Source identifier. If ``str`` type, it is interpreted as a source location,
            such as local file path or URL. If ``bytes`` or ``memoryview`` type, then
            they are interpreted as in-memory data.

        demux_options (Dict[str, Any]):
            *Optional:* Demux options passed to [spdl.io.async_demux_media][].

        decode_options (Dict[str, Any]):
            *Optional:* Decode options passed to [spdl.io.async_decode_packets][].

        convert_options (Dict[str, Any]):
            *Optional:* Convert options passed to [spdl.io.async_convert_frames][].

        use_nvdec:
            *Optional:* If True, use NVDEC to decode the media.

    Returns:
        (Buffer): An object implements buffer protocol.
            To be passed to casting functions like [spdl.io.to_numpy][],
            [spdl.io.to_torch][] or [spdl.io.to_numba][].
    """
    demux_options = demux_options or {}
    decode_options = decode_options or {}
    convert_options = convert_options or {}
    packets = yield demux_media(media_type, src, **demux_options)
    if use_nvdec:
        frames = yield decode_packets_nvdec(packets, **decode_options)
    else:
        frames = yield decode_packets(packets, **decode_options)
    yield convert_frames(frames, **convert_options)


def _check_arg(var, key, decode_options):
    if var is not None:
        if key in decode_options:
            raise ValueError(
                f"`{key}` is given but also specified in `decode_options`."
            )
        decode_options[key] = var


def batch_load_image(
    srcs: List[Union[str, bytes]],
    width: int | None,
    height: int | None,
    pix_fmt: str | None = "rgb24",
    demux_options: Optional[Dict[str, Any]] = None,
    decode_options: Optional[Dict[str, Any]] = None,
    convert_options: Optional[Dict[str, Any]] = None,
    strict: bool = True,
):
    """Batch load images.

    ??? example
        ```python
        srcs = [
            "test1.jpg",
            "test1.png",
        ]
        future = batch_load_image(
            srcs,
            width=124,
            height=96,
            pix_fmt="rgb24",
        )
        buffer = future.result()  # blocking wait
        array = spdl.io.to_numpy(buffer)
        # An array with shape HWC==[2, 96, 124, 3]
        ```

    Args:
        srcs: List of source identifiers.

        width: *Optional:* Resize the frame.

        height: *Optional:* Resize the frame.

        pix_fmt:
            *Optional:* Change the format of the pixel.

        demux_options (Dict[str, Any]):
            *Optional:* Demux options passed to [spdl.io.demux_media][].

        decode_options (Dict[str, Any]):
            *Optional:* Decode options passed to [spdl.io.decode_packets][].

        convert_options (Dict[str, Any]):
            *Optional:* Convert options passed to [spdl.io.convert_frames][].

        strict:
            *Optional:* If True, raise an error if any of the images failed to load.

    Returns:
        (Buffer): An object implements buffer protocol.
            To be passed to casting functions like [spdl.io.to_numpy][],
            [spdl.io.to_torch][] or [spdl.io.to_numba][].
    """
    if not srcs:
        raise ValueError("`srcs` must not be empty.")

    demux_options = demux_options or {}
    decode_options = decode_options or {}
    convert_options = convert_options or {}

    _check_arg(width, "width", decode_options)
    _check_arg(height, "height", decode_options)
    _check_arg(pix_fmt, "pix_fmt", decode_options)

    @spdl.utils.chain_futures
    def _decode(src):
        packets = yield demux_media("image", src, **demux_options)
        yield decode_packets(packets, **decode_options)

    @spdl.utils.chain_futures
    def _convert(frames_futures):
        frames = yield spdl.utils.wait_futures(frames_futures, strict=strict)
        yield spdl.io.convert_frames(frames, **convert_options)

    return _convert([_decode(src) for src in srcs])
