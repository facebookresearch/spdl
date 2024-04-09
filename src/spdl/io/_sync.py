from concurrent.futures import Future
from typing import Callable, List, Optional, Tuple, Union

from . import _common

__all__ = [
    "chain_futures",
    "wait_futures",
    "convert_frames_cpu",
    "convert_frames",
    "decode_packets",
    "decode_packets_nvdec",
    "demux_media",
    "streaming_demux",
]


class _cb_chain:
    def __init__(self, set_result, set_exception, *tasks):
        self.set_result = set_result
        self.set_exception = set_exception
        self.tasks = list(tasks)

        self._futures = set()

    def _chain(self, future):
        self._futures.add(future)
        future.add_done_callback(self._cb)

    def _cb(self, fut):
        try:
            result = fut.result()
        except Exception as e:
            self.set_exception(e)
        else:
            if self.tasks:
                self._chain(self.tasks.pop(0)(result))
            else:
                self.set_result(result)
        finally:
            self._futures.discard(fut)


def chain_futures(future: Future, *fn: Callable[..., Future]) -> Future:
    f = Future()
    f.set_running_or_notify_cancel()

    # pyre-ignore: [16]
    f.__spdl_cb_chain = _cb_chain(f.set_result, f.set_exception, *fn)
    f.__spdl_cb_chain._chain(future)
    return f


def wait_futures(futures: List[Future]) -> Future:
    f = Future()
    f.set_running_or_notify_cancel()

    num_futures = len(futures)
    results = [None for _ in range(num_futures)]

    def _cb(future):
        i = futures.index(future)
        results[i] = future.result()

        nonlocal num_futures
        if (num_futures := num_futures - 1) == 0:
            f.set_result(results)

    for future in futures:
        future.add_done_callback(_cb)

    # pyre-ignore: [16]
    f.__spdl_futures = futures

    return f


def streaming_demux(
    media_type: str,
    src: Union[str, bytes],
    timestamps: List[Tuple[float, float]],
    **kwargs,
) -> List[Future]:
    """Demux the given time windows from the source.

    Args:
        media_type: ``"audio"`` or ``"video"``.
        src (str or bytes): Source identifier, such as path or URL.
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
    src: Union[str, bytes],
    timestamp: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> Future:
    """Demux image or one chunk of audio/video region from the source.

    Args:
        media_type: ``"audio"``, ``"video"`` or ``"image"``.
        src (str or bytes): Source identifier, such as path or URL.
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
