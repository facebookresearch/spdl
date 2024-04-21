from spdl.lib import _libspdl

__all__ = [
    "AsyncIOFailure",
    "IOConfig",
    "DecodeConfig",
    "ThreadPoolExecutor",
]

try:
    _IOConfig = _libspdl.IOConfig
    _DecodeConfig = _libspdl.DecodeConfig
    _ThreadPoolExecutor = _libspdl.ThreadPoolExecutor
except Exception:
    _IOConfig = object
    _DecodeConfig = object
    _ThreadPoolExecutor = object


# Exception class used to signal the failure of C++ op to Python.
# Not exposed to user code.
class AsyncIOFailure(RuntimeError):
    """Exception type used to pass the error message from libspdl."""

    pass


class IOConfig(_IOConfig):
    """Custom IO config.

    Other Args:
        format (str):
            *Optional* Overwrite format. Can be used if the source file does not have
            a header.

        format_options (Dict[str, str]):
            *Optional* Provide demuxer options

        buffer_size (int):
            *Opitonal* Override the size of internal buffer used for demuxing.

    ??? note "Example: Loading headeless audio file (raw PCM)"
        ```python
        # Say, this file contains raw PCM samples.
        # One way to generate such a file is,
        # ffmpeg -f lavfi -i 'sine=duration=3' -f s16le -c:a pcm_s16le sample.raw
        src = "sample.raw"

        # This won't work
        packets = await spdl.io.async_demux_media("audio", src)

        # This works.
        cfg = IOConfig(format="s16le")
        packets = await spdl.io.async_demux_media("audio", src, io_config=cfg)
        ```
    """

    pass


class DecodeConfig(_DecodeConfig):
    """Custom decode config.

    Other Args:
        decoder (str):
            *Optional* Override decoder.

        decoder_options (Dict[str, str]):
            *Optional* Provide decoder options

    ??? note "Example: Specifying the decoder for H264"
        ```python
        # Use libopenh264 decoder to decode video
        cfg = DecodeConfig(decoder="libopenh264")

        frames = await spdl.io.async_decode_packets(
            await spdl.io.async_demux_media("video", src),
            decode_config=cfg)
        ```

    ??? note "Example: Change the number of threads internal to FFmpeg decoder"
        ```python
        # Let FFmpeg chose the optimal number of threads for decoding.
        # Note: By default, SPDL specifies decoders to be single thread.
        cfg = DecodeConfig(decoder_options={"threads": "0"})

        frames = await spdl.io.async_decode_packets(
            await spdl.io.async_demux_media("video", src),
            decode_config=cfg)
        ```
    """

    pass


class ThreadPoolExecutor(_ThreadPoolExecutor):
    """Custom thread pool executor to perform tasks.

    Note:
        This is mainly for testing.

    Args:
        num_threads (int): The number of threads.
        thread_name_prefix (str): The prefix of the thread name.
    """

    pass
