from spdl.lib import _libspdl

__all__ = [
    "AsyncIOFailure",
    "DemuxConfig",
    "DecodeConfig",
    "Executor",
]


# Exception class used to signal the failure of C++ op to Python.
# Not exposed to user code.
class AsyncIOFailure(RuntimeError):
    """Exception type used to pass the error message from libspdl."""

    pass


def DemuxConfig(**kwargs):
    """Customize demuxing behavior.

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
        >>> import asyncio
        >>> import spdl.io
        >>>
        >>> # Say, this file contains raw PCM samples.
        >>> # One way to generate such a file is,
        >>> # ffmpeg -f lavfi -i 'sine=duration=3' -f s16le -c:a pcm_s16le sample.raw
        >>> src = "sample.raw"
        >>>
        >>> # This won't work
        >>> # packets = asyncio.run(spdl.io.async_demux_media("audio", src))
        >>>
        >>> # This works.
        >>> cfg = DemuxConfig(format="s16le")
        >>> packets = asyncio.run(spdl.io.async_demux_media("audio", src, io_config=cfg))
        >>>
        ```
    """
    return _libspdl.DemuxConfig(**kwargs)


def DecodeConfig(**kwargs):
    """Customize decoding behavior.

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
    return _libspdl.DecodeConfig(**kwargs)


def Executor(num_threads: int, thread_name_prefix: str):
    """Custom thread pool executor.

    Args:
        num_threads: The size of the executor thread pool.
        thread_name_prefix: The prefix of the thread names in the thread pool.

    ??? note "Example: Specifying custom thread pool"
        ```python
        # Use a thread pool different from default one
        exec = Executor(num_threads=10, thread_name_prefix="custom_exec")

        packets = await spdl.io.async_demux_media("video", src, executor=exec)
        frames = await spdl.io.async_decode_packets(packets, executor=exec)
        buffer = await spdl.io.async_convert_frames(frames, executor=exec)
        ```
    """
    return _libspdl.ThreadPoolExecutor(num_threads, thread_name_prefix)
