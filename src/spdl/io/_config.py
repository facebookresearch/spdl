from spdl.io import CUDAConfig, DecodeConfig, DemuxConfig, EncodeConfig
from spdl.lib import _libspdl

__all__ = [
    "demux_config",
    "decode_config",
    "encode_config",
    "cuda_config",
]


def demux_config(**kwargs) -> DemuxConfig:
    """Customize demuxing behavior.

    Other Args:
        format (str):
            *Optional* Overwrite format. Can be used if the source file does not have
            a header.

        format_options (dict[str, str]):
            *Optional* Provide demuxer options

        buffer_size (int):
            *Opitonal* Override the size of internal buffer used for demuxing.

    Returns:
        Config object.

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
        >>> cfg = demux_config(format="s16le")
        >>> packets = asyncio.run(spdl.io.async_demux_media("audio", src, demux_config=cfg))
        >>>
        ```
    """
    return _libspdl.DemuxConfig(**kwargs)


def decode_config(**kwargs) -> DecodeConfig:
    """Customize decoding behavior.

    Other Args:
        decoder (str):
            *Optional* Override decoder.

        decoder_options (dict[str, str]):
            *Optional* Provide decoder options

    Returns:
        Config object.

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


def cuda_config(device_index: int, **kwargs) -> CUDAConfig:
    """Sprcify the CUDA device and memory management.

    Args:
        device_index (int): The device to move the data to.

    Other Args:
        stream (int):
            *Optional:* Pointer to a custom CUDA stream. By default, it uses the
            per-thread default stream.

            The value corresponds to `uintptr_t` of CUDA API.

            !!! note

                Host to device buffer transfer is performed in a thread different than
                Python main thread.

                Since the frame data are available only for the duration of the
                background job, the transfer is performed with synchronization.

                It is possible to provide the same stream as the one used in Python's
                main thread, but it might introduce undesired synchronization.

            ??? note "How to retrieve CUDA stream pointer on PyTorch"

                An example to fetch the default stream from PyTorch.

                ```python
                stream = torch.cuda.Stream()
                cuda_stream = stream.cuda_stream
                ```

        allocator (tuple[Callable[[int, int, int], int], Callable[[int], None]]):
            *Optional:* A pair of custom CUDA memory allcoator and deleter functions.

            The allocator function, takes the following arguments, and
            return the address of the allocated memory.

            - Size: `int`
            - CUDA device index: `int`
            - CUDA stream address: `int` (`uintptr_t`)

            An example of such function is
            [PyTorch's CUDA caching allocator][torch.cuda.caching_allocator_alloc].

            The deleter takes the address of memory allocated
            by the allocator and free the memory.

            An example of such function is
            [PyTorch's CUDA caching allocator][torch.cuda.caching_allocator_delete].
    """
    return _libspdl.CUDAConfig(device_index=device_index, **kwargs)


def encode_config(**kwargs) -> EncodeConfig:
    """Customize encoding behavior.

    Other Args:
        muxer (str): *Optional* Multiplexer (container) format or output device.

        muxer_options (str): *Optional* Multiplexer (container) format or output device.

        encoder (str): *Optional* Override encoder.

        encoder_options (dict[str, str]): *Optional* Provide encoder options.

        format (str): *Optional* Override encoder format. Such as "yuv420p".

        width (int): *Optional* Resize image to the given width.

        height (int): *Optional* Resize image to the given height.

        scale_algo (str): *Optional* The algorithm used to scale the image.

            See `sws_flags` entry at https://ffmpeg.org/ffmpeg-scaler.html#sws_005fflags
            for the available values and the detail.

        filter_desc (str): *Optional* Additional filtering applied before width/height/format conversion.

        bit_rate (int): *Optional* Override bit rate.

        compression_level (int): *Optional* Override compression level.

        sqcale (int): *Optional* Override scale.

        gop_size (int): *Optional* Override GOP size.

        max_bframes (int): *Optional* Override maximum number of B-Frames.
    """
    return _libspdl.EncodeConfig(**kwargs)
