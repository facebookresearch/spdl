# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from spdl.io import (
    AudioEncodeConfig,
    CPUStorage,
    CUDAConfig,
    DecodeConfig,
    DemuxConfig,
    VideoEncodeConfig,
)

from .lib import _libspdl, _libspdl_cuda

__all__ = [
    "demux_config",
    "decode_config",
    "video_encode_config",
    "audio_encode_config",
    "cuda_config",
    "cpu_storage",
]


def demux_config(**kwargs) -> DemuxConfig:
    """Customize demuxing behavior.

    Args:
        format (str):
            *Optional* Overwrite format. Can be used if the source file does not have
            a header.

        format_options (dict[str, str]):
            *Optional* Provide demuxer options

        buffer_size (int):
            *Opitonal* Override the size of internal buffer used for demuxing.

    Returns:
        Config object.

    .. admonition:: Example: Loading headeless audio file (raw PCM)

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
       >>> packets = asyncio.run(spdl.io.async_demux_audio(src, demux_config=cfg))
       >>>
    """
    return _libspdl.DemuxConfig(**kwargs)


def decode_config(**kwargs) -> DecodeConfig:
    """Customize decoding behavior.

    Args:
        decoder (str):
            *Optional* Override decoder.

        decoder_options (dict[str, str]):
            *Optional* Provide decoder options

    Returns:
        Config object.

    .. admonition:: Example: Specifying the decoder for H264

       >>> # Use libopenh264 decoder to decode video
       >>> cfg = DecodeConfig(decoder="libopenh264")
       >>>
       >>> frames = await spdl.io.async_decode_packets(
       ...     await spdl.io.async_demux_media("video", src),
       ...     decode_config=cfg)

    .. admonition:: Example: Change the number of threads internal to FFmpeg decoder

       >>> # Let FFmpeg chose the optimal number of threads for decoding.
       >>> # Note: By default, SPDL specifies decoders to use a single thread.
       >>> cfg = DecodeConfig(decoder_options={"threads": "0"})
       >>>
       >>> frames = await spdl.io.async_decode_packets(
       ...     await spdl.io.async_demux_video(src),
       ...     decode_config=cfg)
    """
    return _libspdl.DecodeConfig(**kwargs)


def cuda_config(device_index: int, **kwargs) -> CUDAConfig:
    """Sprcify the CUDA device and memory management.

    Args:
        device_index (int): The device to move the data to.

        stream (int):
            *Optional:* Pointer to a custom CUDA stream. By default, per-thread
            default stream is used.

            The value corresponds to ``uintptr_t`` of CUDA API.

            .. admonition:: Using PyTorch default CUDA stream

                It is possible to provide the same stream as the one used in Python's
                main thread. For example, you can fetch the default CUDA stream that
                PyTorch is using as follow.

                >>> stream = torch.cuda.Stream()
                >>> cuda_stream = stream.cuda_stream

                .. warning::

                   Using the same stream as a model is running might introduce
                   undesired synchronization.

        allocator (tuple[Callable[[int, int, int], int], Callable[[int], None]]):
            *Optional:* A pair of custom CUDA memory allocator and deleter functions.

            .. rubric:: Allocator

            The allocator function takes the following arguments, and
            return the address of the allocated memory.

            - Size: ``int``
            - CUDA device index: ``int``
            - CUDA stream address: ``int`` (``uintptr_t``)

            .. rubric:: Deleter

            The deleter takes the address of memory allocated
            by the allocator and free the memory.


            An example of such functions are PyTorch's
            :py:func:`~torch.cuda.caching_allocator_alloc` and
            :py:func:`~torch.cuda.caching_allocator_delete`.
    """
    return _libspdl_cuda.cuda_config(device_index, **kwargs)


def video_encode_config(
    *,
    height: int,
    width: int,
    frame_rate: tuple[int, int] | None = None,
    pix_fmt: str = "rgb24",
    bit_rate: int = -1,
    compression_level: int = -1,
    qscale: int = -1,
    gop_size: int = -1,
    max_b_frames: int = -1,
    colorspace: str | None = None,
    color_primaries: str | None = None,
    color_trc: str | None = None,
) -> VideoEncodeConfig:
    """Customize encoding behavior.

    Args:
        height,width (int): The size of the image to encode.

        frame_rate: *Optional* The frame rate.
            If not provided, the default value of encoder is used.

        pix_fmt (str): *Optional* The pixel format of the input frames.
            The default ``"rgb24"`` is for RGB array in NHWC format.
            If using format that is more suitable for video production like
            ``"yuv420p"``, the frame data must be converted to the target
            format before it is fed to encoder.

        bit_rate (int): *Optional* Override bit rate.

        compression_level (int): *Optional* Override compression level.

        sqcale (int): *Optional* Override scale.

        gop_size (int): *Optional* Override GOP size.

        max_bframes (int): *Optional* Override maximum number of B-Frames.

        colorspace,color_primmaries,color_trc (str): *Optional* Override the
            color space, color primaries, and color transftransformation
            characteristics (color linearization function). An example value is
            ``"bt709"``.

            .. seealso::

               - https://trac.ffmpeg.org/wiki/colorspace
    """
    return _libspdl.video_encode_config(
        height=height,
        width=width,
        frame_rate=frame_rate,
        pix_fmt=pix_fmt,
        bit_rate=bit_rate,
        compression_level=compression_level,
        qscale=qscale,
        gop_size=gop_size,
        max_b_frames=max_b_frames,
        colorspace=colorspace,
        color_primaries=color_primaries,
        color_trc=color_trc,
    )


def audio_encode_config(
    *,
    num_channels: int,
    sample_rate: int,
    sample_fmt: str = "fltp",
    bit_rate: int = -1,
    compression_level: int = -1,
    qscale: int = -1,
) -> AudioEncodeConfig:
    """Customize encoding behavior.

    Args:

        num_channels: The number of channels.

        sample_rate: The sample rate.

        sample_fmt: The sample format. This corresponds to channel order
            and data type.

            The following list shows the values for each data type.

            - ``"u8"``; Unsigned 8-bit integer.
            - ``"s16"``: Signed 16-bit integer.
            - ``"s32"``: Signed 32-bit integer.
            - ``"flt"``: 32-bit floating point.
            - ``"dbl"``: 63-bit floating point.

            For planar format (a.k.a channel-first), suffix with ``"p"``.

            e.g.

            - ``"fltp"``: 32bit float, channel-first.
            - ``"s16"``: Signed 16-bit integer, channel-last.

        bit_rate (int): *Optional* Override bit rate.

        compression_level (int): *Optional* Override compression level.

        sqcale (int): *Optional* Override scale.
    """
    return _libspdl.audio_encode_config(
        num_channels=num_channels,
        sample_fmt=sample_fmt,
        sample_rate=sample_rate,
        bit_rate=bit_rate,
        compression_level=compression_level,
        qscale=qscale,
    )


def cpu_storage(size: int, pin_memory=True) -> CPUStorage:
    """Allocate a block of memory.

    This function allocates a block of memory. The intended usage is to make
    the data transfer from CPU to GPU faster and overlaps the data tansfer
    and GPU computation.

    .. admonition:: Example: Use page-locked memory for faster CUDA transfer.

       >>> packets = spdl.io.demux_image(sample.path)
       >>> frames = spdl.io.decode_packets(packets)

       >>> size = frames.width * frames.height * 3
       >>> storage = spdl.io.cpu_storage(size, pin_memory=True)

       >>> buffer = spdl.io.convert_frames(frames, storage=storage)
       >>> stream = torch.cuda.Stream(device=0)
       >>> cuda_config = spdl.io.cuda_config(device_index=0, stream=stream.cuda_stream)
       >>> buffer = spdl.io.transfer_buffer(buffer, cuda_config=cuda_config)
       >>> tensor = spdl.io.to_torch(buffer)

    Args:
        size: The size of memory to allocate in bytes.
        pin_memory: If ``True``, the memory region is page-locked, so that GPUs
            can access them independently without help from CPU.

    Returns:
        The resulting memory block.
    """
    if pin_memory:
        return _libspdl_cuda.cpu_storage(size=size)
    else:
        return _libspdl.cpu_storage(size=size)
