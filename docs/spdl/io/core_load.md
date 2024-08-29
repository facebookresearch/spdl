# Core I/O APIs

As seen in ["Overview"](./overview.md) section, loading media into array/tensor takes multiple steps, and core APIs implement these steps.

Each function roughly corresponds to one call into underlying C++ implementation (except generators), and there is no intra-operation concurrency.

## Examples

??? note "Example: Convert audio frames to a contiguous buffer, then cast it to NumPy array."

    ```python
    >>> import spdl.io

    >>> def _load_audio(src):
    ...     packets = spdl.io.demux_audio(src)
    ...     frames = spdl.io.decode_packets(packets)
    ...     buffer = spdl.io.convert_frames(frames)
    ...     array = spdl.io.to_numpy(buffer)
    ...     return array
    >>> array = _load_audio("sample.wav")
    >>>
    ```

??? note "Example: Convert video frames to a contiguous buffer, transfer it to a CUDA device, then cast the resulting buffer to Numba CUDA tensor."

    ```python
    >>> def _load_video_to_numba(src):
    ...     packets = spdl.io.demux_video(src)
    ...     frames = spdl.io.decode_packets(packets)
    ...     buffer = spdl.io.convert_frames(frames, cuda_config=spdl.io.cuda_config(device_index=0))
    ...     tensor = spdl.io.to_numba(buffer)
    ...     return tensor
    >>> tensor = _load_video_to_numba("sample.mp4")
    >>>
    ```

??? note "Example: Convert batch image frames to a contiguous buffer, transfer it to a CUDA device using PyTorch's CUDA caching allocator, then cast the resulting buffer to PyTorch tensor."

    ```python
    >>> import asyncio
    >>> import spdl.io
    >>> import torch
    >>>
    >>> async def _load_image(src):
    ...     packets = await spdl.io.async_demux_media("image", src)
    ...     return await spdl.io.async_decode_packets(packets)
    >>>
    >>> async def _batch_load_image(srcs):
    ...     tasks = [asyncio.create_task(_load_image(src)) for src in srcs]
    ...     frames = await asyncio.gather(*tasks)
    ...
    ...     buffer = await spdl.io.async_convert_frames(
    ...         frames,
    ...         cuda_config=spdl.io.cuda_config(
    ...             device_index=0,
    ...             allocator=(
    ...                 torch.cuda.caching_allocator_alloc,
    ...                 torch.cuda.caching_allocator_delete,
    ...             ),
    ...         ),
    ...     )
    ...     return spdl.io.to_torch(buffer)
    >>> tensor = asyncio.run(_batch_load_image(["sample1.jpg", "sample2.png"]))
    >>>
    ```

## Demuxing

::: spdl.io.demux_audio
::: spdl.io.async_demux_audio
::: spdl.io.demux_video
::: spdl.io.async_demux_video
::: spdl.io.demux_image
::: spdl.io.async_demux_image

## Demuxing (multi)

::: spdl.io.streaming_demux_audio
::: spdl.io.async_streaming_demux_audio
::: spdl.io.streaming_demux_video
::: spdl.io.async_streaming_demux_video

## Decoding

::: spdl.io.decode_packets
::: spdl.io.async_decode_packets
::: spdl.io.decode_packets_nvdec
::: spdl.io.async_decode_packets_nvdec
::: spdl.io.streaming_decode_packets
::: spdl.io.async_streaming_decode_packets
::: spdl.io.decode_image_nvjpeg
::: spdl.io.async_decode_image_nvjpeg

## Frame conversion

::: spdl.io.convert_frames
::: spdl.io.async_convert_frames

## Buffer transfer

::: spdl.io.transfer_buffer
::: spdl.io.async_transfer_buffer
