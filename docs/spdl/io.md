# ``spdl.io``

The `spdl.io` module implements the core functionalities to load media data into arrays.

Loading media data are consisted of 4 phases.

1. Demuxing the source media
2. Decoding the packets
3. Converting the packets to contiguous buffer
4. Casting the buffer to array

The individual functionalities are implemented in C++ with multi-threading, so they can
run free from Python's GIL contention.

The `spdl.io` module exposes these funtionalities as asynchronous functions, a.k.a coroutines.
With coroutines, one can execute multiple tasks of different kinds (including the ones from
other packages) concurrently.

You can combine multiple coroutines to create a complex task, and the event loop will monitor
the execution status and make the tasks are executed efficiently.

## Example:

!!! note

    Demuxing, decoding and conversion tasks are immediately queued to the background
    thread pool, even before Python code `await`s them.

### Loading an image

```python
async def load_image(src):
    # Demux image
    packets = await spdl.io.async_demux_image(src):

    # Decode packets into frames
    frames = await spdl.io.async_decode_packets(packets)

    # Convert the frames into buffer
    buffer = await spdl.io.async_convert_frames(frames)

    # Convert the buffer into NumPy array.
    return spdl.io.to_numpy(buffer)

array = await load_image("foo.jpg")
```

### Loading audio/video clips from a source

```python
import spdl.io

# Demux the first 1 second, then next 1 second of the audio.
src, ts = "foo.wav", [(0, 1), (1, 2)]

# Use `spdl.io.async_demux_video` for demuxing video
async for packets in spdl.io.async_demux_audio("foo.wav", ts):

    # The rest is the same as image decoding
    frames = await spdl.io.async_decode_packets(packets)
    buffer = await spdl.io.async_convert_frames(frames)
    array = spdl.io.to_numpy(buffer)
```

### Loading images into a batch

```python
import asyncio
import spdl.io


# Define a coroutine that decodes a single image into frames
async def decode_image(src, width=112, height=112, pix_fmt="rgb24"):
    packets = await spdl.io.async_demux_image(src):
    return await spdl.io.async_decode_packets(
        packets, width=width, height=height, pix_fmt=pix_fmt)


async def batch_decode_image(srcs):
    frames = await asyncio.gather(*[decode_image(src) for src in srcs])
    batch = await spdl.io.async_convert_frames(frames)
    return spdl.io.to_numpy(batch)


array = await batch_decode_image(["foo.jpg", "bar.png"])
# Array with shape [2, 112, 112, 3]
```

## Demuxing

::: spdl.io
    options:
      show_source: false
      members:
      - async_demux_audio
      - async_demux_video
      - async_demux_image

## Decoding

::: spdl.io
    options:
      show_source: false
      members:
      - async_decode_packets
      - async_decode_packets_nvdec

## Buffer conversion

::: spdl.io
    options:
      show_source: false
      members:
      - async_convert_frames_cpu
      - async_convert_frames

## Array conversion

::: spdl.io
    options:
      show_root_toc_entry: false
      members:
      - to_numpy
      - to_torch
      - to_numba
