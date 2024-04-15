# IO function examples

The `spdl.io` module implements the core functionalities to load media data into arrays.

Loading media data are consisted of 4 phases.

1. Demux the source media into packets
2. Decode the packets and obtain raw frames
3. Convert the frames to contiguous buffer
4. Cast the buffer to array

The individual functionalities are implemented in C++ with multi-threading, so they can
run free from Python's GIL contention.

The `spdl.io` module exposes these funtionalities as asynchronous functions of two types.
One as coroutines using asyncio and the other as parallel tasks utilizing
`concurrent.futures.Future`.

The asyncio APIs are more flexible. It is easy to compose multiple coroutines and create
a complex task. You can mix different kinds of coroutines, including
ones from other packages. However, coroutines are executable only inside of async loop.

Future-based asynchronous functions can be used in regular Python code without async loop,
however composition of tasks is tricky and less flexible compared to coroutine-based equivalent.

## Async IO

!!! note

    This is just an illustration of how to combine low-level APIs.
    For loading images into buffer, there are [spdl.io.async_load_media][]
    and  [spdl.io.async_batch_load_image][].

### Loading an image

```python
async def load_image(src: src):
    # Demux image
    packets = await spdl.io.async_demux_media("image", src):

    # Decode packets into frames
    frames = await spdl.io.async_decode_packets(packets)

    # Convert the frames into buffer
    buffer = await spdl.io.async_convert_frames(frames)

    # Convert the buffer into NumPy array.
    return spdl.io.to_numpy(buffer)

array = await load_image("foo.jpg")
```

### Loading a batch of images

Loading a batch of images consists of the following steps.

1. Simultaneously decode multiple images.
2. Wait till all the decodings are done.
3. Gather the resulting frames (and check errors).
4. Convert the frames to buffer.

As you can see, one has to wait for the image decoding to be complete
before the frames can be converted to buffer.

When a task has a waiting step in the middle, it becomes difficult to
write the procedure in an efficient and elegant manner, even using
concurrent future, because unless the program execution returns from the
function call, the client code cannot perform other tasks while waiting.
The most performant way is to breakdown the task into smaller functions,
each of which returns a Future object, and let the application code
assemble these functions and Future objects.

```python
# Define a coroutine that decodes a single image into frames (but not to buffer)
async def decode_image(src: str, width: int, height: int, pix_fmt="rgb24"):
    packets = await spdl.io.async_demux_media("image", src):
    # Decode, format and resize
    frames = await spdl.io.async_decode_packets(
        packets, width=width, height=height, pix_fmt=pix_fmt)
    return frames


async def load_image_batch(srcs: List[str], width: int, height: int):
    tasks = [asyncio.create_task(decode_image(src, width, height)) for src in srcs]
    await asyncio.wait(tasks)

    frames = []
    for task, src in zip(tasks, srcs):
        # Error handling. Log the failed job and skip.
        if exception := task.exception():
            print(f"Failed to decode image {src}. Reason: {exception}")
            continue
        frames.append(task.result())

    # Convert a list of image frames into a single buffer as a batch
    buffer = await spdl.io.async_convert_frames(frames)
    return spdl.io.to_numpy(buffer)


array = await load_image_batch(["foo.jpg", "bar.png"], width=121, height=121)
```

### Loading audio/video clips from a source

```python
import spdl.io

# Demux the first 1 second, then next 1 second of the audio.
src, ts = "foo.wav", [(0, 1), (1, 2)]

# Use `spdl.io.async_streaming_demux("video", ...)` for demuxing video
async for packets in spdl.io.async_streaming_demux("audio", "foo.wav", ts):

    # The rest is the same as image decoding
    frames = await spdl.io.async_decode_packets(packets)
    buffer = await spdl.io.async_convert_frames(frames)
    array = spdl.io.to_numpy(buffer)
```


## Concurrent Future

The concurrent version of demux/decode/convert functions return an object of
`concurrent.futures.Future` type. Usually, the client code must call
`Future.result()` to wait until the result is ready before moving on to the
next step, but this makes it cumbersome to chain the subsequent operations,
and makes it difficult to perfom multiple operations concurrently from Python.

So we implemented two helper functions which facilitate chaining the
concurrent operations.

* [``spdl.utils.chain_futures``][spdl.utils.chain_futures] is a decorator which converts Future Generator to
  a function that returns one Future, which is fullfilled when the Future Generator
  is exhausted. The intermediate Futures are automatically chained and called via
  callback function.
* [``spdl.utils.wait_futures``][spdl.utils.wait_futures] can be used when the client code need to wait for multiple
  ``Future`` objects to fullfill before moving onto the next operation.

### Loading an image

```python
@spdl.io.chain_futures
def load_image(src):
    # Chain demux, decode and buffer conversion.
    # The result is `concurrent.futures.Future` object
    packets = yield spdl.io.demux_media("image", src)
    frames = yield spdl.io.decode_packets(packets)
    yield spdl.io.convert_frames(frames)


# Kick off the task
future = load_image(src)

# Wait until the result is ready
buffer = future.result()

# Convert the buffer into NumPy array.
array = spdl.io.to_numpy(buffer)
```

### Loading a batch of image

```python
import spdl.io


@spdl.io.chain_futures
def _decode(src, width, height):
    packets = yield spdl.io.demux_media("image", src)
    yield spdl.io.decode_packets(packets, width=width, height=height)


@spdl.io.chain_futures
def _convert(frames_futures):
    frames = yield spd.io.wait_futures(frames_futures)
    yield spdl.io.convert_frames(frames)


def batch_decode_image(srcs: List[str], width: int = 121, height: int = 121):
    return _convert([_decode(src, width, height) for src in srcs])


# Kick off the task
futures = batch_decode_image(["foo.jpg", "bar.png"])

# Wait
buffer = future.result()

# Convert the buffer into NumPy array.
array = spdl.io.to_numpy(buffer)
```
