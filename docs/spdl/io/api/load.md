# High-level I/O APIs

## Load audio/video/image

### Note

When loading audio/video and when ``timestamps`` option is provided, multiple segments
are loaded from the source.
In this case, when one window is demuxed, this function dispatches the decoding operation
and continue demuxing other windows.
This way, decoding packets from different windows are performed concurrently.

``` mermaid
graph TD
  Source -->|Seek and demux Window1| Packets1 -->|Decode| Frames1 -->|Convert| Buffer1;
  Source -->|Seek and demux Window2| Packets2 -->|Decode| Frames2 -->|Convert| Buffer2;
  Source -->|Seek and demux Window3| Packets3 -->|Decode| Frames3 -->|Convert| Buffer3;
```

::: spdl.io.async_load_audio
::: spdl.io.async_load_video
::: spdl.io.async_load_image

## Load a batch of images

### Note

When loading a batch of images as one buffer, it is more efficient to concurrently
demux, decode and resize, then convert the resulting frames into one buffer.

``` mermaid
graph TD
  Source1 -->|Demux, decode and resize| Frames1 -->|Convert| Buffer;
  Source2 -->|Demux, decode and resize| Frames2 -->|Convert| Buffer;
  Source3 -->|Demux, decode and resize| Frames3 -->|Convert| Buffer;
```

::: spdl.io.async_load_image_batch
