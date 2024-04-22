# Low-level I/O APIs

The low-level I/O APIs are components to build complex/concurrent decoding sequences.

There are three steps to decode source data into buffer (contiguous memory).

``` mermaid
graph LR
  Source -->|Demux| Packets;
  Packets -->|Decode| Frames;
  Frames -->|Convert| Buffer;
```

The following functions implement each stage

*Demuxing*

- Single sequence [async_demux_media][spdl.io.async_demux_media] ([demux_media][spdl.io.demux_media])
- Multiple sequences [async_streaming_demux][spdl.io.async_streaming_demux] ([streaming_demux][spdl.io.streaming_demux])

*Decoding*

- CPU decoding [async_decode_packets][spdl.io.async_decode_packets] ([decode_packets][spdl.io.decode_packets])
- NVDEC decoding [async_decode_packets_nvdec][spdl.io.async_decode_packets_nvdec] ([decode_packets_nvdec][spdl.io.decode_packets_nvdec])

*Frame conversion*

- [async_convert_frames][spdl.io.async_convert_frames] ([convert_frames][spdl.io.convert_frames])


Please refer to [High-level I/O APIs](./io.md) for examples of APIs built from low-level APIs.

!!! note

    When low-level I/O functions are called, tasks are immediately queued to the task queue of
    the thread pool executor.
    The executor might start processing them before Python code `await`s the corresponding
    coroutines.

## Streaming demuxing

::: spdl.io.async_streaming_demux
::: spdl.io.streaming_demux

## Demuxing

::: spdl.io.async_demux_media
::: spdl.io.demux_media

## Decoding

::: spdl.io.async_decode_packets
::: spdl.io.decode_packets

::: spdl.io.async_decode_packets_nvdec
::: spdl.io.decode_packets_nvdec

## Buffer conversion

::: spdl.io.async_convert_frames
::: spdl.io.convert_frames
