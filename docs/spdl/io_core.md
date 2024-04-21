# Low-level I/O APIs

!!! note

    When low-level I/O functions are called, tasks are immediately queued to the task queue of
    the thread pool executor.
    The executor might start processing them before Python code `await`s the corresponding
    coroutines.

## Demuxing

::: spdl.io.async_streaming_demux
::: spdl.io.streaming_demux

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
