# ``spdl.io`` (Async)

## High-level APIs

::: spdl.io
    options:
      members:
      - async_load_media
      - async_batch_load_image

## Low-level APIs

!!! note

    Demuxing, decoding and conversion tasks are queued to the thread pool executor task queue,
    and the executor might start processing them before Python code `await`s the corresponding
    coroutines.

## Demuxing

::: spdl.io
    options:
      show_source: false
      members:
      - async_streaming_demux
      - async_demux_media

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
