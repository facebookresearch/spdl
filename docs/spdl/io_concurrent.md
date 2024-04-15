# ``spdl.io``

## High-level APIs

::: spdl.io
    options:
      members:
      - load_media
      - batch_load_image

## Low-level APIs

!!! note

    Each low-level concurrent APIs return ``concurrent.futures.Future`` object.
    You can use [spdl.utils.chain_futures][] and [spdl.utils.wait_futures][] to 
    combine return values from low-level APIs to build complex task without
    blocking.

## Demuxing

::: spdl.io
    options:
      show_source: false
      members:
      - streaming_demux
      - demux_media

## Decoding

::: spdl.io
    options:
      show_source: false
      members:
      - decode_packets
      - decode_packets_nvdec

## Buffer conversion

::: spdl.io
    options:
      show_source: false
      members:
      - convert_frames_cpu
      - convert_frames
