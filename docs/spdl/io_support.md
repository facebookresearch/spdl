# Helper structures

## Preprocessing

When using FFmpeg-based decoders, FFmpeg can also apply preprocessing via
[filters](https://ffmpeg.org/ffmpeg-filters.html).

Client code can pass a custom filter string to `decode` functions.
The following fuctions help build filter parameter for common usecases.

!!! note
    Filter is also used for trimming the packets for the user-specified
    timestamp. When demuxing and decoding audio/video for a specific time
    window, packets returned by demuxers contain frames outside of the window,
    because they are necessary to correctly decode frames. This process also
    creates frames outside of the window. Filtering, (`trim` and `atrim`) is
    used to remove these frames.

    So when you create a custom filter for audio/video, make sure that the
    resulting filter removes frames outside of the user specified window.

::: spdl.io.get_audio_filter_desc
::: spdl.io.get_video_filter_desc
::: spdl.io.get_filter_desc


## Configs
::: spdl.io.IOConfig
::: spdl.io.DecodeConfig

## Exceptions

::: spdl.io.AsyncIOFailure
    options:
      show_bases: true
