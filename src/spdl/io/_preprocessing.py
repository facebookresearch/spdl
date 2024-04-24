import math
from typing import Any, Dict, Tuple

from spdl.lib import _libspdl

__all__ = ["get_audio_filter_desc", "get_video_filter_desc", "get_filter_desc"]


def get_audio_filter_desc(
    *,
    sample_rate: int | None = None,
    num_channels: int | None = None,
    sample_fmt: str | None = None,
    timestamp: Tuple[float, float] | None = None,
    num_frames: int | None = None,
    filter_desc: str | None = None,
) -> str:
    """Construct FFmpeg filter expression for preprocessing audio.

    Args:
        sample_rate (int):
            *Optional:* Change the sample rate.

        num_channels (int):
            *Optional:* Change the number of channels.

        sample_fmt (str):
            *Optional:* Change the format of sample.
            Valid values are (`"u8"`, `"u8p"`, `s16`, `s16p`,
            `"s32"`, `"s32p"`, `"flt"`, `"fltp"`, `"s64"`,
            `"s64p"`, `"dbl"`, `"dblp"`).

        timestamp (Tuple[float, float]):
            *Optional:* Trim the audio by start and end time.
            This has to match the value passed to demux functions.
            It can be retrieved from `timestamp` attribute of `Packets` object.

        num_frames (int):
            *Optional:* Fix the number of output frames by
            dropping the exceeding frames or padding with silence.

    Returns:
        Filter description.
    """
    parts = []
    if num_channels is not None:
        parts.append(f"aformat=channel_layouts={num_channels}c")
    if sample_rate is not None:
        parts.append(f"aresample={sample_rate}")

    if timestamp is not None:
        start, end = timestamp
        ts = []
        if not math.isinf(start):
            ts.append(f"start={start}")
        if not math.isinf(end):
            ts.append(f"end={end}")
        if ts:
            parts.append(f"atrim={':'.join(ts)}")

    if num_frames is not None:
        parts.append("apad")
        parts.append(f"atrim=end_sample={num_frames}")
    if filter_desc is not None:
        parts.append(filter_desc)
    if sample_fmt is not None:
        parts.append(f"aformat=sample_fmts={sample_fmt}")

    return ",".join(parts)


def get_video_filter_desc(
    *,
    frame_rate: Tuple[int, int] | None = None,
    timestamp: Tuple[float, float] | None = None,
    width: int | None = None,
    height: int | None = None,
    pix_fmt: str | None = None,
    num_frames: int | None = None,
    pad_mode: str | None = None,
    filter_desc: str | None = None,
) -> str:
    """Construct FFmpeg filter expression for preprocessing video/image.

    Args:
        frame_rate (int):
            __Video__: *Optional:* Change the frame rate.

        timestamp (Tuple[float, float]):
            *Optional:* Trim the audio by start and end time.
            This has to match the value passed to demux functions.
            It can be retrieved from `timestamp` attribute of `Packets` object.

        width (int):
            __Video__, __Image__: *Optional:* Change the resolution of the frame.

        height (int):
            __Video__, __Image__: *Optional:* Change the resolution of the frame.

        pix_fmt (str):
            __Video__, __Image__: *Optional:* Change the pixel format.
            Valid values are (`"gray8"`, `"rgba"`, `"rgb24"`, `"yuv444p"`,
            `yuv420p`, `yuv422p`, `nv12`).

        num_frames (int):
            __Video__, __Image__: *Optional:* Fix the number of output frames by
            dropping the exceeding frames or padding.
            The default behavior when padding is to repeat the last frame.
            This can be changed to fixed color frame with `pad_mode` argument.

        pad_mode (str):
            __Video__, *Optional:* Change the padding frames to the given color.

    Returns:
        Filter description.
    """
    parts = []
    if frame_rate is not None:
        parts.append(f"fps={frame_rate[0]}/{frame_rate[1]}")

    if timestamp is not None:
        start, end = timestamp
        ts = []
        if not math.isinf(start):
            ts.append(f"start={start}")
        if not math.isinf(end):
            ts.append(f"end={end}")
        if ts:
            parts.append(f"trim={':'.join(ts)}")

    if num_frames is not None:
        pad = (
            "tpad=stop=-1:stop_mode=clone"
            if pad_mode is None
            else f"tpad=stop=-1:stop_mode=add:color={pad_mode}"
        )
        parts.append(pad)
        parts.append(f"trim=end_frame={num_frames}")
    if width is not None or height is not None:
        scale = []
        if width is not None:
            scale.append(f"width={width}")
        if height is not None:
            scale.append(f"height={height}")
        parts.append(f"scale={':'.join(scale)}")
    if filter_desc is not None:
        parts.append(filter_desc)
    if pix_fmt is not None:
        parts.append(f"format=pix_fmts={pix_fmt}")
    return ",".join(parts)


def get_filter_desc(packets, filter_args: Dict[str, Any] | None = None):
    """Get the filter to process the given packets.

    Args:
        packets (Packets): Packet to process.

        filter_args: Passed to [spdl.io.get_audio_filter_desc][] or
            [spdl.io.get_video_filter_desc][].
    """
    filter_args = filter_args or {}
    match type(packets):
        case _libspdl.AudioPackets:
            return get_audio_filter_desc(timestamp=packets.timestamp, **filter_args)
        case _libspdl.VideoPackets:
            return get_video_filter_desc(timestamp=packets.timestamp, **filter_args)
        case _libspdl.ImagePackets:
            return get_video_filter_desc(timestamp=None, **filter_args)
        case _:
            raise TypeError(f"Unexpected type: {type(packets)}")
