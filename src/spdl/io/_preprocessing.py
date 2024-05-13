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
    scale_width: int | None = None,
    scale_height: int | None = None,
    scale_algo: str = "bicubic",
    scale_mode: str | None = "pad",
    crop_width: int | None = None,
    crop_height: int | None = None,
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
            *Optional:* Trim the video by start and end time.
            This has to match the value passed to demux functions.
            It can be retrieved from `timestamp` attribute of `Packets` object.

        scale_width:
            __Video__, __Image__: *Optional:* Change the resolution of the frame.

        scale_height:
            __Video__, __Image__: *Optional:* Change the resolution of the frame.

        scale_algo:
            __Video__, __Image__: *Optional:* Scaling algorithm.
            See https://ffmpeg.org/ffmpeg-scaler.html for the available values.

        scale_mode:
            __Video__, __Image__: *Optional:* How to handle the different aspect
            ratio when changing the resolution of the frame.

            - `"pad"`: Scale the image so that the entire content of the original
              image is present with padding.
            - `"crop"`: Scale the image first to cover the entire region of the
              output resolution, then crop the excess.
            - `None`: The image is scaled as-is.

        crop_width:
            __Video__, __Image__: *Optional:* Crop the image at center (after scaling).

        crop_height:
            __Video__, __Image__: *Optional:* Crop the image at center (after scaling).

        pix_fmt (str):
            __Video__, __Image__: *Optional:* Change the pixel format.
            Valid values are (`"gray8"`, `"rgba"`, `"rgb24"`, `"yuv444p"`,
            `yuv420p`, `yuv422p`, `nv12`).

        num_frames (int):
            __Video__: *Optional:* Fix the number of output frames by
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

    if scale_width is not None or scale_height is not None:
        w = scale_width or 0
        h = scale_height or 0
        scale = [f"{w=}", f"{h=}", f"flags={scale_algo}"]
        if scale_mode is None:
            parts.append(f"scale={':'.join(scale)}")
        elif scale_mode == "pad":
            scale.append("force_original_aspect_ratio=decrease")
            parts.append(f"scale={':'.join(scale)}")
            parts.append(f"pad={w=}:{h=}:x=-1:y=-1:color={pad_mode or 'black'}")
        elif scale_mode == "crop":
            scale.append("force_original_aspect_ratio=increase")
            parts.append(f"scale={':'.join(scale)}")
            parts.append(f"crop={w=}:{h=}")
        else:
            raise ValueError(
                f"Unexpected `scale_mode` value ({scale_mode}). "
                'Expected values are "pad", "crop", or None.'
            )
    if crop_width is not None or crop_height is not None:
        parts.append(f"crop=w={crop_width or 0}:h={crop_height or 0}")
    if num_frames is not None:
        pad = (
            "tpad=stop=-1:stop_mode=clone"
            if pad_mode is None
            else f"tpad=stop=-1:stop_mode=add:color={pad_mode}"
        )
        parts.append(pad)
        parts.append(f"trim=end_frame={num_frames}")
    if filter_desc is not None:
        parts.append(filter_desc)
    if pix_fmt is not None:
        parts.append(f"format=pix_fmts={pix_fmt}")
    return ",".join(parts)


def get_filter_desc(packets, **filter_args):
    """Get the filter to process the given packets.

    Args:
        packets (Packets): Packet to process.

    Other args:
        filter_args: Passed to [spdl.io.get_audio_filter_desc][] or
            [spdl.io.get_video_filter_desc][].
    """
    match type(packets):
        case _libspdl.AudioPackets:
            return get_audio_filter_desc(timestamp=packets.timestamp, **filter_args)
        case _libspdl.VideoPackets:
            return get_video_filter_desc(timestamp=packets.timestamp, **filter_args)
        case _libspdl.ImagePackets:
            return get_video_filter_desc(timestamp=None, **filter_args)
        case _:
            raise TypeError(f"Unexpected type: {type(packets)}")
