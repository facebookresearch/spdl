# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from spdl.io.lib import _libspdl

__all__ = [
    "get_ffmpeg_log_level",
    "set_ffmpeg_log_level",
    "get_ffmpeg_filters",
    "get_ffmpeg_versions",
]


def get_ffmpeg_filters() -> list[str]:
    """Get the list of available filter names."""
    return _libspdl.get_ffmpeg_filters()


def get_ffmpeg_log_level() -> int:
    """Get the current log level of FFmpeg."""
    return _libspdl.get_ffmpeg_log_level()


def set_ffmpeg_log_level(val: int, /) -> None:
    """Set the log level of FFmpeg.

    Args:
        val: Log level. The larger, the more verbose.

            The following values are common values, the corresponding ``ffmpeg``'s
            ``-loglevel`` option value and description.

            - ``-8`` (``quiet``):
              Print no output.
            - ``0`` (``panic``):
              Something went really wrong and we will crash now.
            - ``8`` (``fatal``):
              Something went wrong and recovery is not possible.
              For example, no header was found for a format which depends
              on headers or an illegal combination of parameters is used.
            - ``16`` (``error``):
              Something went wrong and cannot losslessly be recovered.
              However, not all future data is affected.
            - ``24`` (``warning``):
              Something somehow does not look correct.
              This may or may not lead to problems.
            - ``32`` (``info``):
              Standard information.
            - ``40`` (``verbose``):
              Detailed information.
            - ``48`` (``debug``):
              Stuff which is only useful for libav* developers.
            - ``56`` (``trace``):
              Extremely verbose debugging, useful for libav* development.

    """
    _libspdl.set_ffmpeg_log_level(val)


def get_ffmpeg_versions() -> dict[str, tuple[int, int, int]]:
    """Get the versions of FFmpeg libraries

    Returns:
        dict: mapping from library names to version string,

    .. admonition:: Example:

       >>> get_ffmpeg_versions()
       {'libavcodec': (60, 3, 100), 'libavdevice': (60, 1, 100), 'libavfilter': (9, 3, 100), 'libavformat': (60, 3, 100), 'libavutil': (58, 2, 100)}
    """
    return _libspdl.get_ffmpeg_versions()
