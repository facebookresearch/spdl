from spdl.lib import _libspdl

__all__ = [
    "get_ffmpeg_log_level",
    "set_ffmpeg_log_level",
]


def get_ffmpeg_log_level() -> int:
    """Get the current log level of FFmpeg."""
    return _libspdl.get_ffmpeg_log_level()


def set_ffmpeg_log_level(val: int, /) -> None:
    """Set the log level of FFmpeg.

    Args:
        val: Log level. The larger, the more verbose.

            The following values are common values, the corresponding ``ffmpeg``'s
            ``-loglevel`` option value and desription.

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
