"""Utility tools for traversing LibriSpeech dataset"""
from pathlib import Path

from . import _utils

__all__ = ["get_flist"]


def get_flist(split: str) -> Path:
    """Download the file that contains the list of paths of LibriSpeech dataset.

    Args:
        split: Split of the dataset.
            Valid values are `"test-clean"`, `"test-other"`.

    Returns:
        (Path): Path to the downloaded file.

    Preview of the content. They are tab-separated, and the second column is the
    number of samples.

    * "test-clean"

        ```
        LibriSpeech/test-clean/672/122797/672-122797-0033.flac\t20560
        LibriSpeech/test-clean/2094/142345/2094-142345-0041.flac\t22720
        LibriSpeech/test-clean/2830/3980/2830-3980-0026.flac\t23760
        ...
        ```

    * "test-other"

        ```
        LibriSpeech/test-other/2414/128291/2414-128291-0020.flac\t20000
        LibriSpeech/test-other/7902/96592/7902-96592-0020.flac\t21040
        LibriSpeech/test-other/8188/269290/8188-269290-0057.flac\t23520
        ...
        ```
    """
    vals = ["test-clean", "test-other"]
    if split not in vals:
        raise ValueError(f"`split` must be one of {vals}")

    return _utils.fetch("librispeech", f"librispeech.{split}.tsv")
