"""Utility tools for traversing Kinetics dataset"""

from pathlib import Path

from . import _utils

__all__ = ["get_flist"]


def get_flist(split: str) -> Path:
    """Download the file that contains the list of paths of Kinetics 400 dataset.

    Args:
        split: `"train"`, `"test"` or `"val"`

    Returns:
        (Path): Path to the downloaded file.

    Preview of the content

    * "train"

        ```
        train/-JEFz5Z0EWU_000010_000020.mp4
        train/-FjBFRT36pU_000100_000110.mp4
        train/-2-nYjSwo8U_000042_000052.mp4
        ...
        ```

    * "test"

        ```
        test/05bnplpTSq8_000185_000195.mp4
        test/0YWaUPvui5s_000009_000019.mp4
        test/-gGGgY9NWiw_000028_000038.mp4
        ...
        ```

    * "val"

        ```
        val/0tlGOxUQ0Kw_000074_000084.mp4
        val/-05qSkAhM6Y_000205_000215.mp4
        val/-LISB_b8rIw_000049_000059.mp4
        ...
        ```
    """
    vals = ["train", "test", "val"]

    if split not in vals:
        raise ValueError(f"`split` must be one of {vals}")

    return _utils.fetch("kinetics", f"kinetics400.{split}.tsv")
