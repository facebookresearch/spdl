"""Utility tools for traversing ImageNet dataset"""

import re
from pathlib import Path

from . import _utils

__all__ = ["get_flist", "get_mappings", "parse_wnid"]


def get_flist(split: str) -> Path:
    """Download the file that contains the list of paths of ImageNet dataset.

    Args:
        split: `"train"`, `"test"` or `"val"`

    Returns:
        (Path): Path to the downloaded file.

    Preview of the content

    * "train"

        ```
        train/n01440764/n01440764_10042.JPEG
        train/n01440764/n01440764_10027.JPEG
        train/n01440764/n01440764_10293.JPEG
        ...
        ```

    * "test"

        ```
        test/ILSVRC2012_test_00000018.JPEG
        test/ILSVRC2012_test_00000006.JPEG
        test/ILSVRC2012_test_00000194.JPEG
        ...
        ```

    * "val"

        ```
        val/n01440764/ILSVRC2012_val_00006697.JPEG
        val/n01440764/ILSVRC2012_val_00010306.JPEG
        val/n01440764/ILSVRC2012_val_00009346.JPEG
        ...
        ```

    """
    vals = ["train", "test", "val"]

    if split not in vals:
        raise ValueError(f"`split` must be one of {vals}")

    return _utils.fetch("imagenet", f"imagenet.{split}.tsv")


def get_mappings():
    """Get the mapping from WordNet ID to class and label.

    1000 IDs from ILSVRC2012 is used. The class indices are the index of
    sorted WordNet ID, which corresponds to most models publicly available.

    Returns:
        (dict[str, int]): Mapping from WordNet ID to class index.
        (dict[str, Tuple[str]]): Mapping from WordNet ID to list of labels.

    ??? example
        ```python
        >>> class_mapping, label_mapping = get_mappings()
        >>> print(class_mapping["n03709823"])
        636
        >>> print(label_mapping[636])
        ('mailbag', 'postbag')

        ```
    """
    class_mapping = {}
    label_mapping = {}

    path = _utils.fetch("imagenet", "categories.tsv")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line := line.strip():
                class_, wnid, labels = line.split("\t")[:3]
                class_ = int(class_)
                class_mapping[wnid] = class_
                label_mapping[class_] = tuple(labels.split(","))
    return class_mapping, label_mapping


def parse_wnid(s: str):
    """Parse a WordNet ID (nXXXXXXXX) from string.

    Args:
        s (str): String to parse

    Returns:
        (str): Wordnet ID if found otherwise an exception is raised.
            If the string contain multiple WordNet IDs, the first one is returned.
    """
    if match := re.search(r"n\d{8}", s):
        return match.group(0)
    raise ValueError(f"The given string does not contain WNID: {s}")
