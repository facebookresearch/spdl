"""Utility tools for traversing ImageNet dataset"""

import importlib.resources
import re


def get_mappings():
    """Get the mapping from WordNet ID to class and label.

    1000 IDs from ILSVRC2012 is used. The class indices are the index of
    sorted WordNet ID, which corresponds to most models publicly available.

    Returns:
        dict[str, int]: Mapping from WordNet ID to class index.
        dict[str, Tuple[str]]: Mapping from WordNet ID to list of labels.

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

    path = importlib.resources.files(__package__).joinpath("categories.txt")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line := line.strip():
                class_, wnid, labels = line.split("\t")
                class_ = int(class_)
                class_mapping[wnid] = class_
                label_mapping[class_] = tuple(labels.split(","))
    return class_mapping, label_mapping


def parse_wnid(s: str):
    """Parse a WordNet ID (nXXXXXXXX) from string.

    Args:
        s (str): String to parse

    Returns:
        str: Wordnet ID if found otherwise an exception is raised.
            If the string contain multiple WordNet IDs, the first one is returned.
    """
    if match := re.search(r"n\d{8}", s):
        return match.group(0)
    raise ValueError(f"The given string does not contain WNID: {s}")
