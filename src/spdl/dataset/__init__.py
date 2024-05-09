"""spdl.dataset module implements catalogs for common datasets.

!!! note

    This module is still in early development, and it does not yet have practical functionality.

It provides iterator/map-style interface to datasets.

Unlike conventional dataset implementations, the dataset implemented
in this module does not return tensor/arary object.
Instead it returns the information to load the target data. This includes
(relative) path, time stamp, class ID and so on.
"""

from ._dataset import AudioData, DataSet, ImageData

__all__ = [
    "DataSet",
    "AudioData",
    "ImageData",
]
