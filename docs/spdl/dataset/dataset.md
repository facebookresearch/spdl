# DataSet

!!! note

    This module is still in early development, and it does not yet have practical functionality.

`spdl.dataset` module implements catalogs for common datasets.
It provides iterator/map-style interface to datasets.

Unlike conventional dataset implementations, the dataset implemented
in this module does not return tensor/arary object.
Instead it returns the information to load the target data. This includes
(relative) path, time stamp, class ID and so on.

## Representation of a samle in DataSet

::: spdl.dataset
    options:
      show_signature_annotations: true
      show_bases: true
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - ImageData
        - AudioData

## DataSet Reperesentation

::: spdl.dataset
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - DataSet
