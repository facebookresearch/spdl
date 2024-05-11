from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Generic, List, TypeVar

__all__ = ["DataSet", "ImageData", "AudioData"]

Sample = TypeVar("Sample")


@dataclass
class ImageData:
    """Image data sample."""

    src: str
    """Route to the file, relative to the dataset root location."""

    cls: int
    """Class of the image."""


@dataclass
class AudioData:
    """Audio data sample."""

    src: str
    """Route to the file, relative to the dataset root location."""

    sample_rate: int
    """Sample rate."""

    num_frames: int
    """Number of frames in the audio.

    !!! note

        There are cases where different decoders return slightly different number of
        samples. This `num_frames` attribute is mainly used for sorting the dataset.
    """


class DataSet(Generic[Sample]):
    """Dataset interface.

    ``DataSet`` class is a facade for the actual dataset implementation.
    The actual ``DataSet`` instances must be created by dedicated factory functions.

    !!! note "See Also"

        - [spdl.dataset.imagenet.ImageNet][]
        - [spdl.dataset.librispeech.LibriSpeech][]
    """

    def __init__(self, _impl):
        self._impl = _impl

    @property
    def attributes(self):
        """The attributes each dataset sample has.

        This corresponds to the attributes of sample class that each DataSet instance
        handles.

        ??? note "Example"

            ```python
            >>> dataset = spdl.dataset.librispeech.LibriSpeech("test-other")
            >>> dataset.attributes
            ['src', 'num_frames']
            ```

        """
        return self._impl.attributes

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        ??? note "Example"

            ```python
            >>> dataset = spdl.dataset.imagenet.ImageNet(split="train")
            >>> len(dataset)
            1281167
            ```
        """
        return len(self._impl)

    def __iter__(self) -> Iterator[Sample]:
        """Return iterator object that iterates dataset samples.

        ??? note "Example"

            ```python
            >>> dataset = spdl.dataset.librispeech.LibriSpeech("test-other")
            >>> for record in dataset:
            ...     print(record)
            Record(src='LibriSpeech/test-other/2414/128291/2414-128291-0020.flac', sample_rate=16000, num_frames=20000)
            Record(src='LibriSpeech/test-other/7902/96592/7902-96592-0020.flac', sample_rate=16000, num_frames=21040)
            ...
            ```
        """
        return iter(self._impl)

    def __getitem__(self, key: int | slice) -> Sample | List[Sample]:
        """Return the sample at the given key.

        ??? note "Example"

            ```python
            >>> dataset = spdl.dataset.librispeech.LibriSpeech("test-other")
            >>> print(dataset[0])
            Record(src='LibriSpeech/test-other/2414/128291/2414-128291-0020.flac', sample_rate=16000, num_frames=20000)
            >>> print(dataset[:3])
            [Record(src='LibriSpeech/test-other/2414/128291/2414-128291-0020.flac', sample_rate=16000, num_frames=20000), Record(src='LibriSpeech/test-other/7902/96592/7902-96592-0020.flac', sample_rate=16000, num_frames=21040), Record(src='LibriSpeech/test-other/8188/269290/8188-269290-0057.flac', sample_rate=16000, num_frames=23520)]
            ```
        """
        return self._impl[key]

    def __repr__(self):
        return repr(self._impl)

    def shuffle(self) -> None:
        """Shuffle dataset samples in-place.

        ??? note "Example"

            ```python
            >>> dataset = spdl.dataset.librispeech.LibriSpeech("test-other")
            >>> dataset.shuffle()
            >>> print(dataset)
            ╒═════════════════════════════════════════════════════════════════════════════════════╕
            │Dataset: librispeech_test_other                                                      │
            │The number of records: 2939                                                          │
            ╞══════════════════════════════════════════════════════════╤═════════════╤════════════╡
            │ src                                                      │ sample_rate │ num_frames │
            ├──────────────────────────────────────────────────────────┼─────────────┼────────────┤
            │ LibriSpeech/test-other/4852/28312/4852-28312-0019.flac   │       16000 │      54160 │
            │ LibriSpeech/test-other/6432/63722/6432-63722-0048.flac   │       16000 │     113360 │
            │ LibriSpeech/test-other/1688/142285/1688-142285-0041.flac │       16000 │     111040 │
            │                           ...                            │     ...     │    ...     │
            │ LibriSpeech/test-other/3764/168670/3764-168670-0032.flac │       16000 │      64640 │
            │ LibriSpeech/test-other/1998/29455/1998-29455-0009.flac   │       16000 │      59760 │
            │ LibriSpeech/test-other/8461/281231/8461-281231-0002.flac │       16000 │     151520 │
            ╘══════════════════════════════════════════════════════════╧═════════════╧════════════╛
            ```
        """
        self._impl.shuffle()

    def sort(self, attribute: str, desc: bool = False) -> None:
        """Sort dataset by attribute.

        Args:
            attribute: Key to sort the dataset
            desc: If true, the dataset is sorted in descending order, otherwise
                it is sorted in ascending order.

        ??? note "Example"

            ```python
            >>> dataset = spdl.dataset.librispeech.LibriSpeech("test-other")
            >>> print(dataset)
            ╒═════════════════════════════════════════════════════════════════════════════════════╕
            │Dataset: librispeech_test_other                                                      │
            │The number of records: 2939                                                          │
            ╞══════════════════════════════════════════════════════════╤═════════════╤════════════╡
            │ src                                                      │ sample_rate │ num_frames │
            ├──────────────────────────────────────────────────────────┼─────────────┼────────────┤
            │ LibriSpeech/test-other/2414/128291/2414-128291-0020.flac │       16000 │      20000 │
            │ LibriSpeech/test-other/7902/96592/7902-96592-0020.flac   │       16000 │      21040 │
            │ LibriSpeech/test-other/8188/269290/8188-269290-0057.flac │       16000 │      23520 │
            │                           ...                            │     ...     │    ...     │
            │ LibriSpeech/test-other/6070/86744/6070-86744-0018.flac   │       16000 │     538320 │
            │ LibriSpeech/test-other/7018/75789/7018-75789-0029.flac   │       16000 │     539600 │
            │ LibriSpeech/test-other/4294/14317/4294-14317-0014.flac   │       16000 │     552160 │
            ╘══════════════════════════════════════════════════════════╧═════════════╧════════════╛
            >>> dataset.sort("num_frames", desc=True)
            ╒═════════════════════════════════════════════════════════════════════════════════════╕
            │Dataset: librispeech_test_other                                                      │
            │The number of records: 2939                                                          │
            ╞══════════════════════════════════════════════════════════╤═════════════╤════════════╡
            │ src                                                      │ sample_rate │ num_frames │
            ├──────────────────────────────────────────────────────────┼─────────────┼────────────┤
            │ LibriSpeech/test-other/4294/14317/4294-14317-0014.flac   │       16000 │     552160 │
            │ LibriSpeech/test-other/7018/75789/7018-75789-0029.flac   │       16000 │     539600 │
            │ LibriSpeech/test-other/6070/86744/6070-86744-0018.flac   │       16000 │     538320 │
            │                           ...                            │     ...     │    ...     │
            │ LibriSpeech/test-other/8188/269290/8188-269290-0057.flac │       16000 │      23520 │
            │ LibriSpeech/test-other/7902/96592/7902-96592-0020.flac   │       16000 │      21040 │
            │ LibriSpeech/test-other/2414/128291/2414-128291-0020.flac │       16000 │      20000 │
            ╘══════════════════════════════════════════════════════════╧═════════════╧════════════╛
            ```
        """
        self._impl.sort(attribute, desc)


def split(dataset: DataSet, n: int, path_pattern: str):
    """Split the dataset and create new datasets.

    Args:
        n: The number of splits.

        path_pattern: The path pattern to which the split datsets are stored.
            The index will be filled with [str.format][] function, (i.e.
            `path_pattern.format(i)`).

    Returns:
        (List[DataSet]): Split DataSet objects.
    """
    return dataset._impl.split(n, path_pattern)