#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Example demonstrating a WebDataset-like pipeline using SPDL.

This example shows how to build a data loading pipeline similar to WebDataset
that processes tar files containing image and label pairs. The pipeline:

1. Reads tar files containing samples with paired files (e.g., `sample001.jpg`, `sample001.txt`)
2. Groups files by their base name (key) into samples
3. Decodes images and processes labels
4. Batches samples for efficient processing

WebDataset is a popular library for PyTorch that stores datasets in tar archives,
making it efficient for streaming large datasets. This example demonstrates how
to achieve similar functionality using SPDL's :py:func:`spdl.io.iter_tarfile`
function combined with :py:class:`~spdl.pipeline.Pipeline`.

**Example tar file structure:**

.. code-block:: text

   dataset.tar
   ├── sample001.jpg
   ├── sample001.txt
   ├── sample002.jpg
   ├── sample002.txt
   └── ...

**Running the example:**

.. code-block:: shell

   $ python webdataset_pipeline.py

This will create a sample tar file, build a pipeline to process it, and
demonstrate how the data flows through the pipeline.

**Key components:**

- :py:func:`create_sample_dataset`: Creates a test tar file with image/label pairs
- :py:func:`iter_tar_files`: Source that yields tar file paths
- :py:func:`load_and_extract_tar`: Loads tar file and extracts all files
- :py:func:`group_by_key`: Groups files by their base name into samples
- :py:func:`decode_sample`: Decodes images from the sample dictionary
- :py:func:`get_webdataset_pipeline`: Builds the complete pipeline
"""

# pyre-strict

import io
import logging
import tarfile
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import spdl.io
from PIL import Image
from spdl.pipeline import Pipeline, PipelineBuilder

_LG: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "create_sample_dataset",
    "iter_tar_files",
    "load_and_extract_tar",
    "group_by_key",
    "decode_sample",
    "get_webdataset_pipeline",
    "run_pipeline_example",
    "main",
]


def create_sample_dataset(output_path: Path, num_samples: int = 10) -> None:
    """Create a sample tar file with image and label pairs.

    This creates a WebDataset-style tar archive where each sample consists of
    an image file (.jpg) and a text label file (.txt) with the same base name.

    Args:
        output_path: Path where the tar file will be created.
        num_samples: Number of samples to include in the dataset.
    """
    with tarfile.open(output_path, "w") as tar:
        for i in range(num_samples):
            sample_key = f"sample{i:03d}"

            # Create a simple colored image
            img = Image.new("RGB", (64, 64), color=(i * 25 % 256, 100, 150))
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="JPEG")
            img_data = img_buffer.getvalue()

            # Add image to tar
            img_info = tarfile.TarInfo(name=f"{sample_key}.jpg")
            img_info.size = len(img_data)
            tar.addfile(img_info, io.BytesIO(img_data))

            # Create label text
            label = f"label_{i}\n"
            label_data = label.encode("utf-8")

            # Add label to tar
            label_info = tarfile.TarInfo(name=f"{sample_key}.txt")
            label_info.size = len(label_data)
            tar.addfile(label_info, io.BytesIO(label_data))

    _LG.info("Created sample dataset at %s with %d samples", output_path, num_samples)


def iter_tar_files(tar_paths: list[Path]) -> Iterator[Path]:
    """Iterate over tar file paths.

    This is a simple source that yields tar file paths. In a real scenario,
    this could be a list of many tar files stored in a distributed file system.

    Args:
        tar_paths: List of tar file paths to process.

    Yields:
        Path: Tar file paths to process.
    """
    for path in tar_paths:
        yield path


def load_and_extract_tar(tar_path: Path) -> Iterator[tuple[str, bytes]]:
    """Load a tar file and extract all files using :py:func:`spdl.io.iter_tarfile`.

    Args:
        tar_path: Path to the tar file.

    Yields:
        Tuple of ``(filename, content)`` for each file in the tar archive.
    """
    with open(tar_path, "rb") as f:
        tar_data = f.read()

    for filename, content in spdl.io.iter_tarfile(tar_data):
        # Convert memoryview to bytes for easier handling
        yield filename, bytes(content)


def group_by_key(tar_files: Iterator[tuple[str, bytes]]) -> Iterator[dict[str, Any]]:
    """Group files from tar archive by their base name (key).

    This function implements the WebDataset grouping logic where files with
    the same base name are grouped into a single sample. For example:
    - sample001.jpg and sample001.txt -> {"__key__": "sample001", "jpg": <data>, "txt": <data>}

    Args:
        tar_files: Iterator of ``(filename, content)`` tuples from tar file.

    Yields:
        Dictionary representing a sample with keys for different file types.
        Each dictionary has a ``__key__`` field with the base name and keys
        for each file extension (e.g., "jpg", "txt").
    """
    current_sample: dict[str, Any] = {}
    current_key: str | None = None

    for filename, content in tar_files:
        # Extract key (base name) and extension
        base_name = Path(filename).stem
        extension = Path(filename).suffix[1:]  # Remove the dot

        # If we're starting a new sample, yield the previous one
        if current_key is not None and base_name != current_key:
            if current_sample:
                yield current_sample
            current_sample = {}

        # Add file to current sample
        current_key = base_name
        current_sample["__key__"] = current_key
        current_sample[extension] = content

    # Yield the last sample
    if current_sample:
        yield current_sample


def decode_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Decode image and process label from a sample.

    Args:
        sample: Dictionary with "__key__", "jpg", and "txt" keys.

    Returns:
        Dictionary with decoded image as PIL Image and decoded label.
    """
    result = {"key": sample["__key__"]}

    # Decode image if present
    if "jpg" in sample:
        img_bytes = sample["jpg"]
        img = Image.open(io.BytesIO(img_bytes))
        result["image"] = img
        result["image_size"] = img.size

    # Decode label if present
    if "txt" in sample:
        label_bytes = sample["txt"]
        label = label_bytes.decode("utf-8").strip()
        result["label"] = label

    return result


def get_webdataset_pipeline(
    tar_paths: list[Path],
    batch_size: int = 4,
    buffer_size: int = 8,
    num_threads: int = 4,
) -> Pipeline:
    """Build a WebDataset-like pipeline for processing tar files.

    The pipeline structure:
    1. Source: Iterate over tar file paths
    2. Load tar files and extract contents using :py:func:`spdl.io.iter_tarfile`
    3. Group files by key (WebDataset-style)
    4. Decode images and labels
    5. Batch samples
    6. Sink: Buffer for consumer

    Args:
        tar_paths: List of tar file paths to process.
        batch_size: Number of samples per batch.
        buffer_size: Size of output buffer.
        num_threads: Number of threads for parallel processing.

    Returns:
        Pipeline configured for WebDataset-style processing.
    """
    pipeline = (
        PipelineBuilder()
        .add_source(iter_tar_files(tar_paths))
        # Load and extract tar files in parallel
        .pipe(load_and_extract_tar, concurrency=2)
        # Flatten the iterator of iterators
        .disaggregate()
        # Group files by their base name
        .pipe(group_by_key)
        # Flatten again after grouping
        .disaggregate()
        # Decode images and labels in parallel
        .pipe(decode_sample, concurrency=num_threads)
        # Batch samples
        .aggregate(batch_size)
        .add_sink(buffer_size)
        .build(num_threads=num_threads)
    )
    return pipeline


def run_pipeline_example() -> None:
    """Run the WebDataset pipeline example.

    Creates a sample dataset, builds the pipeline, and processes all samples.
    """
    # Create a temporary tar file with sample data
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / "dataset.tar"
        create_sample_dataset(tar_path, num_samples=10)

        _LG.info("Building pipeline...")
        pipeline = get_webdataset_pipeline(
            tar_paths=[tar_path], batch_size=3, num_threads=2
        )

        _LG.info("Processing data through pipeline...")
        num_batches = 0
        num_samples = 0

        with pipeline.auto_stop():
            for batch in pipeline:
                num_batches += 1
                batch_size = len(batch)
                num_samples += batch_size

                _LG.info("Batch %d: %d samples", num_batches, batch_size)

                # Display info about first sample in batch
                if batch:
                    sample = batch[0]
                    _LG.info(
                        "  Sample: key=%s, label=%s, image_size=%s",
                        sample["key"],
                        sample.get("label", "N/A"),
                        sample.get("image_size", "N/A"),
                    )

        _LG.info("Processed %d samples in %d batches", num_samples, num_batches)


def main() -> None:
    """Main entry point for the WebDataset pipeline example."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    _LG.info("=" * 60)
    _LG.info("WebDataset-like Pipeline Example using SPDL")
    _LG.info("=" * 60)

    run_pipeline_example()

    _LG.info("=" * 60)
    _LG.info("Example completed successfully!")
    _LG.info("=" * 60)


if __name__ == "__main__":
    main()
