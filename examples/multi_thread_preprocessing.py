#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""This example shows how to run PyTorch tarnsform in SPDL Pipeline,
and compares its performance against PyTorch DataLoader.

Each pipeline reads images from the ImageNet dataset, and applies
resize, batching, and pixel normalization then the data is transferred
to GPU.

In the PyTorch and TorchVision native solution, the images are decoded
and resized using Pillow, batched with :py:func:`torch.utils.data.default_collate`,
pixel normalization is applied with :py:class:`torchvision.transforms.Normalize`,
and data are transferred to GPU with :py:func:`torch.Tensor.cuda`.

Using :py:class:`torch.utils.data.DataLoader`, the batch is created and
normalized in subprocess and transferred to the main process before they are
sent to GPU.

The following diagram illustrates this.

.. include:: ../plots/multi_thread_preprocessing_chart_torch.txt

On the other hand, SPDL Pipeline executes the transforms in the main process.
SPDL pipeline uses its own implementation for decode, resize and batching image data.

.. include:: ../plots/multi_thread_preprocessing_chart_spdl.txt

This script runs the pipeline with different configurations described bellow while
changing the number of workers.

1. Image decoding and resizing
2. Image decoding, resizing, and batching
3. Image decoding, resizing, batching, and normalization
4. Image decoding, resizing, batching, normalization, and transfer to GPU

The following result was obtained.

.. include:: ../plots/multi_thread_preprocessing_plot.txt

The following observations can be made.

- In both implementations, the throughput peaks around 16 workers,
  and then decreases as the number of workers.
- The throughput increases when batching images, then decreases
  as additional processing is added.
- The degree of improvement from batching in SPDL is significantly
  higher than in PyTorch. (more than 2x at 16 workers.)
- The peak througput is almost 2.7x in SPDL than in PyTorch.
"""

import logging
import time
from collections.abc import Iterable
from typing import Any

import spdl.io
import torch
from spdl.dataloader import PipelineBuilder

from torchvision.datasets import ImageNet
from torchvision.transforms import Compose, Normalize, PILToTensor, Resize

__all__ = [
    "entrypoint",
    "exp_torch_native",
    "exp_spdl",
    "run_dataloader",
]


logging.getLogger().setLevel(logging.ERROR)

PREFETCH_FACTOR = 10_000
# so as to prevent the pipeline from being blocked on the foreground


def run_dataloader(
    dataloader: Iterable,
    max_items: int = 3200,
) -> tuple[int, float]:
    """Run the given dataloader and measure its performance.

    Args:
        dataloader: The dataloader to benchmark.
        max_items: The maximum number of items to process.

    Returns:
        The number of items processed and the elapsed time in seconds.
    """
    num_images = 0
    t0 = time.monotonic()
    try:
        for data, _ in dataloader:
            num_images += 1 if data.ndim == 3 else len(data)
            if num_images >= max_items:
                break
    finally:
        elapsed = time.monotonic() - t0
    return num_images, elapsed


def get_dataset(
    root: str,
    *,
    disable_loader: bool = False,
) -> ImageNet:
    if disable_loader:
        return ImageNet(root=root, loader=lambda x: x)

    transforms: list[Any] = [Resize((224, 224)), PILToTensor()]

    return ImageNet(root=root, transform=Compose(transforms))


def exp_torch_native(
    root_dir: str,
    num_workers: int,
    batch_size: int | None = None,
    normalize: bool = False,
    transfer: bool = False,
) -> tuple[int, float]:
    """Load data with PyTorch native operation using PyTorch DataLoader.

    This is the baseline for comparison.

    Args:
        root_dir: The root directory of the ImageNet dataset.
        num_workers: The number of workers to use.
        batch: Whether to batch the data.
        normalize: Whether to normalize the data. Only applicable when ``batch`` is True.
        transfer: Whether to transfer the data to GPU.

    Returns:
        The number of items processed and the elapsed time in seconds.
    """
    dataset = get_dataset(root_dir)

    normalize_transform = Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    def collate(item):
        batch, cls = torch.utils.data.default_collate(item)
        if normalize:
            batch = batch.float() / 255
            batch = normalize_transform(batch)
        return batch, cls

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=None if batch_size is None else collate,
        prefetch_factor=PREFETCH_FACTOR,
    )

    if transfer:

        def with_transfer(dataloader):
            for tensor, cls in dataloader:
                tensor = tensor.cuda()
                yield tensor, cls

        dataloader = with_transfer(dataloader)

    with torch.no_grad():
        return run_dataloader(dataloader)


def exp_spdl(
    root_dir: str,
    num_workers: int,
    batch_size: int | None = None,
    normalize: bool = False,
    transfer: bool = False,
) -> tuple[int, float]:
    """Load data with SPDL operation using SPDL Pipeline.

    Args:
        root_dir: The root directory of the ImageNet dataset.
        num_workers: The number of workers to use.
        batch: Whether to batch the data.
        normalize: Whether to normalize the data. Only applicable when ``batch`` is True.
        transfer: Whether to transfer the data to GPU.

    Returns:
        The number of items processed and the elapsed time in seconds.

    """
    dataset = get_dataset(root_dir, disable_loader=True)

    filter_desc = spdl.io.get_video_filter_desc(
        scale_width=224,
        scale_height=224,
    )

    def decode_image(item):
        path, cls = item
        packets = spdl.io.demux_image(path)
        frames = spdl.io.decode_packets(packets, filter_desc=filter_desc)
        return frames, cls

    def convert(items):
        frames, cls = list(zip(*items))
        buffer = spdl.io.convert_frames(frames)
        tensor = spdl.io.to_torch(buffer).permute(0, 3, 1, 2)
        return tensor, cls

    builder = (
        PipelineBuilder()
        .add_source(iter(dataset))
        .pipe(decode_image, concurrency=num_workers)
        .aggregate(batch_size or 1)
        .pipe(convert)
    )

    if normalize:
        transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        def normalize(item):
            tensor, cls = item
            tensor = tensor.float() / 255
            tensor = transform(tensor)
            return tensor, cls

        builder = builder.pipe(normalize)

    if transfer:
        builder = builder.pipe(lambda item: (item[0].cuda(), item[1]))

    builder = builder.add_sink(PREFETCH_FACTOR)
    pipeline = builder.build(num_threads=num_workers)

    with torch.no_grad(), pipeline.auto_stop():
        return run_dataloader(pipeline)


def run_test(root_dir, max_items, **kwargs):
    data = {}
    num_workers_ = [1, 2, 4, 8, 16, 32]
    for func in [exp_torch_native, exp_spdl]:  # exp_torch_thread, exp_spdl]:
        print(func.__name__)
        print("\tnum_workers\tFPS")
        y = []
        for num_workers in num_workers_:
            num_images, elapsed = func(root_dir, num_workers, **kwargs)
            qps = num_images / elapsed
            y.append(qps)
            print(f"\t{num_workers}\t{qps:8.2f} ({num_images} / {elapsed:5.2f})")

        data[func.__name__] = (num_workers_, y)

    return data


def _print(data, kwargs):
    for i, (x, y) in enumerate(data.values()):
        if i == 0:
            print("\t".join(str(v) for v in x))
        print("\t".join(f"{v:.2f}" for v in y))


def entrypoint(
    root_dir: str,
    batch_size: int,
    max_items: int = 3200,
):
    """The main entrypoint for CLI.

    Args:
        root_dir: The root directory of the ImageNet dataset.
        batch_size: The batch size to use.
        max_items: The maximum number of items to process.
    """
    argset = (
        {"batch_size": None},
        {"batch_size": batch_size},
        {"batch_size": batch_size, "normalize": True},
        {"batch_size": batch_size, "normalize": True, "transfer": True},
    )

    for kwargs in argset:
        print(kwargs)
        data = run_test(root_dir=root_dir, max_items=max_items, **kwargs)
        _print(data, kwargs)


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        help="Directory where the ImageNet dataset is stored.",
        default="/home/moto/local/imagenet/",
    )
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument(
        "--max-items",
        default=3200,
        type=int,
        help="The maximum number of items (images) to process.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    entrypoint(
        _args.root_dir,
        _args.batch_size,
        _args.max_items,
    )
