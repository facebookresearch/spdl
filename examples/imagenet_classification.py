# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark the performance of loading images from local file systems and
classifying them using a GPU.

This script builds the data loader and instantiates an image
classification model in a GPU.
The data loader transfers the batch image data to the GPU concurrently, and
the foreground thread run the model on data one by one.

.. include:: ../plots/imagenet_classification_chart.txt

To run the benchmark,  pass it to the script like the following.

.. code-block::

   python imagenet_classification.py
       --root-dir ~/imagenet/
       --split val
"""

# pyre-ignore-all-errors

import contextlib
import logging
import time
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import spdl.io
import spdl.utils
import torch
from spdl.dataloader import DataLoader
from spdl.source.imagenet import ImageNet
from torch import Tensor
from torch.profiler import profile

_LG = logging.getLogger(__name__)


__all__ = [
    "entrypoint",
    "benchmark",
    "get_decode_func",
    "get_dataloader",
    "get_model",
    "ModelBundle",
    "Classification",
    "Preprocessing",
]


def _parse_args(args):
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=float("inf"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--buffer-size", type=int, default=16)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--no-compile", action="store_false", dest="compile")
    parser.add_argument("--no-bf16", action="store_false", dest="use_bf16")
    parser.add_argument("--use-nvdec", action="store_true")
    parser.add_argument("--use-nvjpeg", action="store_true")
    args = parser.parse_args(args)
    if args.trace:
        args.max_batches = 60
    return args


# Handroll the transforms so as to support `torch.compile`
class Preprocessing(torch.nn.Module):
    """Perform pixel normalization and data type conversion.

    Args:
        mean: The mean value of the dataset.
        std: The standard deviation of the dataset.
    """

    def __init__(self, mean: Tensor, std: Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: Tensor) -> Tensor:
        """Normalize the given image batch.

        Args:
            x: The input image batch. Pixel values are expected to be
                in the range of ``[0, 255]``.
        Returns:
            The normalized image batch.
        """
        x = x.float() / 255.0
        return (x - self.mean) / self.std


class Classification(torch.nn.Module):
    """Classification()"""

    def forward(self, x: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """Given a batch of features and labels, compute the top1 and top5 accuracy.

        Args:
            images: A batch of images. The shape is ``(batch_size, 3, 224, 224)``.
            labels: A batch of labels. The shape is ``(batch_size,)``.

        Returns:
            A tuple of top1 and top5 accuracy.
        """

        probs = torch.nn.functional.softmax(x, dim=-1)
        top_prob, top_catid = torch.topk(probs, 5)
        top1 = (top_catid[:, :1] == labels).sum()
        top5 = (top_catid == labels).sum()
        return top1, top5


class ModelBundle(torch.nn.Module):
    """ModelBundle()

    Bundle the transform, model backbone, and classification head into a single module
    for a simple handling."""

    def __init__(self, model, preprocessing, classification, use_bf16):
        super().__init__()
        self.model = model
        self.preprocessing = preprocessing
        self.classification = classification
        self.use_bf16 = use_bf16

    def forward(self, images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """Given a batch of images and labels, compute the top1, top5 accuracy.

        Args:
            images: A batch of images. The shape is ``(batch_size, 3, 224, 224)``.
            labels: A batch of labels. The shape is ``(batch_size,)``.

        Returns:
            A tuple of top1 and top5 accuracy.
        """

        x = self.preprocessing(images)

        if self.use_bf16:
            x = x.to(torch.bfloat16)

        output = self.model(x)

        return self.classification(output, labels)


def _expand(vals, batch_size, res):
    return torch.tensor(vals).view(1, 3, 1, 1).expand(batch_size, 3, res, res).clone()


def get_model(
    batch_size: int,
    device_index: int,
    compile: bool,
    use_bf16: bool,
    model_type: str = "mobilenetv3_large_100",
) -> ModelBundle:
    """Build computation model, including transfor, model, and classification head.

    Args:
        batch_size: The batch size of the input.
        device_index: The index of the target GPU device.
        compile: Whether to compile the model.
        use_bf16: Whether to use bfloat16 for the model.
        model_type: The type of the model. Passed to ``timm.create_model()``.

    Returns:
        The resulting computation model.
    """
    import timm

    device = torch.device(f"cuda:{device_index}")

    model = timm.create_model(model_type, pretrained=True)
    model = model.eval().to(device=device)

    if use_bf16:
        model = model.to(dtype=torch.bfloat16)

    preprocessing = Preprocessing(
        mean=_expand([0.4850, 0.4560, 0.4060], batch_size, 224),
        std=_expand([0.2290, 0.2240, 0.2250], batch_size, 224),
    ).to(device)

    classification = Classification().to(device)

    if compile:
        with torch.no_grad():
            mode = "max-autotune"
            model = torch.compile(model, mode=mode)
            preprocessing = torch.compile(preprocessing, mode=mode)

    return ModelBundle(model, preprocessing, classification, use_bf16)


def get_decode_func(
    device_index: int,
    width: int = 224,
    height: int = 224,
) -> Callable[[list[tuple[str, int]]], Awaitable[tuple[Tensor, Tensor]]]:
    """Get a function to decode images from a list of paths.

    Args:
        device_index: The index of the target GPU device.
        width: The width of the decoded image.
        height: The height of the decoded image.

    Returns:
        Async function to decode images in to batch tensor of NCHW format
        and labels of shape ``(batch_size, 1)``.
    """
    device = torch.device(f"cuda:{device_index}")

    filter_desc = spdl.io.get_video_filter_desc(
        scale_width=256,
        scale_height=256,
        crop_width=width,
        crop_height=height,
        pix_fmt="rgb24",
    )

    async def decode_images(items: list[tuple[str, int]]):
        paths = [item for item, _ in items]
        labels = [[item] for _, item in items]
        labels = torch.tensor(labels, dtype=torch.int64).to(device)
        buffer = await spdl.io.async_load_image_batch(
            paths,
            width=None,
            height=None,
            pix_fmt=None,
            strict=True,
            filter_desc=filter_desc,
            device_config=spdl.io.cuda_config(
                device_index=0,
                allocator=(
                    torch.cuda.caching_allocator_alloc,
                    torch.cuda.caching_allocator_delete,
                ),
            ),
        )
        batch = spdl.io.to_torch(buffer)
        batch = batch.permute((0, 3, 1, 2))
        return batch, labels

    return decode_images


def _get_experimental_nvjpeg_decode_function(
    device_index: int,
    width: int = 224,
    height: int = 224,
):
    device = torch.device(f"cuda:{device_index}")
    device_config = spdl.io.cuda_config(
        device_index=device_index,
        allocator=(
            torch.cuda.caching_allocator_alloc,
            torch.cuda.caching_allocator_delete,
        ),
    )

    async def decode_images_nvjpeg(items: list[tuple[str, int]]):
        paths = [item for item, _ in items]
        labels = [[item] for _, item in items]
        labels = torch.tensor(labels, dtype=torch.int64).to(device)
        buffer = await spdl.io.async_load_image_batch_nvjpeg(
            paths,
            device_config=device_config,
            width=width,
            height=height,
            pix_fmt="rgb",
            # strict=True,
        )
        batch = spdl.io.to_torch(buffer)
        return batch, labels

    return decode_images_nvjpeg


def _get_experimental_nvdec_decode_function(
    device_index: int,
    width: int = 224,
    height: int = 224,
):
    device = torch.device(f"cuda:{device_index}")
    device_config = spdl.io.cuda_config(
        device_index=device_index,
        allocator=(
            torch.cuda.caching_allocator_alloc,
            torch.cuda.caching_allocator_delete,
        ),
    )

    async def decode_images_nvdec(items: list[tuple[str, int]]):
        paths = [item for item, _ in items]
        labels = [[item] for _, item in items]
        labels = torch.tensor(labels, dtype=torch.int64).to(device)
        buffer = await spdl.io.async_load_image_batch_nvdec(
            paths,
            device_config=device_config,
            width=width,
            height=height,
            pix_fmt="rgba",
            strict=True,
        )
        batch = spdl.io.to_torch(buffer)[:, :-1, :, :]
        return batch, labels

    return decode_images_nvdec


def get_dataloader(
    src: Iterator[tuple[str, int]],
    batch_size: int,
    decode_func: Callable[[list[tuple[str, int]]], Awaitable[tuple[Tensor, Tensor]]],
    buffer_size: int,
    num_threads: int,
) -> DataLoader:
    """Build the dataloader for the ImageNet classification task.

    The dataloader uses the ``decode_func`` for decoding images concurrently and
    send the resulting data to GPU.

    Args:
        src: The source of the data. See :py:func:`source`.
        batch_size: The number of images in a batch.
        decode_func: The function to decode images.
        buffer_size: The size of the buffer for the dataloader sink
        num_threads: The number of worker threads.

    """
    return DataLoader(
        src,
        batch_size=batch_size,
        drop_last=True,
        aggregator=decode_func,
        buffer_size=buffer_size,
        num_threads=num_threads,
        timeout=20,
    )


def benchmark(
    dataloader: Iterator[tuple[Tensor, Tensor]],
    model: ModelBundle,
    max_batches: int = float("nan"),
) -> None:
    """The main loop that measures the performance of dataloading and model inference.

    Args:
        loader: The dataloader to benchmark.
        model: The model to benchmark.
        max_batches: The number of batch before stopping.
    """

    _LG.info("Running inference.")
    num_frames, num_correct_top1, num_correct_top5 = 0, 0, 0
    t0 = time.monotonic()
    try:
        for i, (batch, labels) in enumerate(dataloader):
            if i == 20:
                t0 = time.monotonic()
                num_frames, num_correct_top1, num_correct_top5 = 0, 0, 0

            with (
                torch.profiler.record_function(f"iter_{i}"),
                spdl.utils.trace_event(f"iter_{i}"),
            ):
                top1, top5 = model(batch, labels)

                num_frames += batch.shape[0]
                num_correct_top1 += top1
                num_correct_top5 += top5

            if i + 1 >= max_batches:
                break
    finally:
        elapsed = time.monotonic() - t0
        if num_frames != 0:
            num_correct_top1 = num_correct_top1.item()
            num_correct_top5 = num_correct_top5.item()
            fps = num_frames / elapsed
            _LG.info(f"FPS={fps:.2f} ({num_frames}/{elapsed:.2f})")
            acc1 = 0 if num_frames == 0 else num_correct_top1 / num_frames
            _LG.info(f"Accuracy (top1)={acc1:.2%} ({num_correct_top1}/{num_frames})")
            acc5 = 0 if num_frames == 0 else num_correct_top5 / num_frames
            _LG.info(f"Accuracy (top5)={acc5:.2%} ({num_correct_top5}/{num_frames})")


def _get_dataloader(args, device_index) -> DataLoader:
    src = ImageNet(args.root_dir, split=args.split)

    if args.use_nvjpeg:
        decode_func = _get_experimental_nvjpeg_decode_function(device_index)
    elif args.use_nvdec:
        decode_func = _get_experimental_nvdec_decode_function(device_index)
    else:
        decode_func = get_decode_func(device_index)

    return get_dataloader(
        src,
        args.batch_size,
        decode_func,
        args.buffer_size,
        args.num_threads,
    )


def entrypoint(args: list[int] | None = None):
    """CLI entrypoint. Run pipeline, transform and model and measure its performance."""

    args = _parse_args(args)
    _init_logging(args.debug)
    _LG.info(args)

    device_index = 0
    model = get_model(args.batch_size, device_index, args.compile, args.use_bf16)
    dataloader = _get_dataloader(args, device_index)

    trace_path = f"{args.trace}"
    if args.use_nvjpeg:
        trace_path = f"{trace_path}.nvjpeg"
    if args.use_nvdec:
        trace_path = f"{trace_path}.nvdec"

    with (
        torch.no_grad(),
        profile() if args.trace else contextlib.nullcontext() as prof,
        spdl.utils.tracing(f"{trace_path}.pftrace", enable=args.trace is not None),
    ):
        benchmark(dataloader, model, args.max_batches)

    if args.trace:
        prof.export_chrome_trace(f"{trace_path}.json")


def _init_logging(debug=False):
    fmt = "%(asctime)s [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level)


if __name__ == "__main__":
    entrypoint()
