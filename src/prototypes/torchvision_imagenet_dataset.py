"""Test the performance of SPDL and PyTorch DataLoader integration"""

import contextlib
import logging
import time
from pathlib import Path

import spdl.io
import spdl.utils

import torch
from pytorch_spdl.dataloader import DataLoader
from spdl.io import ImageFrames
from torch import Tensor
from torch.profiler import profile

from torchvision.datasets.imagenet import ImageNet


def _parse_args(args):
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help=(
            "Defines how many decoding jobs to run concurrently. "
            "There will be at most `prefetch_factor` * `num_threads` jobs."
        ),
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="The directory where ImageNet dataset is stored.",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        help="If provided, trace the execution.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
    )
    parser.add_argument(
        "--use-nvjpeg",
        action="store_true",
    )
    args = parser.parse_args(args)
    if args.trace and args.max_batches is None:
        args.max_batches = 300
    return args


def _decode(path: str, width=224, height=224, pix_fmt="rgb24") -> ImageFrames:
    packets = spdl.io.demux_image(path)
    return spdl.io.decode_packets(
        packets,
        filter_desc=spdl.io.get_video_filter_desc(
            scale_width=224, scale_height=224, pix_fmt="rgb24"
        ),
    )


def _batch_decode(samples: list[tuple[str, int]]) -> tuple[Tensor, Tensor]:
    classes = torch.tensor([cls_ for _, cls_ in samples])

    frames = [_decode(path) for path, _ in samples]
    buffer = spdl.io.convert_frames(frames)
    buffer = spdl.io.transfer_buffer(
        buffer,
        cuda_config=spdl.io.cuda_config(
            device_index=0,
            allocator=(
                torch.cuda.caching_allocator_alloc,
                torch.cuda.caching_allocator_delete,
            ),
        ),
    )
    return spdl.io.to_torch(buffer), classes


def _batch_decode_nvjpeg(samples: list[tuple[str, int]]) -> tuple[Tensor, Tensor]:
    classes = torch.tensor([cls_ for _, cls_ in samples])

    buffer = spdl.io.load_image_batch_nvjpeg(
        [path for path, _ in samples],
        cuda_config=spdl.io.cuda_config(
            device_index=0,
            allocator=(
                torch.cuda.caching_allocator_alloc,
                torch.cuda.caching_allocator_delete,
            ),
        ),
        width=224,
        height=224,
        pix_fmt="rgb",
        # strict=True,
    )
    batch = spdl.io.to_torch(buffer)
    return batch, classes


def _main(args=None):
    args = _parse_args(args)
    _init_logging(args.debug)

    dataloader = DataLoader(
        ImageNet(root=args.root, loader=lambda x: x),
        collate_fn=_batch_decode_nvjpeg if args.use_nvjpeg else _batch_decode,
        batch_size=args.batch_size,
        num_workers=args.num_threads,
        prefetch_factor=args.prefetch_factor,
    )

    dataloader = iter(dataloader)
    for _ in range(50):
        next(dataloader)

    with (
        profile() if args.trace is not None else contextlib.nullcontext() as prof,
        spdl.utils.tracing(
            f"{args.trace}.pftrace", buffer_size=4096 * 16, enable=args.trace
        ),
    ):
        t0 = time.monotonic()
        num_frames = 0
        try:
            for i, (batch, classes) in enumerate(dataloader):
                # print(batch.shape, batch.dtype, classes.shape, classes.dtype)
                num_frames += batch.shape[0]

                if args.max_batches is not None and i == args.max_batches - 1:
                    break

        finally:
            elapsed = time.monotonic() - t0
            QPS = num_frames / elapsed
            print(f"{QPS=:.2f}: {elapsed:.2f} [sec], {num_frames=}")

    if args.trace:
        prof.export_chrome_trace(f"{args.trace}.json")


def _init_logging(debug):
    fmt = "%(asctime)s [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level)


if __name__ == "__main__":
    _main()
