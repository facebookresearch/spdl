"""Referench script to benchmark the performance of PyTorch DataLoader"""

import contextlib
import logging
import time
from pathlib import Path

import torch
from torch import Tensor
from torch.profiler import profile
from torchvision.datasets.imagenet import ImageNet

from torchvision.transforms import Compose, PILToTensor, Resize


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
        "--num-workers",
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
    args = parser.parse_args(args)
    if args.trace and args.max_batches is None:
        args.max_batches = 300
    return args


def _main(args=None):
    args = _parse_args(args)
    _init_logging(args.debug)

    dataloader = torch.utils.data.DataLoader(
        ImageNet(args.root, transform=Compose([Resize((224, 224)), PILToTensor()])),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        timeout=5,
        pin_memory=True,
    )

    dataloader = iter(dataloader)
    for _ in range(50):
        next(dataloader)

    with (profile() if args.trace is not None else contextlib.nullcontext() as prof,):
        t0 = time.monotonic()
        num_frames = 0
        try:
            for i, (batch, classes) in enumerate(dataloader):
                batch = batch.to("cuda")
                classes = classes.to("cuda")
                # print(batch.shape, batch.dtype, classes.shape, classes.dtype)
                num_frames += batch.shape[0]

                if args.max_batches is not None and i == args.max_batches - 1:
                    break

        finally:
            elapsed = time.monotonic() - t0
            QPS = num_frames / elapsed
            print(f"{QPS=:.2f}: {elapsed:.2f} [sec], {num_frames=}")

    if args.trace:
        prof.export_chrome_trace(f"{args.trace}.native.json")


def _init_logging(debug):
    fmt = "%(asctime)s [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level)


if __name__ == "__main__":
    _main()
