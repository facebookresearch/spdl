"""Test decoding with image classification"""

import concurrent.futures
import contextlib
import logging
import os
import time
from pathlib import Path

os.environ["TORCH_LOGS"] = "recompiles,graph_breaks"

import spdl.io
import spdl.utils

import timm
import torch
from spdl.dataloader._task_runner import (
    apply_async,
    apply_concurrent,
    BackgroundTaskProcessor,
)
from spdl.dataloader._utils import _iter_flist
from spdl.dataset.imagenet import get_mappings, parse_wnid
from torch.profiler import profile

_LG = logging.getLogger(__name__)


def _parse_args(args):
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--input-flist", type=Path, required=True)
    parser.add_argument("--mode", choices=["async", "concurrent"], default="async")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--queue-size", type=int, default=16)
    parser.add_argument("--num-demux-threads", type=int, default=8)
    parser.add_argument("--num-decode-threads", type=int, default=16)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--no-compile", action="store_false", dest="compile")
    parser.add_argument("--no-bf16", action="store_false", dest="use_bf16")
    args = parser.parse_args(args)
    if args.trace:
        args.max_samples = args.batch_size * 40
    return args


def _expand(vals, batch_size, res):
    return torch.tensor(vals).view(1, 3, 1, 1).expand(batch_size, 3, res, res).clone()


# Handroll the transforms so as to support `torch.compile`
class Transform(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        x = x.float() / 255.0
        return (x - self.mean) / self.std


class Classification(torch.nn.Module):
    def forward(self, x, classes):
        probs = torch.nn.functional.softmax(x, dim=-1)
        top_prob, top_catid = torch.topk(probs, 5)
        top1 = (top_catid[:, :1] == classes).sum()
        top5 = (top_catid == classes).sum()
        return top1, top5


class ModelBundle(torch.nn.Module):
    def __init__(self, model, transform, classification, use_bf16):
        super().__init__()
        self.model = model
        self.transform = transform
        self.classification = classification
        self.use_bf16 = use_bf16

    def forward(self, x, classes):
        x = x.permute((0, 3, 1, 2))
        x = self.transform(x)

        if self.use_bf16:
            x = x.to(torch.bfloat16)

        output = self.model(x)

        return self.classification(output, classes)


def _get_model(batch_size, device, compile, use_bf16):
    model = timm.create_model("mobilenetv3_large_100", pretrained=True)
    model = model.eval().to(device=device)
    if use_bf16:
        model = model.to(dtype=torch.bfloat16)

    transform = Transform(
        mean=_expand([0.4850, 0.4560, 0.4060], batch_size, 224),
        std=_expand([0.2290, 0.2240, 0.2250], batch_size, 224),
    ).to(device)

    classification = Classification().to(device)

    if compile:
        with torch.no_grad():
            mode = "max-autotune"
            model = torch.compile(model, mode=mode)
            transform = torch.compile(transform, mode=mode)

    return ModelBundle(model, transform, classification, use_bf16)


def _run_inference(dataloader, model, device):
    _LG.info("Running inference.")
    t0 = time.monotonic()
    num_frames, num_correct_top1, num_correct_top5 = 0, 0, 0
    try:
        for i, (buffer, classes) in enumerate(dataloader):
            # Ignore the first batch.
            if i == 2:
                t0 = time.monotonic()
                num_frames, num_correct_top1, num_correct_top5 = 0, 0, 0

            with torch.profiler.record_function(f"iter_{i}"):
                batch = spdl.io.to_torch(buffer)
                classes = torch.tensor(classes, dtype=torch.int64).to(device)

                top1, top5 = model(batch, classes)

                num_frames += batch.shape[0]
                num_correct_top1 += top1
                num_correct_top5 += top5
    finally:
        elapsed = time.monotonic() - t0
        num_correct_top1 = num_correct_top1.item()
        num_correct_top5 = num_correct_top5.item()
        fps = num_frames / elapsed
        _LG.info(f"FPS={fps:.2f} ({num_frames}/{elapsed:.2f})")
        acc1 = 0 if num_frames == 0 else num_correct_top1 / num_frames
        _LG.info(f"Accuracy (top1)={acc1:.2%} ({num_correct_top1}/{num_frames})")
        acc5 = 0 if num_frames == 0 else num_correct_top5 / num_frames
        _LG.info(f"Accuracy (top5)={acc5:.2%} ({num_correct_top5}/{num_frames})")


def _get_batch_generator(args):
    srcs_gen = _iter_flist(
        args.input_flist,
        prefix=args.prefix,
        batch_size=args.batch_size,
        n=args.worker_id,
        N=args.num_workers,
        max=args.max_samples,
        drop_last=True,
    )

    class_mapping, _ = get_mappings()

    async def _async_decode_func(paths):
        with torch.profiler.record_function("async_decode"):
            classes = [[class_mapping[parse_wnid(p)]] for p in paths]
            buffer = await spdl.io.async_batch_load_image(
                paths,
                width=224,
                height=224,
                pix_fmt="rgb24",
                strict=False,
                convert_options={"cuda_device_index": 0},
            )
            return buffer, classes

    @spdl.utils.chain_futures
    def _decode_func(paths):
        with torch.profiler.record_function("decode"):
            classes = [[class_mapping[parse_wnid(p)]] for p in paths]
            buffer = yield spdl.io.batch_load_image(
                paths,
                width=224,
                height=224,
                pix_fmt="rgb24",
                strict=False,
                convert_options={"cuda_device_index": 0},
            )
            f = concurrent.futures.Future()
            f.set_result((buffer, classes))
            yield f

    match args.mode:
        case "concurrent":
            return apply_concurrent(_decode_func, srcs_gen)
        case "async":
            return apply_async(_async_decode_func, srcs_gen)


def _main(args=None):
    args = _parse_args(args)
    _init(args.debug, args.num_demux_threads, args.num_decode_threads, args.worker_id)
    _LG.info(args)

    batch_gen = _get_batch_generator(args)

    device = torch.device("cuda:0")
    model = _get_model(args.batch_size, device, args.compile, args.use_bf16)

    with (
        torch.no_grad(),
        profile() if args.trace else contextlib.nullcontext() as prof,
        BackgroundTaskProcessor(batch_gen, args.queue_size) as dataloader,
    ):
        _run_inference(dataloader, model, device)

    if args.trace:
        trace_path = f"{args.trace}.{args.worker_id}"
        prof.export_chrome_trace(f"{trace_path}.json")


def _init(debug, num_demux_threads, num_decode_threads, worker_id):
    _init_logging(debug, worker_id)

    spdl.utils.set_ffmpeg_log_level(16)
    spdl.utils.init_folly(
        [
            f"--spdl_demuxer_executor_threads={num_demux_threads}",
            f"--spdl_decoder_executor_threads={num_decode_threads}",
            f"--logging={'DBG' if debug else 'INFO'}",
        ]
    )


def _init_logging(debug=False, worker_id=None):
    fmt = "%(asctime)s [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s"
    if worker_id is not None:
        fmt = f"[{worker_id}:%(thread)d] {fmt}"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level)


if __name__ == "__main__":
    _main()
