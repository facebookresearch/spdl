"""Test decoding with image classification"""

import concurrent.futures
import logging
import time
from pathlib import Path

import spdl.io
import spdl.utils

import timm
import torch
import torchvision.transforms
from spdl.dataloader._task_runner import (
    apply_async,
    apply_concurrent,
    BackgroundTaskProcessor,
)
from spdl.dataloader._utils import _iter_flist
from spdl.dataset.imagenet import get_mappings, parse_wnid

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
    args = parser.parse_args(args)
    if args.trace:
        args.max = args.batch_size * 10
    return args


def _get_model(device):
    model = timm.create_model("mobilenetv3_large_100", pretrained=True)
    model = model.to(device)
    model.eval()

    config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.transforms_factory.create_transform(**config)

    class to_float(torch.nn.Module):
        def forward(self, x):
            return x.float() / 255.0

    transform = torch.nn.Sequential(
        torchvision.transforms.CenterCrop([224]),
        to_float(),
        torchvision.transforms.Normalize(
            mean=[0.4850, 0.4560, 0.4060],
            std=[0.2290, 0.2240, 0.2250],
        ),
    )
    transform = transform.to(device)
    return model, transform


def _run_inference(dataloader, device, transform, model):
    t0 = time.monotonic()
    num_frames, num_correct_top1, num_correct_top5 = 0, 0, 0
    try:
        for buffer, classes in dataloader:
            batch = spdl.io.to_torch(buffer)
            classes = torch.tensor(classes, dtype=torch.int64)
            num_frames += batch.shape[0]

            batch = batch.permute((0, 3, 1, 2)).to(device)
            batch = transform(batch)

            output = model(batch)
            probs = torch.nn.functional.softmax(output, dim=-1)
            top_prob, top_catid = torch.topk(probs.cpu(), 5)
            num_correct_top1 += (top_catid[:, :1] == classes).sum().item()
            num_correct_top5 += (top_catid == classes).sum().item()
    finally:
        elapsed = time.monotonic() - t0
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
    )

    class_mapping, _ = get_mappings()

    async def _async_decode_func(paths):
        classes = [[class_mapping[parse_wnid(p)]] for p in paths]
        buffer = await spdl.io.async_batch_load_image(
            paths,
            width=256,
            height=256,
            pix_fmt="rgb24",
            strict=False,
        )
        return buffer, classes

    @spdl.utils.chain_futures
    def _decode_func(paths):
        classes = [[class_mapping[parse_wnid(p)]] for p in paths]
        buffer = yield spdl.io.batch_load_image(
            paths,
            width=256,
            height=256,
            pix_fmt="rgb24",
            strict=False,
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

    dataloader = BackgroundTaskProcessor(batch_gen, args.queue_size)
    device = torch.device("cuda:0")
    model, transform = _get_model(device)
    print(transform)

    # Warm up
    torch.zeros([1, 1], device=device)

    trace_path = f"{args.trace}.{args.worker_id}"
    with spdl.utils.tracing(trace_path, enable=args.trace is not None):
        try:
            dataloader.start()
            with torch.inference_mode():
                _run_inference(dataloader, device, transform, model)
        except (KeyboardInterrupt, Exception):
            _LG.exception("Exception occured while running the inference.")
            dataloader.request_stop()
        finally:
            dataloader.join()


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
