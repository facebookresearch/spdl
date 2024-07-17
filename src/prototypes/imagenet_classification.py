"""Test decoding with image classification"""

# pyre-ignore-all-errors

import contextlib
import logging
import time
from pathlib import Path

import spdl.io
import spdl.utils
import timm
import torch
from spdl.dataloader import PipelineBuilder
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
    parser.add_argument("--max-samples", type=int, default=float("inf"))
    parser.add_argument("--prefix", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--queue-size", type=int, default=16)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--no-compile", action="store_false", dest="compile")
    parser.add_argument("--no-bf16", action="store_false", dest="use_bf16")
    parser.add_argument("--use-nvdec", action="store_true")
    parser.add_argument("--use-nvjpeg", action="store_true")
    args = parser.parse_args(args)
    if args.trace:
        args.max_samples = args.batch_size * 60
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


def _run_inference(dataloader, model):
    _LG.info("Running inference.")
    t0 = time.monotonic()
    num_frames, num_correct_top1, num_correct_top5 = 0, 0, 0
    try:
        for i, (batch, classes) in enumerate(dataloader):
            if i == 20:
                t0 = time.monotonic()
                num_frames, num_correct_top1, num_correct_top5 = 0, 0, 0

            with (
                torch.profiler.record_function(f"iter_{i}"),
                spdl.utils.trace_event(f"iter_{i}"),
            ):

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


def _get_batch_generator(args, device):
    def src():
        with open(args.input_flist) as f:
            i = 0
            for line in f:
                if line := line.strip():
                    yield args.prefix + line
                    if (i := i + 1) >= args.max_samples:
                        return

    class_mapping, _ = get_mappings()

    w, h = 224, 224

    filter_desc = spdl.io.get_video_filter_desc(
        scale_width=256, scale_height=256, crop_width=w, crop_height=h, pix_fmt="rgb24"
    )

    async def _async_decode_func(paths):
        classes = [[class_mapping[parse_wnid(p)]] for p in paths]
        classes = torch.tensor(classes, dtype=torch.int64).to(device)
        buffer = await spdl.io.async_load_image_batch(
            paths,
            width=None,
            height=None,
            pix_fmt=None,
            strict=True,
            filter_desc=filter_desc,
            cuda_config=spdl.io.cuda_config(
                device_index=0,
                allocator=(
                    torch.cuda.caching_allocator_alloc,
                    torch.cuda.caching_allocator_delete,
                ),
            ),
        )
        batch = spdl.io.to_torch(buffer)
        batch = batch.permute((0, 3, 1, 2))
        return batch, classes

    async def _async_decode_nvdec(paths):
        classes = [[class_mapping[parse_wnid(p)]] for p in paths]
        classes = torch.tensor(classes, dtype=torch.int64).to(device)
        buffer = await spdl.io.async_load_image_batch_nvdec(
            paths,
            cuda_config=spdl.io.cuda_config(
                device_index=0,
                allocator=(
                    torch.cuda.caching_allocator_alloc,
                    torch.cuda.caching_allocator_delete,
                ),
            ),
            width=w,
            height=h,
            pix_fmt="rgba",
            strict=True,
        )
        batch = spdl.io.to_torch(buffer)[:, :-1, :, :]
        return batch, classes

    async def _async_decode_nvjpeg(paths):
        classes = [[class_mapping[parse_wnid(p)]] for p in paths]
        classes = torch.tensor(classes, dtype=torch.int64).to(device)
        buffer = await spdl.io.async_load_image_batch_nvjpeg(
            paths,
            cuda_config=spdl.io.cuda_config(
                device_index=0,
                allocator=(
                    torch.cuda.caching_allocator_alloc,
                    torch.cuda.caching_allocator_delete,
                ),
            ),
            width=w,
            height=h,
            pix_fmt="rgb",
            # strict=True,
        )
        batch = spdl.io.to_torch(buffer)
        return batch, classes

    apl = PipelineBuilder().add_source(src()).aggregate(args.batch_size, drop_last=True)

    if args.use_nvdec:
        apl.pipe(_async_decode_nvdec, concurrency=7)
    elif args.use_nvjpeg:
        apl.pipe(_async_decode_nvjpeg, concurrency=args.num_threads)
    else:
        apl.pipe(_async_decode_func, concurrency=args.num_threads)
    apl.add_sink(args.queue_size)
    return apl.build(num_threads=args.num_threads)


def _main(args=None):
    args = _parse_args(args)
    _init(args.debug)
    _LG.info(args)

    device = torch.device("cuda:0")
    model = _get_model(args.batch_size, device, args.compile, args.use_bf16)
    pipeline = _get_batch_generator(args, device)
    print(pipeline)

    trace_path = f"{args.trace}"
    if args.use_nvjpeg:
        trace_path = f"{trace_path}.nvjpeg"
    if args.use_nvdec:
        trace_path = f"{trace_path}.nvdec"

    with (
        torch.no_grad(),
        profile() if args.trace else contextlib.nullcontext() as prof,
        spdl.utils.tracing(f"{trace_path}.pftrace", enable=args.trace is not None),
        pipeline.auto_stop(timeout=1),
    ):
        _run_inference(pipeline.get_iterator(), model)

    if args.trace:
        prof.export_chrome_trace(f"{trace_path}.json")


def _init(debug):
    _init_logging(debug)

    spdl.utils.set_ffmpeg_log_level(16)


def _init_logging(debug=False):
    fmt = "%(asctime)s [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level)


if __name__ == "__main__":
    _main()
