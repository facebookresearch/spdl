"""Test decoding with image classification"""

import logging
import time
from pathlib import Path

import spdl.io
import spdl.utils

import timm
import torch
import torchvision.transforms
from spdl.dataloader._task_runner import apply_async, BackgroundTaskProcessor
from spdl.dataloader._utils import _iter_flist

_LG = logging.getLogger(__name__)


def _parse_args(args):
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--input-flist", type=Path, required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-demux-threads", type=int, required=True)
    parser.add_argument("--num-decode-threads", type=int, required=True)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--trace", type=Path)
    return parser.parse_args(args)


def _get_model(device):
    model = timm.create_model("mobilenetv3_large_100", pretrained=True)
    model = model.to(device)
    model.eval()

    config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.transforms_factory.create_transform(**config)

    class to_float(torch.nn.Module):
        def forward(self, x):
            # print(x.shape, x.dtype, x.device)
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
    num_frames = 0
    try:
        for buffer in dataloader:
            batch = spdl.io.to_torch(buffer)
            # print(f"{batch.shape=}, {batch.dtype=}, {batch.device=}")
            num_frames += batch.shape[0]

            batch = batch.permute((0, 3, 1, 2)).to(device)
            batch = transform(batch)

            output = model(batch)
            probabilities = torch.nn.functional.softmax(output, dim=-1)
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            # print(f"{top5_prob.shape=}, {top5_catid.shape}")
    finally:
        elapsed = time.monotonic() - t0
        qps = num_frames / elapsed
        _LG.info(f"QPS={qps:.2f} ({num_frames}/{elapsed:.2f})")


def _main(args=None):
    args = _parse_args(args)
    _init(
        args.debug,
        args.num_demux_threads,
        args.num_decode_threads,
        args.worker_id,
    )

    async def _decode_func(paths):
        return await spdl.io.async_batch_load_image(
            paths,
            width=256,
            height=256,
            pix_fmt="rgb24",
            strict=False,
        )

    gen = _iter_flist(
        args.input_flist,
        prefix=args.prefix,
        batch_size=args.batch_size,
        n=args.worker_id,
        N=args.num_workers,
        max=args.max_samples,
    )

    dataloader = BackgroundTaskProcessor(apply_async(_decode_func, gen))
    device = torch.device("cuda:0")
    model, transform = _get_model(device)
    print(transform)

    dataloader.start()
    try:
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
