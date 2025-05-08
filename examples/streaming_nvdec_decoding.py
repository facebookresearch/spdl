# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""This example shows how to decode a video with GPU in streaming fashion."""

__all__ = [
    "main",
    "parse_args",
    "run",
    "decode",
    "torch_cuda_warmup",
]

import argparse
import contextlib
import logging
import pathlib
import time

import spdl.io
import torch
from PIL import Image
from spdl.io import CUDAConfig
from torch.profiler import profile

# pyre-strict


def parse_args(args: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments.

    Args:
        args: The command line arguments. By default it reads ``sys.argv``.

    Returns:
        Tuple of parsed arguments and unused arguments, as returned by
        :py:meth:`argparse.ArgumentParser.parse_known_args`.
    """

    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--input-file", required=True, help="The input video to process."
    )
    parser.add_argument(
        "--plot-dir",
        type=pathlib.Path,
        help="If provided, plot the result to the given directory.",
    )
    parser.add_argument(
        "--trace-path",
        help="If provided, trace the execution. e.g. 'trace.json.gz'",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        help="The CUDA device index. By default it use the last one.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Rescale the video to this width. Provide -1 to disable.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=240,
        help="Rescale the video to this height. Provide -1 to disable.",
    )
    return parser.parse_known_args(args)


def decode(
    src: str,
    device_config: CUDAConfig,
    post_processing_params: dict[str, int],
    profiler: torch.profiler.profile | None,
    plot_dir: pathlib.Path | None,
) -> None:
    """Decode video in streaming fashion with optional resizing, profiling and exporting.

    Args:
        src: The path or URL to the source video.
        device_config: The GPU configuration.
        post_processing_params: Post processing argument.
            See :py:func:`spdl.io.streaming_load_video_nvdec`.
        profiler: PyTorch Profiler or ``None``.
        plot_dir: If provided, the decoded frames are exported as images to the directory.
    """
    streamer = spdl.io.streaming_load_video_nvdec(
        src,
        device_config,
        num_frames=32,
        post_processing_params=post_processing_params,
    )

    i, num_frames = 0, 0
    t0 = time.monotonic()
    for buffers in streamer:
        buffer = spdl.io.nv12_to_rgb(buffers, device_config=device_config, sync=True)
        tensor = spdl.io.to_torch(buffer)
        num_frames += len(tensor)

        if plot_dir is not None:
            for f in tensor.permute(0, 2, 3, 1):
                img = Image.fromarray(f.cpu().numpy())
                img.save(plot_dir / f"{i:05d}.png")
                i += 1

        if profiler is not None:
            profiler.step()
            if num_frames >= 500:
                break

    elapsed = time.monotonic() - t0
    qps = num_frames / elapsed
    print(f"Processed {num_frames} frames in {elapsed:.1f} sec. QPS: {qps:.1f}")


def torch_cuda_warmup(device_index: int | None) -> tuple[int, torch.cuda.Stream]:
    """Initialize the CUDA context perform dry-run.

    Args:
        device_index: The CUDA device to use. If ``None``, the last available device is used.
    """
    assert torch.cuda.is_available()

    cuda_index: int = device_index or (torch.cuda.device_count() - 1)
    stream = torch.cuda.Stream(device=cuda_index)
    with torch.cuda.stream(stream):
        a = torch.empty([32, 3, 1080, 1920])
        a.pin_memory().to(f"cuda:{cuda_index}", non_blocking=True)
    stream.synchronize()
    return cuda_index, stream


def run(
    src: str,
    device_index: int | None,
    post_processing_params: dict[str, int],
    profiler: torch.profiler.profile,
    plot_dir: pathlib.Path,
) -> None:
    """Run the benchmark."""
    cuda_index, stream = torch_cuda_warmup(device_index)

    device_config = spdl.io.cuda_config(
        device_index=cuda_index,
        allocator=(
            torch.cuda.caching_allocator_alloc,
            torch.cuda.caching_allocator_delete,
        ),
        stream=stream.cuda_stream,
    )

    for i in range(3):
        with torch.autograd.profiler.record_function(f"decode_{i}"):
            decode(src, device_config, post_processing_params, profiler, plot_dir)


def main(args: list[str] | None = None) -> None:
    """The main entrypoint for the CLI."""
    ns, _ = parse_args(args)

    logging.basicConfig(level=logging.INFO)

    prof = None
    post_process = {
        "scale_width": ns.width if ns.width > 0 else None,
        "scale_height": ns.height if ns.height > 0 else None,
    }
    with contextlib.ExitStack() as stack:
        if ns.trace_path:
            prof = stack.enter_context(
                profile(
                    with_stack=True,
                    on_trace_ready=lambda p: p.export_chrome_trace(ns.trace_path),
                )
            )

        run(ns.input_file, ns.device_index, post_process, prof, ns.plot_dir)


if __name__ == "__main__":
    main()
