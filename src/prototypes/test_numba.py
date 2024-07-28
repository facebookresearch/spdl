# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test numba __cuda_array_interface__ integration"""

# pyre-ignore-all-errors

from pathlib import Path

import numba.cuda as cuda

import numpy as np
import spdl.io
import spdl.utils


def _parse_args():
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("-i", "--input-video", required=True)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--nvdec", action="store_true")
    parser.add_argument("--num-demuxing-threads", type=int, default=1)
    parser.add_argument("--num-decoding-threads", type=int, default=1)
    # fmt: on
    return parser.parse_args()


def _main():
    args = _parse_args()

    from numba import cuda

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    timestamps = [(0, 0.1), (2, 2.1), (10, 10.1)]
    # cuda.select_device(args.gpu)
    # Needed because SPDL initializes CUDA context only in decoding threads, but
    # the main thread also need a context when copying data to host.
    # Other ways of initializing CUDA context also work, but this one is simple.
    cuda.select_device(args.gpu)
    array = test_nvdec(args, timestamps)
    _plot(array, args.output_dir, "rgba")


def test_nvdec(args, timestamps):
    @spdl.utils.chain_futures
    def _load_video_nvdec():
        gen = spdl.io.streaming_demux("video", args.input_video, timestamps=timestamps)
        arrays = []
        for future in gen:
            packets = yield future
            buffer = yield spdl.io.decode_packets_nvdec(
                packets,
                cuda_device_index=-1 if args.gpu is None else args.gpu,
                width=args.width or -1,
                height=args.height or -1,
                pix_fmt="rgba",
            )
            arrays.append(cuda.as_cuda_array(buffer).copy_to_host())
        yield spdl.utils.create_future(arrays)

    return _load_video_nvdec().result()


def _plot(arrays, output_dir, pix_fmt: str | None):
    from PIL import Image

    mode = None
    if pix_fmt is not None:
        mode = pix_fmt.upper()
        print(f"{mode=}")
    for i, array in enumerate(arrays):
        print(f"{array.shape=}")
        for j, frame in enumerate(array):
            print(f"{frame.shape=}")
            if frame.shape[0] == 1:
                frame = frame[0]
            elif frame.shape[0] in [3, 4]:
                frame = np.moveaxis(frame, 0, 2)
            print(f"{frame.shape=}")
            path = output_dir / f"{i}_{j}.png"
            print(f"Saving {path}", flush=True)
            Image.fromarray(frame, mode).save(path)


if __name__ == "__main__":
    _main()
