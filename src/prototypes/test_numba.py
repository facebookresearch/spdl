"""Test numba __cuda_array_interface__ integration"""

from pathlib import Path

import spdl.libspdl
from matplotlib import pyplot as plt
from numba import cuda


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-video",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
    )
    parser.add_argument(
        "--decoder",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        required=True,
    )
    return parser.parse_args()


def _main():
    args = _parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    future = spdl.libspdl.decode_video(
        args.input_video,
        timestamps=[(0, 0.1)],
        decoder=args.decoder,
        decoder_options=None if args.gpu is None else {"gpu": f"{args.gpu}"},
        cuda_device_index=args.gpu,
    )
    decoded_frames = future.get()

    for i, frames in enumerate(decoded_frames):
        buffer = spdl.libspdl._to_buffer(frames)
        array = cuda.as_cuda_array(buffer).copy_to_host()
        for j, frame in enumerate(array):
            plt.imshow(frame[0])
            plt.savefig(output_dir / f"{i}_{j}.png")


if __name__ == "__main__":
    _main()
