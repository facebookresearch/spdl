"""Test numba __cuda_array_interface__ integration"""

from pathlib import Path

import numpy as np

import spdl.libspdl


def _parse_args():
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("-i", "--input-video", required=True)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--decoder")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--nvdec", action="store_true")
    parser.add_argument("--num-demuxing-threads", type=int, default=1)
    parser.add_argument("--num-decoding-threads", type=int, default=1)
    # fmt: on
    return parser.parse_args()


def _main():
    args = _parse_args()
    spdl.libspdl.init_folly(
        [
            f"--spdl_demuxer_executor_threads={args.num_demuxing_threads}",
            f"--spdl_decoder_executor_threads={args.num_decoding_threads}",
            f"--logging={'DBG' if args.debug else 'INFO'}",
        ]
    )

    from numba import cuda

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    timestamps = [(0, 0.1)]  # , (2, 2.1), (10, 10.1)]
    if args.nvdec:
        cuda.select_device(args.gpu)
        # Needed because SPDL initializes CUDA context only in decoding threads, but
        # the main thread also need a context when copying data to host.
        # Other ways of initializing CUDA context also work, but this one is simple.
        buffers = test_nvdec(args, timestamps)
    else:
        buffers = test_ffmpeg(args, timestamps)

    arrays = [cuda.as_cuda_array(buf).copy_to_host() for buf in buffers]
    _plot(arrays, args.output_dir)


def test_nvdec(args, timestamps):
    future = spdl.libspdl.decode_video_nvdec(
        args.input_video,
        timestamps=timestamps,
        cuda_device_index=-1 if args.gpu is None else args.gpu,
        width=args.width or -1,
        height=args.height or -1,
    )
    return [spdl.libspdl._BufferWrapper(fut) for fut in future.get()]


def test_ffmpeg(args, timestamps):
    future = spdl.libspdl.decode_video(
        args.input_video,
        timestamps=timestamps,
        decoder=args.decoder,
        decoder_options=None if args.gpu is None else {"gpu": f"{args.gpu}"},
        cuda_device_index=-1 if args.gpu is None else args.gpu,
        width=args.width,
        height=args.height,
    )
    return [spdl.libspdl._to_buffer(fut) for fut in future.get()]


def _plot(arrays, output_dir):
    from PIL import Image

    for i, array in enumerate(arrays):
        print(array.shape)
        for j, frame in enumerate(array):
            if frame.shape[0] == 1:
                frame = frame[0]
            else:
                frame = np.moveaxis(frame, 0, 2)
            path = output_dir / f"{i}_{j}.png"
            print(f"Saving {path}", flush=True)
            Image.fromarray(frame).save(path)


if __name__ == "__main__":
    _main()
