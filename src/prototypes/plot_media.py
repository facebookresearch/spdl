"""Plot the frames generated by NVDEC decoder"""

import asyncio
import logging

from pathlib import Path

import numpy as np

import spdl.io
import spdl.utils


def _parse_python_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "-i", "--input", help="Input media file.", type=Path, required=True
    )
    parser.add_argument(
        "-t",
        "--type",
        help="Input media file type.",
        choices=["video", "image"],
        default="video",
    )
    parser.add_argument(
        "--pix-fmt",
        default="rgba",
    )
    parser.add_argument(
        "-o", "--output-dir", help="Output directory.", type=Path, required=True
    )
    parser.add_argument("--gpu", type=int)
    parser.add_argument("others", nargs="*")
    return parser.parse_args()


async def decode_packets(packets, gpu, pix_fmt):
    if gpu is None:
        frames = await spdl.io.async_decode_packets(packets, pix_fmt=pix_fmt)
        buffer = await spdl.io.async_convert_frames(frames)
        return spdl.io.to_numpy(buffer)
    frames = await spdl.io.async_decode_packets_nvdec(
        packets, cuda_device_index=gpu, pix_fmt=pix_fmt
    )
    buffer = await spdl.io.async_convert_frames(frames)
    return spdl.io.to_torch(buffer).cpu().numpy()


async def decode_video(src, gpu, pix_fmt):
    ts = [(1, 1.05), (10, 10.05), (20, 20.05)]
    gen = spdl.io.async_streaming_demux("video", src, timestamps=ts)
    aws = [decode_packets(packets, gpu, pix_fmt) async for packets in gen]
    return await asyncio.gather(*aws)


async def decode_image(src, gpu, pix_fmt):
    packets = await spdl.io.async_demux_media("image", src)
    array = await decode_packets(packets, gpu, pix_fmt)
    return [array[None]]


def decode(src_path, type, gpu, pix_fmt):
    """Test the python wrapper of SPDL"""

    src = str(src_path.resolve())
    if type == "video":
        coro = decode_video(src, gpu, pix_fmt)
    if type == "image":
        coro = decode_image(src, gpu, pix_fmt)
    return asyncio.run(coro)


def _plot(frameset, gpu, pix_fmt, output_dir):
    from PIL import Image

    device = "cpu" if gpu is None else "nvdec"

    mode = None
    match pix_fmt:
        case "rgba" | "bgra":
            mode = pix_fmt.upper()
        case _:
            pass

    for i, frames in enumerate(frameset):
        for j, array in enumerate(frames):
            print(array.shape)  # CHW
            if array.shape[0] == 1:
                array = array[0]
            elif array.shape[0] <= 4:
                array = np.moveaxis(array, 0, 2)
            path = output_dir / f"{i}_{j}_{pix_fmt}_{device}.png"
            print(f"Saving {path}", flush=True)
            Image.fromarray(array, mode=mode).save(path)


def _main():
    args = _parse_python_args()
    _init(args.debug)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = decode(
        args.input,
        args.type,
        args.gpu,
        args.pix_fmt,
    )

    _plot(frames, args.gpu, args.pix_fmt, output_dir)


def _init(debug, demuxer_threads=1, decoder_threads=1):
    logging.basicConfig(level=logging.INFO)
    if debug:
        logging.getLogger("spdl").setLevel(logging.DEBUG)
        spdl.utils.set_ffmpeg_log_level(40)

    folly_args = [
        f"--spdl_demuxer_executor_threads={demuxer_threads}",
        f"--spdl_decoder_executor_threads={decoder_threads}",
        f"--logging={'DBG' if debug else 'INFO'}",
    ]
    spdl.utils.init_folly(folly_args)


if __name__ == "__main__":
    _main()
