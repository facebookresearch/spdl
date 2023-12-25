import logging

from spdl import libspdl


def _parse_python_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("-i", "--input-video", help="Input video file.", required=True)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("others", nargs="*")
    return parser.parse_args()


def _plot(frames, k):
    import matplotlib.pyplot as plt

    for i, frame in enumerate(frames):
        plt.imshow(frame)
        plt.savefig(f"tmp/frame_{k}_{i:03d}.png")


def _main():
    args = _parse_python_args()
    _init_logging(args.debug)
    libspdl.init_folly(args.others)
    if args.debug:
        libspdl.set_ffmpeg_log_level(54)

    src = args.input_video
    src2 = f"mmap://{src}"

    configs = [
        {"src": src, "timestamps": [0.0]},
        {
            "src": src2,
            "timestamps": [0.0],
            "frame_rate": (60, 1),
            "width": 36,
            "height": 48,
            "pix_fmt": "rgb24",
        },
    ]

    for i, cfg in enumerate(configs):
        if args.gpu:
            cfg["decoder"] = "h264_cuvid"
            # cfg["cuda_device_index"] = 0

        engine = libspdl.Engine(10)
        engine.enqueue(**cfg)
        decoded_frames = engine.dequeue()
        print(len(decoded_frames))
        print(decoded_frames.width, decoded_frames.height)
        sliced = decoded_frames[2:7:2]
        print(len(sliced))
        del sliced
        a = libspdl.to_numpy(decoded_frames.to_video_buffer(), format="NHWC")
        print(a.shape, a.dtype)
        if args.plot:
            _plot(a, 2 * i)
        a = libspdl.to_numpy(decoded_frames.to_video_buffer(0), format="NHWC")
        print(a.shape, a.dtype)
        if args.plot:
            _plot(a, 2 * i + 1)


def _init_logging(debug):
    logging.basicConfig(level=logging.INFO)
    if debug:
        logging.getLogger("spdl").setLevel(logging.DEBUG)


if __name__ == "__main__":
    _main()
