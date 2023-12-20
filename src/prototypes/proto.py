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
        print(frame.shape)
        plt.imshow(frame)
        plt.savefig(f"tmp/frame_{k}_{i:03d}.png")


def _main():
    args = _parse_python_args()
    libspdl.init_folly(args.others)

    _init_logging(args.debug)

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
        a = libspdl.to_numpy(engine.dequeue().get_batch(), format="NHWC")
        print(a.shape, a.dtype)
        if args.plot:
            _plot(a, i)


def _init_logging(debug):
    logging.basicConfig(level=logging.INFO)
    if debug:
        logging.getLogger("spdl").setLevel(logging.DEBUG)


if __name__ == "__main__":
    _main()
