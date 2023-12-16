import sys
import logging

import numpy as np

from spdl.lib import libspdl


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("-i", "--input-video", help="Input video file.", required=True)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def _plot(frames):
    import matplotlib.pyplot as plt

    if frames.shape[1] in [1, 3]:
        frames = np.moveaxis(frames, 1, -1)

    for frame in frames:
        print(frame.shape)
        plt.imshow(frame)
        plt.show()


def _main():
    args = _parse_args()

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

    for cfg in configs:
        engine = libspdl.Engine(3, 6, 10)
        engine.enqueue(**cfg)
        buffer = engine.dequeue()
        a = np.array(buffer, copy=False)
        del buffer
        print(a.shape, a.dtype)
        if args.plot:
            _plot(a)


def _init_logging(debug):
    logging.basicConfig(level=logging.INFO)
    if debug:
        logging.getLogger("spdl").setLevel(logging.DEBUG)

    libspdl.init_folly(sys.argv[0])


if __name__ == "__main__":
    _main()
