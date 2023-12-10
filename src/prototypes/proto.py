import logging

import numpy as np
import matplotlib.pyplot as plt

from spdl.lib import libspdl


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("-i", "--input-video", help="Input video file.", required=True)
    return parser.parse_args()


def _main():
    args = _parse_args()
    _init_logging(args.debug)

    src = args.input_video
    src2 = f"mmap://{src}"

    engine = libspdl.Engine(3, 6, 10)
    engine.enqueue(src, [0.0])
    buffer = engine.dequeue()
    a = np.array(buffer, copy=False)
    del buffer
    print(a.shape, a.dtype)
    for frame in a:
        print(frame.shape)
        plt.imshow(frame[0])
        plt.show()

    engine = libspdl.Engine(
        3, 6, 10, frame_rate=(60, 1), width=36, height=48, pix_fmt="rgb24"
    )
    engine.enqueue(src2, [0.0])
    buffer = engine.dequeue()
    a = np.array(buffer, copy=False)
    del buffer
    print(a.shape, a.dtype)
    for frame in a:
        print(frame.shape)
        plt.imshow(frame)
        plt.show()


def _init_logging(debug):
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)


if __name__ == "__main__":
    _main()
