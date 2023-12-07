import logging
import os

from spdl.lib import libspdl


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _main():
    args = _parse_args()
    _init_logging(args.debug)

    src = "NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4"
    src2 = f"mmap://{src}"

    engine = libspdl.Engine(3, 6, 10)
    engine.enqueue(src, [3.0])
    engine.enqueue(src2, [3.0])
    engine.dequeue()
    engine.dequeue()


def _init_logging(debug):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO
    )
    os.environ["GLOG_logtostderr"] = "1"

if __name__ == '__main__':
    _main()
