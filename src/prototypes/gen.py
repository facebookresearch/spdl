import time

from typing import Generator, Any
from concurrent.futures import Future, CancelledError

import spdl.io
import spdl.utils

def _load_image(src):
    packets = yield spdl.io.demux_media("image", src)
    print(f"{packets=}")
    frames = yield spdl.io.decode_packets(packets)
    print(f"{frames=}")
    yield spdl.io.convert_frames(frames)


def _main():
    spdl.utils.init_folly([])
    future = spdl.io.chain_futures(_load_image("test.png"))
    print(future.result())


if __name__ == "__main__":
    _main()
