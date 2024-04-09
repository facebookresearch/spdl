import gc
import sys

import numpy as np

import spdl.io


def _decode_video(src, pix_fmt=None):
    return spdl.io.chain_futures(
        spdl.io.demux_media("video", src),
        spdl.io.decode_packets,
        spdl.io.convert_frames_cpu,
    ).result()


def test_buffer_conversion_refcount(get_sample):
    """NumPy array created from Buffer should increment a reference to the buffer
    so that array keeps working after the original Buffer variable is deleted.
    """
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 100 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)

    buf = _decode_video(sample.path, pix_fmt="rgb24")

    assert hasattr(buf, "__array_interface__")
    print(f"{buf.__array_interface__=}")

    gc.collect()

    n = sys.getrefcount(buf)

    arr = np.array(buf, copy=False)

    n1 = sys.getrefcount(buf)
    assert n1 == n + 1

    print(f"{arr.__array_interface__=}")
    assert arr.__array_interface__ is not buf.__array_interface__
    assert arr.__array_interface__ == buf.__array_interface__

    vals = arr.tolist()

    # Not sure if this will properly fail in case that NumPy array does not
    # keep the reference to the Buffer object. But let's do it anyways
    del buf
    gc.collect()

    vals2 = arr.tolist()
    assert vals == vals2
