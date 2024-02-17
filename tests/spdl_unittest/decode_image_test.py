import numpy as np
import spdl.utils
from spdl import libspdl


def test_decode_image():
    src = "test.png"
    future = libspdl.decode_image(src)
    result = future.get()

    array = libspdl.to_numpy(result[0])

    assert array.dtype == np.uint8
    assert array.shape == (1, 1, 300, 660)
