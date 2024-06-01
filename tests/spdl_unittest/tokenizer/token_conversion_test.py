import numpy as np
import spdl.io
from spdl.lib import _libspdl


def test_convert_tokens_1d():
    tokens = list(range(10))
    ref = np.array(tokens, dtype=np.int32)

    for i in range(1, 10):
        buffer = _libspdl.convert_tokens_1d(tokens[:i])
        array = spdl.io.to_numpy(buffer)

        np.testing.assert_array_equal(array, ref[:i])

        buffer = _libspdl.convert_tokens_1d(tokens, i)
        array = spdl.io.to_numpy(buffer)

        np.testing.assert_array_equal(array, ref[:i])

    for i in range(11, 20):
        buffer = _libspdl.convert_tokens_1d(tokens, i)
        array = spdl.io.to_numpy(buffer)

        np.testing.assert_array_equal(array[:10], ref)
        assert np.all(array[10:i] == 0)


def _to_array(tokens: list[list[int]]):
    ref = np.zeros((len(tokens), max(len(t) for t in tokens)), dtype=np.int32)
    for i in range(len(tokens)):
        ref[i, : len(tokens[i])] = tokens[i]
    return ref


def test_convert_tokens_2d():
    tokens = [list(range(10))[::-1], list(range(20))]
    ref = _to_array(tokens)

    buffer = _libspdl.convert_tokens_2d(tokens)
    array = spdl.io.to_numpy(buffer)

    np.testing.assert_array_equal(ref, array)
