# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import spdl.io._transfer
import torch
from spdl.io import transfer_tensor
from spdl.io._transfer import _recursive_apply


def fn(x: int) -> int:
    return 2 * x


@dataclass
class _Item:
    x: int
    y: int
    z: int = field(default=1, init=False)


def test_recursive_apply_basic():
    obj = (1, 2, 3)
    assert _recursive_apply(fn, obj) == (2, 4, 6)

    obj = [4, 5, 6]
    assert _recursive_apply(fn, obj) == [8, 10, 12]

    obj = {"a": 7, "b": 8}
    assert _recursive_apply(fn, obj) == {"a": 14, "b": 16}

    obj = defaultdict(list)
    obj["a"].append(9)
    obj["b"].append(10)

    ref = defaultdict(list)
    ref["a"].append(18)
    ref["b"].append(20)

    assert _recursive_apply(fn, obj) == ref

    obj = _Item(11, 12)
    ref = _Item(22, 24)
    ref.z = fn(ref.z)
    assert _recursive_apply(fn, obj) == ref


def test_recursive_apply_nested():
    obj = (1, 2, (3, 4, 5), 6)
    assert _recursive_apply(fn, obj) == (2, 4, (6, 8, 10), 12)

    obj = [1, 2, [3, 4, 5], 6]
    assert _recursive_apply(fn, obj) == [2, 4, [6, 8, 10], 12]

    obj = {"a": 1, "b": 2, "c": [3, 4, 5], "d": 6}
    assert _recursive_apply(fn, obj) == {"a": 2, "b": 4, "c": [6, 8, 10], "d": 12}

    obj = (1, 2, (3, 4, (5, 6, 7), 8), 9)
    assert _recursive_apply(fn, obj) == (2, 4, (6, 8, (10, 12, 14), 16), 18)

    obj = [1, 2, [3, 4, [5, 6, [7, 8, 9]]], 10]
    assert _recursive_apply(fn, obj) == [2, 4, [6, 8, [10, 12, [14, 16, 18]]], 20]

    obj = {"a": 1, "b": 2, "c": {"d": 3, "e": 4, "f": {"g": 5, "h": 6, "i": 7}}, "j": 8}
    assert _recursive_apply(fn, obj) == {
        "a": 2,
        "b": 4,
        "c": {"d": 6, "e": 8, "f": {"g": 10, "h": 12, "i": 14}},
        "j": 16,
    }

    obj = _Item(1, 2)
    obj.z = _Item(5, 6)
    ref = _Item(2, 4)
    ref.z = _Item(10, 12)
    ref.z.z = 2
    assert _recursive_apply(fn, obj) == ref


@patch("torch.cuda.stream")
@patch("torch.cuda.Stream")
@patch("torch.cuda.device_count")
def test_gpu_transfer(
    mock_device_count,
    mock_Stream,
    mock_stream_func,
):
    """The data is transferred to CUDA asynchronously.

    transfer_tensor() does the following 5 things.

    1. Create CUDA stream. (call torch.cuda.Stream with cuda device)
    2. Activate the stream. (call torch.cuda.stream with 1)
    3. `pin_memory` is called.
    4. `to` is called with CUDA device and non_blocking=True
    5. The stream is synchronized.
    """

    def _test():
        assert not hasattr(spdl.io._transfer._THREAD_LOCAL, "transfer")

        mock_stream_obj = MagicMock()
        device = torch.device("cuda:0")
        data = torch.zeros((3, 4, 5))

        mock_device_count.return_value = 1
        mock_Stream.return_value = mock_stream_obj
        with (
            patch.object(data, "pin_memory", return_value=data) as mock_pin_memory,
            patch.object(data, "to", return_value=data) as mock_to,
        ):
            _ = transfer_tensor({"foo": data})
            # Check 3
            mock_pin_memory.assert_called_once()
            # Check 4
            mock_to.assert_called_once_with(device, non_blocking=True)

        # check 1
        mock_Stream.assert_called_once_with(device)
        # check 2
        mock_stream_func.assert_called_once_with(mock_stream_obj)
        # Check 5
        mock_stream_obj.synchronize.assert_called_once()

        assert hasattr(spdl.io._transfer._THREAD_LOCAL, "transfer")

    with ThreadPoolExecutor(max_workers=1) as exec:
        exec.submit(_test).result()
