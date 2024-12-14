# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile

import numpy as np
import spdl.io


def test_load_npz():
    """spdl.io.load_npz() should load a .npz file."""
    x = np.arange(10)
    y = np.sin(x)

    with tempfile.TemporaryFile() as f:
        np.savez(f, x=x, y=y)
        f.seek(0)
        data = spdl.io.load_npz(f.read())

    assert np.array_equal(data["x"], x)
    assert np.array_equal(data["y"], y)
