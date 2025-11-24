# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import gc
import sys
import unittest

import numpy as np
import spdl.io
import spdl.io.utils
from spdl.io import get_video_filter_desc

from ..fixture import FFMPEG_CLI, get_sample


class TestBufferConversionRefcount(unittest.TestCase):
    def test_buffer_conversion_refcount(self) -> None:
        """NumPy array created from Buffer should increment a reference to the buffer
        so that array keeps working after the original Buffer variable is deleted.
        """
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc,format=yuv420p "
            "-frames:v 100 sample.mp4"
        )
        sample = get_sample(cmd)

        buf = spdl.io.load_video(
            sample.path,
            filter_desc=get_video_filter_desc(pix_fmt="rgb24"),
        )

        self.assertTrue(hasattr(buf, "__array_interface__"))
        print(f"{buf.__array_interface__=}")

        gc.collect()

        n = sys.getrefcount(buf)

        arr = np.array(buf, copy=False)

        n1 = sys.getrefcount(buf)
        self.assertEqual(n1, n + 1)

        print(f"{arr.__array_interface__=}")
        self.assertIsNot(arr.__array_interface__, buf.__array_interface__)
        self.assertEqual(arr.__array_interface__, buf.__array_interface__)

        vals = arr.tolist()

        # Not sure if this will properly fail in case that NumPy array does not
        # keep the reference to the Buffer object. But let's do it anyways
        del buf
        gc.collect()

        vals2 = arr.tolist()
        self.assertEqual(vals, vals2)
