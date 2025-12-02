# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

from spdl.io.utils import get_ffmpeg_filters


class FFmpegUtilsTest(unittest.TestCase):
    def test_get_ffmpeg_filters_contains_common_filters(self):
        expected_filters = ["aresample", "scale"]
        filters = get_ffmpeg_filters()

        # Assert: Confirm that we have a reasonable number of filters (at least 100)
        self.assertGreater(len(filters), 100)
        # Assert: Confirm that common filters like "aresample" and "scale" are present
        for expected_filter in expected_filters:
            self.assertIn(
                expected_filter,
                filters,
                f"Expected filter '{expected_filter}' not found in available filters",
            )
