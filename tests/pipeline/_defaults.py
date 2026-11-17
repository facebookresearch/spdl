# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test base that restores process-wide pipeline defaults after each case."""

import unittest

from spdl.pipeline import config


class _DefaultsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.addCleanup(
            config.set_default_background_tasks, config.get_default_background_tasks()
        )
        self.addCleanup(
            config.set_default_build_callback, config.get_default_build_callback()
        )
        self.addCleanup(config.set_default_hook_class, config.get_default_hook_class())
        self.addCleanup(
            config.set_default_queue_class, config.get_default_queue_class()
        )
