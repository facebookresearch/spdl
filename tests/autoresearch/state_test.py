# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

from spdl.autoresearch._common._state import (
    _normalize_config,
    _normalize_state,
    SCHEMA_VERSION,
)


class StateTest(unittest.TestCase):
    def test_schema_normalizers_add_versions_and_defaults(self) -> None:
        """Empty dicts are normalized with schema version and default values."""
        config = _normalize_config({})
        state = _normalize_state({})

        self.assertEqual(SCHEMA_VERSION, config["schema_version"])
        self.assertEqual(SCHEMA_VERSION, state["schema_version"])
        self.assertEqual("auto", config["platform"])
        self.assertEqual([], state["history"])
