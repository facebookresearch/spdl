# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import unittest

from spdl.autoresearch._common._visualization import (
    _edge_label,
    _tree_font_sizes,
)


class VisualizationTest(unittest.TestCase):
    def test_tree_edge_label_describes_experiment_evolution(self) -> None:
        """Edge labels show change summary or startup repair attempt number."""
        parent = {"node_id": "001_parent", "name": "parent", "spec": {}}
        child = {
            "node_id": "002_child",
            "name": "child",
            "spec": {
                "change_summary": "raise decode threads",
                "description": "increase decode thread count",
            },
        }
        retry = {
            "node_id": "003_retry",
            "name": "retry",
            "spec": {"_startup_retry_attempt": 2, "description": "repair"},
        }

        self.assertEqual("raise decode threads", _edge_label(parent, child))
        self.assertEqual("startup repair #2", _edge_label(parent, retry))

    def test_tree_font_sizes_grow_with_tree_size(self) -> None:
        """Larger trees get proportionally larger font sizes."""
        small = _tree_font_sizes(4, 1)
        large = _tree_font_sizes(120, 10)

        self.assertGreater(large["title"], small["title"])
        self.assertGreater(large["legend"], small["legend"])
