# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import unittest

from spdl.io import _composite


class CompositeSignatureTest(unittest.TestCase):
    def test_loaders_do_not_accept_arbitrary_kwargs(self) -> None:
        """Composite loaders expose only supported keyword arguments."""
        loaders = (
            _composite.load_audio,
            _composite.load_video,
            _composite.load_image,
            _composite.load_image_batch,
        )

        for loader in loaders:
            with self.subTest(loader=loader.__name__):
                parameter_kinds = {
                    parameter.kind
                    for parameter in inspect.signature(loader).parameters.values()
                }
                self.assertNotIn(inspect.Parameter.VAR_KEYWORD, parameter_kinds)
