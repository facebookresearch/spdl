# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import unittest
from unittest import mock

import spdl.io
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


class LoadImageBatchNvjpegTest(unittest.TestCase):
    def test_forwards_memoryview_source(self) -> None:
        """The public loader normalizes and forwards borrowed image data."""
        source = memoryview(b"image")
        device_config = mock.sentinel.device_config

        with mock.patch.object(
            _composite._core,
            "decode_image_nvjpeg",
        ) as decode_image:
            spdl.io.load_image_batch_nvjpeg(
                [source],
                device_config=device_config,
                width=4,
                height=2,
            )

        decode_image.assert_called_once()
        forwarded_sources = decode_image.call_args.args[0]
        self.assertEqual(len(forwarded_sources), 1)
        self.assertIsInstance(forwarded_sources[0], memoryview)
        self.assertIs(forwarded_sources[0].obj, source.obj)
        self.assertEqual(
            decode_image.call_args.kwargs,
            {
                "scale_width": 4,
                "scale_height": 2,
                "device_config": device_config,
                "pix_fmt": "rgb",
            },
        )
