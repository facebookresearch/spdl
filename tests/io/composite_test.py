# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import call, patch, sentinel

from spdl.io import _composite


class CompositeKwargsTest(unittest.TestCase):
    def test_single_item_loaders_forward_demuxer_kwargs(self) -> None:
        """Compatibility keyword arguments reach each underlying demux helper."""
        cases = (
            ("audio", _composite.load_audio, "demux_audio"),
            ("video", _composite.load_video, "demux_video"),
            ("image", _composite.load_image, "demux_image"),
        )

        for name, loader, demux_name in cases:
            with self.subTest(name=name):
                with (
                    patch.object(
                        _composite._core,
                        demux_name,
                        return_value=sentinel.packets,
                    ) as demux,
                    patch.object(
                        _composite,
                        "_load_packets",
                        return_value=sentinel.buffer,
                    ),
                ):
                    result = loader("source", _adaptor=sentinel.adaptor)

                self.assertIs(result, sentinel.buffer)
                self.assertEqual(demux.call_args.kwargs["_adaptor"], sentinel.adaptor)

    def test_batch_loader_forwards_demuxer_kwargs(self) -> None:
        """Batch image loading forwards compatibility kwargs for every source."""
        with (
            patch.object(
                _composite,
                "_decode",
                side_effect=[sentinel.frame1, sentinel.frame2],
            ) as decode,
            patch.object(
                _composite._core,
                "convert_frames",
                return_value=sentinel.buffer,
            ),
        ):
            result = _composite.load_image_batch(
                ["one", "two"],
                width=None,
                height=None,
                filter_desc=None,
                _adaptor=sentinel.adaptor,
            )

        self.assertIs(result, sentinel.buffer)
        self.assertEqual(
            decode.call_args_list,
            [
                call("one", None, None, None, _adaptor=sentinel.adaptor),
                call("two", None, None, None, _adaptor=sentinel.adaptor),
            ],
        )
