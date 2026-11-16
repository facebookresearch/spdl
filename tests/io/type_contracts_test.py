# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import assert_type, cast, TYPE_CHECKING
from unittest.mock import MagicMock, patch

from spdl.io import _composite, _core

if TYPE_CHECKING:
    from spdl.io import (
        AudioCodec,
        AudioPackets,
        BSF,
        CPUBuffer,
        ImageCodec,
        ImagePackets,
        Muxer,
        VideoCodec,
        VideoPackets,
    )

    def _check_muxer_packets(
        muxer: Muxer, audio: AudioPackets, video: VideoPackets
    ) -> None:
        assert_type(muxer.write(0, audio), None)
        assert_type(muxer.write(0, video), None)

    def _check_image_paths(paths: list[Path]) -> None:
        assert_type(_core.demux_image(paths[0]), ImagePackets)
        assert_type(
            _composite.load_image_batch(paths, width=None, height=None), CPUBuffer
        )

    def _check_bsf_audio(codec: AudioCodec, packets: AudioPackets) -> None:
        bsf = BSF(codec, "null")
        assert_type(bsf, BSF[AudioCodec, AudioPackets])
        assert_type(bsf.filter(packets), AudioPackets | None)
        assert_type(bsf.flush(), AudioPackets | None)
        assert_type(_core.apply_bsf(packets, "null"), AudioPackets | None)

    def _check_bsf_video(codec: VideoCodec, packets: VideoPackets) -> None:
        bsf = BSF(codec, "null")
        assert_type(bsf, BSF[VideoCodec, VideoPackets])
        assert_type(bsf.filter(packets), VideoPackets | None)
        assert_type(bsf.flush(), VideoPackets | None)
        assert_type(_core.apply_bsf(packets, "null"), VideoPackets | None)

    def _check_bsf_image(codec: ImageCodec, packets: ImagePackets) -> None:
        bsf = BSF(codec, "null")
        assert_type(bsf, BSF[ImageCodec, ImagePackets])
        assert_type(bsf.filter(packets), ImagePackets | None)
        assert_type(bsf.flush(), ImagePackets | None)
        assert_type(_core.apply_bsf(packets, "null"), ImagePackets | None)


class TestBsfContract(unittest.TestCase):
    def test_apply_bsf_preserves_no_output(self) -> None:
        """A filter that produces no packets still returns None."""
        packets = cast("AudioPackets", MagicMock())
        with patch.object(_core, "BSF") as factory:
            factory.return_value.filter.return_value = None
            result = _core.apply_bsf(packets, "null")
        self.assertIsNone(result)
        factory.assert_called_once_with(packets.codec, "null")
        factory.return_value.filter.assert_called_once_with(packets, flush=True)

    def test_apply_bsf_requires_codec(self) -> None:
        """Packets without codec metadata retain the existing rejection."""
        mock_packets = MagicMock()
        mock_packets.codec = None
        packets = cast("AudioPackets", mock_packets)
        with self.assertRaisesRegex(ValueError, "does not have codec"):
            _core.apply_bsf(packets, "null")

    def test_reference_frame_retains_owner(self) -> None:
        """The verified dynamic ownership attribute retains the input array."""
        import numpy as np

        array = np.zeros((1, 2), dtype=np.int16)
        frame = SimpleNamespace()
        factory = MagicMock(return_value=frame)
        with patch.object(
            _core,
            "_libspdl",
            SimpleNamespace(create_reference_audio_frame=factory),
        ):
            result = _core.create_reference_audio_frame(array, "s16", 8000, 0)
        self.assertIs(result, frame)
        self.assertIs(frame._array_ref, array)
        factory.assert_called_once_with(
            array=array, sample_fmt="s16", sample_rate=8000, pts=0
        )


class TestImagePathContract(unittest.TestCase):
    def test_batch_path_uses_existing_source_resolver(self) -> None:
        """Path identifiers use the existing resolver without mutating the batch."""
        paths = [Path("one.png")]
        native_demuxer = MagicMock()
        make_demuxer = MagicMock(return_value=native_demuxer)
        frame = MagicMock()
        buffer = MagicMock()
        with (
            patch.object(_core, "_libspdl", SimpleNamespace(make_demuxer=make_demuxer)),
            patch.object(_core, "decode_packets", return_value=frame),
            patch.object(_core, "convert_frames", return_value=buffer) as convert,
        ):
            result = _composite.load_image_batch(
                paths, width=None, height=None, filter_desc=None
            )
        self.assertIs(result, buffer)
        self.assertEqual(paths, [Path("one.png")])
        make_demuxer.assert_called_once_with("one.png", demux_config=None, name=None)
        convert.assert_called_once_with([frame], storage=None)

    def test_path_failure_message_preserves_identifier(self) -> None:
        """Failed Path inputs keep their original identifier in the error message."""
        self.assertEqual(
            _composite._get_err_msg(Path("one.png"), RuntimeError("decode failed")),
            "Failed to decode an image from 'one.png': decode failed.",
        )
