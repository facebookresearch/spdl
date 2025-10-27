# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from tempfile import NamedTemporaryFile

import numpy as np
import spdl.io
from parameterized import parameterized

from ..fixture import FFMPEG_CLI, get_sample, load_ref_video


class TestEncodeVideoMultiColor(unittest.TestCase):
    @parameterized.expand(
        [
            ("rgb24",),
            ("bgr24",),
            ("yuv444p",),
        ]
    )
    def test_encode_video_multi_color(self, pix_fmt: str) -> None:
        height, width = 240, 320
        frame_rate = (30000, 1001)
        duration = 3
        batch_size = 32

        num_frames = int(frame_rate[0] / frame_rate[1] * duration)
        match pix_fmt:
            case "yuv444p":
                shape = (num_frames, 3, height, width)
            case _:
                shape = (num_frames, height, width, 3)
        ref = np.random.randint(0, 255, size=shape, dtype=np.uint8)

        with NamedTemporaryFile(suffix=".raw") as f:
            f.close()  # for windows
            muxer = spdl.io.Muxer(f.name, format="rawvideo")
            encoder = muxer.add_encode_stream(
                config=spdl.io.video_encode_config(
                    height=height,
                    width=width,
                    pix_fmt=pix_fmt,
                    frame_rate=frame_rate,
                ),
            )
            with muxer.open():
                for start in range(0, num_frames, batch_size):
                    frames = spdl.io.create_reference_video_frame(
                        array=ref[start : start + batch_size, ...],
                        pix_fmt=pix_fmt,
                        frame_rate=frame_rate,
                        pts=start,
                    )
                    print(frames, flush=True)

                    if (packets := encoder.encode(frames)) is not None:
                        print(packets)
                        muxer.write(0, packets)

                if (packets := encoder.flush()) is not None:
                    print(packets)
                    muxer.write(0, packets)

            hyp = load_ref_video(
                f.name,
                filter_desc=None,
                shape=shape,
                dtype=np.uint8,
                raw={
                    "pix_fmt": pix_fmt,
                    "width": str(width),
                    "height": str(height),
                },
            )

            np.testing.assert_array_equal(hyp, ref)


class TestEncodeVideoGray(unittest.TestCase):
    @parameterized.expand(
        [
            ("gray8",),
            ("gray16",),
        ]
    )
    def test_encode_video_gray(self, pix_fmt: str) -> None:
        height, width = 240, 320
        frame_rate = (30000, 1001)
        duration = 3
        batch_size = 32

        num_frames = int(frame_rate[0] / frame_rate[1] * duration)
        shape = (num_frames, height, width)
        dtype = np.uint8 if pix_fmt == "gray8" else np.int16
        ref = np.random.randint(0, 255, size=shape, dtype=dtype)

        with NamedTemporaryFile(suffix=".raw") as f:
            f.close()  # for windows
            muxer = spdl.io.Muxer(f.name, format="rawvideo")
            encoder = muxer.add_encode_stream(
                config=spdl.io.video_encode_config(
                    height=height,
                    width=width,
                    pix_fmt=pix_fmt,
                    frame_rate=frame_rate,
                ),
            )
            with muxer.open():
                for start in range(0, num_frames, batch_size):
                    frames = spdl.io.create_reference_video_frame(
                        array=ref[start : start + batch_size, ...],
                        pix_fmt=pix_fmt,
                        frame_rate=frame_rate,
                        pts=start,
                    )
                    print(frames, flush=True)

                    if (packets := encoder.encode(frames)) is not None:
                        print(packets)
                        muxer.write(0, packets)

                if (packets := encoder.flush()) is not None:
                    print(packets)
                    muxer.write(0, packets)

            hyp = load_ref_video(
                f.name,
                filter_desc=None,
                shape=shape,
                dtype=dtype,
                raw={
                    "pix_fmt": pix_fmt,
                    "width": str(width),
                    "height": str(height),
                },
            )

            np.testing.assert_array_equal(hyp, ref)


class TestRemuxVideo(unittest.TestCase):
    def test_remux_video(self) -> None:
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc2 "
            "-frames:v 300 -pix_fmt yuvj420p sample.mp4"
        )

        sample = get_sample(cmd)

        demuxer = spdl.io.Demuxer(sample.path)

        with NamedTemporaryFile(suffix=".mp4") as f:
            f.close()  # for windows
            muxer = spdl.io.Muxer(f.name)
            muxer.add_remux_stream(demuxer.video_codec)

            with muxer.open():
                for packets in demuxer.streaming_demux(duration=1):
                    muxer.write(0, packets)

            ref = load_ref_video(
                sample.path,
                filter_desc=None,
                shape=(300, 1, 360, 320),
                dtype=np.uint8,
            )

            hyp = load_ref_video(
                f.name,
                filter_desc=None,
                shape=(300, 1, 360, 320),
                dtype=np.uint8,
            )

            np.testing.assert_array_equal(hyp, ref)
