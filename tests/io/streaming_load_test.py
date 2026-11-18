# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import spdl.io
from spdl.io._composite import streaming_load_audio, streaming_load_video

from ..fixture import FFMPEG_CLI, get_sample

# 5 seconds of 25 FPS video (125 frames) with audio.
_CMD = (
    f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc=size=64x48:rate=25 "
    "-f lavfi -i sine=sample_rate=16000 -t 5 -c:v libx264 -g 10 "
    "-pix_fmt yuv420p sample.mp4"
)


def _eager_video(src: str, filter_desc: str | None = None) -> np.ndarray:
    if filter_desc is None:
        return spdl.io.to_numpy(spdl.io.load_video(src))
    return spdl.io.to_numpy(spdl.io.load_video(src, filter_desc=filter_desc))


class StreamingLoadVideoTest(unittest.TestCase):
    def test_chunks(self) -> None:
        """Chunks have `num_frames` frames, and together equal the whole video."""
        sample = get_sample(_CMD)
        ref = _eager_video(sample.path)
        for num_frames in (1, 7, 32, 125, 200):
            with self.subTest(num_frames=num_frames):
                chunks = [
                    spdl.io.to_numpy(b)
                    for b in streaming_load_video(sample.path, num_frames)
                ]
                sizes = [len(c) for c in chunks]
                self.assertTrue(all(s == num_frames for s in sizes[:-1]))
                self.assertLessEqual(sizes[-1], num_frames)
                np.testing.assert_array_equal(np.concatenate(chunks), ref, strict=True)

    def test_in_memory_source_and_filter(self) -> None:
        """Bytes sources and custom filters work as in `load_video`."""
        sample = get_sample(_CMD)
        with open(sample.path, "rb") as f:
            data = f.read()
        desc = spdl.io.get_video_filter_desc(
            scale_width=32, scale_height=24, pix_fmt="gray"
        )
        ref = _eager_video(sample.path, desc)
        chunks = [
            spdl.io.to_numpy(b)
            for b in streaming_load_video(data, 50, filter_desc=desc)
        ]
        self.assertEqual([len(c) for c in chunks], [50, 50, 25])
        np.testing.assert_array_equal(np.concatenate(chunks), ref, strict=True)

    def test_no_filter(self) -> None:
        """`filter_desc=None` disables filtering, as in `load_video`."""
        sample = get_sample(_CMD)
        ref = spdl.io.to_numpy(spdl.io.load_video(sample.path, filter_desc=None))
        chunks = [
            spdl.io.to_numpy(b)
            for b in streaming_load_video(sample.path, 64, filter_desc=None)
        ]
        np.testing.assert_array_equal(np.concatenate(chunks), ref, strict=True)

    def test_errors(self) -> None:
        """Invalid sources raise when iterating, and `num_frames` is checked."""
        with self.assertRaises(ValueError):
            streaming_load_video(b"invalid", 0)
        ite = streaming_load_video(b"invalid", 8)
        with self.assertRaises(RuntimeError):
            next(ite)

    def test_failed_init_is_retried(self) -> None:
        """A failure after opening the source raises again on the next call."""
        sample = get_sample(_CMD)
        ite = streaming_load_video(sample.path, 8, filter_desc="no_such_filter")
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                next(ite)

    def test_exhausted(self) -> None:
        """An exhausted iterator keeps raising `StopIteration`."""
        sample = get_sample(_CMD)
        ite = streaming_load_video(sample.path, 200)
        self.assertEqual(len(spdl.io.to_numpy(next(ite))), 125)
        for _ in range(2):
            with self.assertRaises(StopIteration):
                next(ite)


class StreamingLoadAudioTest(unittest.TestCase):
    def test_chunks(self) -> None:
        """Chunks together equal the whole audio."""
        sample = get_sample(_CMD)
        ref = spdl.io.to_numpy(spdl.io.load_audio(sample.path))
        desc = f"{spdl.io.get_audio_filter_desc()},asetnsamples=n=1000:p=0"
        for num_frames in (1, 4, 1000):
            with self.subTest(num_frames=num_frames):
                chunks = [
                    spdl.io.to_numpy(b)
                    for b in streaming_load_audio(
                        sample.path, num_frames, filter_desc=desc
                    )
                ]
                # The default filter outputs planar samples: [channels, samples]
                sizes = [c.shape[-1] for c in chunks]
                self.assertLessEqual(set(sizes[:-1]), {num_frames * 1000})
                np.testing.assert_array_equal(
                    np.concatenate(chunks, axis=-1), ref, strict=True
                )


class StreamingLoadNotPublicTest(unittest.TestCase):
    def test_not_exposed_in_spdl_io(self) -> None:
        """The streaming loaders are not part of the public `spdl.io` API yet."""
        for name in ("streaming_load_video", "streaming_load_audio"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(spdl.io, name))
