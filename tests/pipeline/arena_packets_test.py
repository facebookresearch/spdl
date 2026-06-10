# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import functools
import gc
import unittest
from typing import Any, cast

import numpy as np
import spdl.io
from spdl.pipeline import iterate_in_subprocess, SharedMemorySegmentPool
from spdl.pipeline._arena._offload import _offload, _restore
from spdl.pipeline._arena._registry import _default_registry

from ..fixture import FFMPEG_CLI, get_sample

CMDS = {
    "audio": f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i sine=frequency=1000:sample_rate=48000:duration=3 -c:a pcm_s16le sample.wav",
    "video": f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 25 sample.mp4",
    "image": f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i color=0x000000,format=gray -frames:v 1 sample.png",
}


def _demux(media_type: str) -> Any:
    """Generate a synthetic sample and demux it into a Packets object."""
    sample = get_sample(CMDS[media_type])
    return getattr(spdl.io, f"demux_{media_type}")(sample.path)


def _demux_from_paths(paths: list[tuple[str, str]]) -> list[object]:
    """Demux each ``(media_type, path)`` pair into a Packets object."""
    return [getattr(spdl.io, f"demux_{mt}")(path) for mt, path in paths]


class PacketsOffloadTest(unittest.TestCase):
    def _pool(self, segment_size: int, count: int) -> SharedMemorySegmentPool:
        pool = SharedMemorySegmentPool(segment_size=segment_size, count=count)
        self.addCleanup(pool.unlink)
        self.addCleanup(pool.close)
        return pool

    def _assert_round_trips(self, original: object) -> None:
        pool = self._pool(1 << 20, 2)
        writer, reader = pool.open_writer(), pool.open_reader()
        registry = _default_registry()
        out = _restore(_offload({"p": original}, writer, registry), reader, registry)
        restored = cast(dict[str, Any], out)["p"]
        # Same concrete Packets type, and serializing again yields identical
        # bytes — a strong equivalence check (Packets have no __eq__).
        self.assertIs(type(restored), type(original))
        self.assertEqual(restored.__getstate__(), original.__getstate__())

    def test_video_packets_round_trip(self) -> None:
        """VideoPackets offload to / restore from the arena unchanged."""
        self._assert_round_trips(_demux("video"))

    def test_audio_packets_round_trip(self) -> None:
        """AudioPackets offload to / restore from the arena unchanged."""
        self._assert_round_trips(_demux("audio"))

    def test_image_packets_round_trip(self) -> None:
        """ImagePackets offload to / restore from the arena unchanged."""
        self._assert_round_trips(_demux("image"))

    def test_restored_view_decodes_like_original(self) -> None:
        """Decoding pool-restored (zero-copy view) packets matches the original."""
        packets = _demux("video")
        original = spdl.io.to_numpy(
            spdl.io.convert_frames(spdl.io.decode_packets(packets.clone()))
        )

        pool = self._pool(1 << 20, 2)
        writer, reader = pool.open_writer(), pool.open_reader()
        registry = _default_registry()
        out = _restore(_offload({"p": packets}, writer, registry), reader, registry)
        result = spdl.io.to_numpy(
            spdl.io.convert_frames(
                spdl.io.decode_packets(cast(dict[str, Any], out)["p"])
            )
        )

        np.testing.assert_array_equal(original, result, strict=True)
        del out, result
        gc.collect()

    def test_segment_held_until_packets_released(self) -> None:
        """The pool keeps a segment until the restored (view) packets are gone."""
        pool = self._pool(1 << 20, 2)
        writer, reader = pool.open_writer(), pool.open_reader()
        registry = _default_registry()
        out = _restore(
            _offload({"p": _demux("video")}, writer, registry), reader, registry
        )
        # The restored packets are a zero-copy view: the segment is still pinned.
        self.assertEqual(pool.reclaimed, 0)
        del out
        gc.collect()
        # Once the view is dropped, the segment returns to the pool.
        self.assertEqual(pool.reclaimed, 1)

    def test_packets_stay_inline_above_threshold(self) -> None:
        """With a threshold above the payload, Packets are pickled inline (via
        copyreg) rather than offloaded, and still round-trip correctly."""
        pool = self._pool(1 << 20, 2)
        writer, reader = pool.open_writer(), pool.open_reader()
        registry = _default_registry()
        packets = _demux("video")
        state = packets.__getstate__()
        # A threshold larger than the payload keeps it inline: the envelope then
        # carries the full serialized bytes itself, instead of a small marker.
        inline = _offload({"p": packets}, writer, registry, threshold=len(state) + 1)
        offloaded = _offload({"p": packets}, writer, registry, threshold=len(state))
        self.assertGreater(len(inline), len(state))
        self.assertLess(len(offloaded), len(inline))
        out = _restore(inline, reader, registry)
        self.assertEqual(cast(dict[str, Any], out)["p"].__getstate__(), state)


class IterateInSubprocessPacketsTest(unittest.TestCase):
    def test_packets_round_trip_through_subprocess(self) -> None:
        """Packets yielded by a subprocess iterator round-trip through the arena."""
        # Generate each sample once and keep the SrcInfo objects alive so their
        # temp dirs survive for the duration of the test. The serialized packet
        # state embeds the source path, so the subprocess worker and the local
        # ``expected`` must demux the *same* files for the bytes to match.
        media = ("audio", "video", "image")
        samples = [get_sample(CMDS[mt]) for mt in media]
        paths = [(mt, s.path) for mt, s in zip(media, samples)]
        fn = functools.partial(_demux_from_paths, paths)

        pool = SharedMemorySegmentPool(segment_size=1 << 20, count=4)
        self.addCleanup(pool.unlink)
        self.addCleanup(pool.close)
        got = list(iterate_in_subprocess(fn, arena=pool, buffer_size=2))

        expected = _demux_from_paths(paths)
        self.assertEqual(len(got), len(expected))
        for g, e in zip(got, expected):
            self.assertIs(type(g), type(e))
            self.assertEqual(g.__getstate__(), e.__getstate__())
