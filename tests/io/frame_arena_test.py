# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import concurrent.futures
import unittest

import numpy as np
import spdl.io

from ..fixture import FFMPEG_CLI, get_sample, SrcInfo


def _create_test_image() -> SrcInfo:
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc=size=320x240:rate=1 -frames:v 1 sample.jpg"
    return get_sample(cmd)


class TestFrameArena(unittest.TestCase):
    def test_decode_with_arena(self) -> None:
        """Decoding with arena produces valid output."""
        sample = _create_test_image()
        arena = spdl.io.frame_arena()

        packets = spdl.io.demux_image(sample.path)
        frames = spdl.io.decode_packets(packets, arena=arena)
        buffer = spdl.io.convert_frames(frames)
        result = spdl.io.to_numpy(buffer)

        self.assertEqual(result.ndim, 3)
        self.assertEqual(result.shape[0], 240)
        self.assertEqual(result.shape[1], 320)
        self.assertEqual(result.shape[2], 3)
        self.assertEqual(result.dtype, np.uint8)

    def test_decode_with_arena_matches_baseline(self) -> None:
        """Decoding with arena produces bit-exact output vs. without arena."""
        sample = _create_test_image()
        arena = spdl.io.frame_arena()

        packets_baseline = spdl.io.demux_image(sample.path)
        frames_baseline = spdl.io.decode_packets(packets_baseline)
        baseline = spdl.io.to_numpy(spdl.io.convert_frames(frames_baseline))

        packets_arena = spdl.io.demux_image(sample.path)
        frames_arena = spdl.io.decode_packets(packets_arena, arena=arena)
        with_arena = spdl.io.to_numpy(spdl.io.convert_frames(frames_arena))

        np.testing.assert_array_equal(baseline, with_arena)

    def test_arena_reuse_across_decodes(self) -> None:
        """Arena can be reused across multiple decode operations."""
        sample = _create_test_image()
        arena = spdl.io.frame_arena()

        for _ in range(5):
            packets = spdl.io.demux_image(sample.path)
            frames = spdl.io.decode_packets(packets, arena=arena)
            buffer = spdl.io.convert_frames(frames)
            result = spdl.io.to_numpy(buffer)
            self.assertEqual(result.shape, (240, 320, 3))

        self.assertGreater(arena.total_allocated, 0)

    def test_arena_total_allocated_stable(self) -> None:
        """After warmup, total_allocated should not grow for same-size images."""
        sample = _create_test_image()
        arena = spdl.io.frame_arena()

        # Warmup
        packets = spdl.io.demux_image(sample.path)
        frames = spdl.io.decode_packets(packets, arena=arena)
        spdl.io.convert_frames(frames)

        allocated_after_warmup = arena.total_allocated

        # Decode more images of same size
        for _ in range(10):
            packets = spdl.io.demux_image(sample.path)
            frames = spdl.io.decode_packets(packets, arena=arena)
            spdl.io.convert_frames(frames)

        self.assertEqual(arena.total_allocated, allocated_after_warmup)

    def test_arena_concurrent_decode(self) -> None:
        """Arena works correctly under concurrent decoding."""
        sample = _create_test_image()
        arena = spdl.io.frame_arena()

        def decode_one() -> np.ndarray:
            packets = spdl.io.demux_image(sample.path)
            frames = spdl.io.decode_packets(packets, arena=arena)
            buffer = spdl.io.convert_frames(frames)
            return spdl.io.to_numpy(buffer)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(decode_one) for _ in range(16)]
            results = [f.result() for f in futures]

        for result in results:
            self.assertEqual(result.shape, (240, 320, 3))
            self.assertEqual(result.dtype, np.uint8)

    def test_arena_properties(self) -> None:
        """Arena exposes total_allocated and total_pooled properties."""
        arena = spdl.io.frame_arena(
            initial_size=1024 * 1024,
            max_size=4 * 1024 * 1024,
        )
        self.assertEqual(arena.total_allocated, 1024 * 1024)
        self.assertGreaterEqual(arena.total_pooled, 0)

    def test_arena_concurrent_stress(self) -> None:
        """Stress test: many threads concurrently decoding mixed sizes via the same arena.

        Exercises the arena's thread-local free-lists, multi-bucket allocation,
        and concurrent slab growth under sustained load.
        """
        sizes = [(160, 120), (320, 240), (640, 480), (1280, 720)]
        samples = []
        for w, h in sizes:
            cmd = (
                f"{FFMPEG_CLI} -hide_banner -y -f lavfi "
                f"-i testsrc=size={w}x{h}:rate=1 -frames:v 1 sample_{w}x{h}.jpg"
            )
            samples.append(get_sample(cmd))

        arena = spdl.io.frame_arena()

        num_threads = 8
        decodes_per_thread = 50

        def worker(thread_idx: int) -> int:
            count = 0
            for i in range(decodes_per_thread):
                sample = samples[(thread_idx + i) % len(samples)]
                packets = spdl.io.demux_image(sample.path)
                frames = spdl.io.decode_packets(packets, arena=arena)
                buffer = spdl.io.convert_frames(frames)
                arr = spdl.io.to_numpy(buffer)
                self.assertEqual(arr.ndim, 3)
                count += 1
            return count

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_threads
        ) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            counts = [f.result() for f in futures]

        self.assertEqual(sum(counts), num_threads * decodes_per_thread)
        self.assertGreater(arena.total_allocated, 0)


if __name__ == "__main__":
    unittest.main()
