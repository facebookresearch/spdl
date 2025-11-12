# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from collections import Counter
from functools import partial
from typing import Optional

import numpy as np
from parameterized import parameterized
from spdl.pipeline import iterate_in_subprocess
from spdl.source import (
    DistributedDeterministicSampler,
    DistributedRandomSampler,
    SizedIterable,
    SizedIterableWithShuffle,
)
from spdl.source.utils import embed_shuffle

# pyre-unsafe


class TestDistributedSamplerInterface(unittest.TestCase):
    def test_distributed_sampler_interface(self) -> None:
        """samplers conform to Iterable/IterableWithShuffle protocol"""
        self.assertIsInstance(
            DistributedRandomSampler(9, rank=0, world_size=1), SizedIterableWithShuffle
        )
        self.assertIsInstance(
            DistributedDeterministicSampler(9, rank=0, world_size=1), SizedIterable
        )


class TestDistributedSamplerDeterministic(unittest.TestCase):
    def test_deterministic_iter(self) -> None:
        """without distributed, deterministic iteration behaves same as `range(N)`"""
        N = 30
        sampler = DistributedDeterministicSampler(N, rank=0, world_size=1)
        self.assertEqual(len(sampler), N)
        self.assertEqual(list(sampler), list(range(N)))

    def test_deterministic_iter_distributed(self) -> None:
        """deterministic iteration behaves same as `range(rank, M, world_size)`"""
        N = 26
        for world_size in range(1, N + 1):
            len_ = N // world_size
            max_ = len_ * world_size
            c = Counter()
            for rank in range(world_size):
                print(f"{N=}, {world_size=}, {rank=}, {len_=}, {max_=}")
                sampler = DistributedDeterministicSampler(
                    N, rank=rank, world_size=world_size
                )
                self.assertEqual(len(sampler), len_)

                indices = list(sampler)
                self.assertEqual(indices, list(range(rank, max_, world_size)))
                c.update(indices)

            # Check that together, the samplers covered the whole dataset
            num_iters = N // world_size * world_size
            self.assertEqual(c.total(), num_iters)
            self.assertEqual(len(c.keys()), num_iters)
            self.assertEqual(set(c.keys()), set(range(num_iters)))
            self.assertTrue(all(v == 1 for v in c.values()))


class TestDistributedSamplerRandom(unittest.TestCase):
    def test_shuffle(self) -> None:
        """shuffling makes sampler generates different indices."""
        N = 640
        rank = 3
        world_size = 8

        previous = []
        for epoch in range(100):
            sampler = DistributedRandomSampler(N, rank=rank, world_size=world_size)
            sampler.shuffle(seed=epoch)

            indices = list(sampler)
            print(f"{indices=}")
            self.assertNotEqual(indices, previous)
            previous = indices

    @parameterized.expand(
        [
            (None,),
            (1,),
        ]
    )
    def test_repeat(self, w: Optional[int]) -> None:
        """Without calling shuffle, sampler generates the same sequence."""
        N = 40
        world_size = 8

        weights = None if w is None else [1.0] * N
        for rank in range(world_size):
            previous = []
            for i in range(100):
                sampler = DistributedRandomSampler(
                    N, rank=rank, world_size=world_size, weights=weights
                )

                indices = list(sampler)
                print(f"{indices=}")
                if i > 0:
                    self.assertEqual(indices, previous)
                previous = indices

    @parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    def test_mutual_exclusive(self, shuffle: bool) -> None:
        """Without weights, samplers generate mutually exclusive sets"""
        N = 640
        world_size = 8

        for epoch in range(100):
            c = Counter()
            for rank in range(world_size):
                sampler = DistributedRandomSampler(N, rank=rank, world_size=world_size)
                if shuffle:
                    sampler.shuffle(seed=epoch)
                c.update(sampler)

            self.assertEqual(c.total(), N)
            self.assertEqual(len(c.keys()), N)
            self.assertEqual(set(c.keys()), set(range(N)))
            self.assertTrue(all(v == 1 for v in c.values()))

    @parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    def test_mutual_exclusive_num_draws(self, shuffle: bool) -> None:
        """Without weights, samplers generate mutually exclusive sets"""
        N = 640
        num_draws = 321
        world_size = 8

        for epoch in range(100):
            c = Counter()
            for rank in range(world_size):
                sampler = DistributedRandomSampler(
                    N, rank=rank, world_size=world_size, num_draws=num_draws
                )
                if shuffle:
                    sampler.shuffle(seed=epoch)
                c.update(sampler)

            m = num_draws // world_size * world_size
            self.assertEqual(c.total(), m)
            self.assertEqual(len(c.keys()), m)
            self.assertTrue(all(v == 1 for v in c.values()))


class TestDistributedSamplerWeighted(unittest.TestCase):
    def test_weighted_sampling(self) -> None:
        """Indices are drawn according to the weights"""
        weights = [0.0, 1.0, 3.0, 5.0, 10.0]
        N = len(weights)

        sampler = DistributedRandomSampler(
            N, rank=0, world_size=1, num_draws=1_000_000, weights=weights
        )

        c = Counter(sampler)
        distribution = [c[i] for i in range(N)]

        print(f"{weights=}")
        print(f"{distribution=}")

        ref = np.asarray(weights) / np.sum(weights)
        hyp = np.asarray(distribution) / np.sum(distribution)

        print(f"{ref=}")
        print(f"{hyp=}")

        self.assertTrue(np.allclose(hyp, ref, atol=1e-3))


class TestDistributedSamplerEmbedShuffle(unittest.TestCase):
    def test_embed_shuffle(self) -> None:
        """DistributedSampler is compatibile with embed_shuffle"""
        N = 10
        weights = [1.0 for _ in range(N)]

        s0 = DistributedRandomSampler(N, rank=0, world_size=1, weights=weights)
        s1 = DistributedRandomSampler(N, rank=0, world_size=1, weights=weights)

        s1 = embed_shuffle(s1)

        previous = []
        for i in range(100):
            hyp = list(s1)
            print(f"{hyp=}")

            s0.shuffle(i)
            ref = list(s0)
            print(f"{ref=}")

            self.assertEqual(hyp, ref)
            self.assertNotEqual(hyp, previous)
            previous = hyp


class TestDistributedSamplerIterateInSubprocess(unittest.TestCase):
    def test_iterate_in_subprocess(self) -> None:
        """Iterating in a subprocess generates identical result"""
        N = 10
        weights = [1.0 for _ in range(N)]

        sampler = DistributedRandomSampler(N, rank=0, world_size=1, weights=weights)
        sampler_sub = iterate_in_subprocess(partial(embed_shuffle, sampler))
        sampler = embed_shuffle(sampler)

        previous = []
        for _ in range(100):
            hyp = list(sampler_sub)
            print(f"{hyp=}")
            ref = list(sampler)
            print(f"{ref=}")

            self.assertEqual(hyp, ref)
            self.assertNotEqual(hyp, previous)
            previous = hyp
