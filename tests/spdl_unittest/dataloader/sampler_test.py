# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
from functools import partial

import numpy as np
import pytest
from spdl.pipeline import iterate_in_subprocess
from spdl.source import (
    DistributedDeterministicSampler,
    DistributedRandomSampler,
    SizedIterable,
    SizedIterableWithShuffle,
)
from spdl.source.utils import embed_shuffle

# pyre-unsafe


def testDistributedsampler_interface():
    """samplers conform to Iterable/IterableWithShuffle protocol"""
    assert isinstance(
        DistributedRandomSampler(9, rank=0, world_size=1), SizedIterableWithShuffle
    )
    assert isinstance(
        DistributedDeterministicSampler(9, rank=0, world_size=1), SizedIterable
    )


def testDistributedsamplerdeterministic_iter():
    """without distributed, deterministic iteration behaves same as `range(N)`"""
    N = 30
    sampler = DistributedDeterministicSampler(N, rank=0, world_size=1)
    assert len(sampler) == N
    assert list(sampler) == list(range(N))


def testDistributedsamplerdeterministic_iterDistributed():
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
            assert len(sampler) == len_

            indices = list(sampler)
            assert indices == list(range(rank, max_, world_size))
            c.update(indices)

        # Check that together, the samplers covered the whole dataset
        num_iters = N // world_size * world_size
        assert c.total() == num_iters
        assert len(c.keys()) == num_iters
        assert set(c.keys()) == set(range(num_iters))
        assert all(v == 1 for v in c.values())


def testDistributedsampler_shuffle():
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
        assert indices != previous
        previous = indices


@pytest.mark.parametrize("w", [None, 1])
def testDistributedsampler_repeat(w):
    """Without calling shuffle, sampler generates the same sequence."""
    N = 40
    world_size = 8

    weights = None if w is None else [1] * N
    for rank in range(world_size):
        previous = []
        for i in range(100):
            sampler = DistributedRandomSampler(
                N, rank=rank, world_size=world_size, weights=weights
            )

            indices = list(sampler)
            print(f"{indices=}")
            if i > 0:
                assert indices == previous
            previous = indices


@pytest.mark.parametrize("shuffle", [True, False])
def testDistributedsampler_mutual_exclusive(shuffle):
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

        assert c.total() == N
        assert len(c.keys()) == N
        assert set(c.keys()) == set(range(N))
        assert all(v == 1 for v in c.values())


@pytest.mark.parametrize("shuffle", [True, False])
def testDistributedsampler_mutual_exclusive_num_draws(shuffle):
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
        assert c.total() == m
        assert len(c.keys()) == m
        assert all(v == 1 for v in c.values())


def testDistributedsampler_weighted_sampling():
    """Indices are drawn according to the weights"""
    weights = [0, 1, 3, 5, 10]
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

    assert np.allclose(hyp, ref, atol=1e-3)


def testDistributedsampler_embed_shuffle():
    """DistributedSampler is compatibile with embed_shuffle"""
    N = 10
    weights = [1 for _ in range(N)]

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

        assert hyp == ref
        assert hyp != previous
        previous = hyp


def testDistributedsampler_iterate_in_subprocess():
    """Iterating in a subprocess generates identical result"""
    N = 10
    weights = [1 for _ in range(N)]

    sampler = DistributedRandomSampler(N, rank=0, world_size=1, weights=weights)
    sampler_sub = iterate_in_subprocess(partial(embed_shuffle, sampler))
    sampler = embed_shuffle(sampler)

    previous = []
    for _ in range(100):
        hyp = list(sampler_sub)
        print(f"{hyp=}")
        ref = list(sampler)
        print(f"{ref=}")

        assert hyp == ref
        assert hyp != previous
        previous = hyp
