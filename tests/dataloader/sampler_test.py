# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import functools
import multiprocessing as mp
import pickle
import random
import unittest
import warnings
from collections import Counter
from collections.abc import Callable, Iterable
from functools import partial
from typing import TypeVar

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

_F = TypeVar("_F", bound=Callable[..., object])


def _ignore_fork_warning(fn: _F) -> _F:
    @functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> object:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"This process \(pid=\d+\) is multi-threaded, use of "
                    r"fork\(\) may lead to deadlocks in the child"
                ),
                category=DeprecationWarning,
            )
            return fn(*args, **kwargs)

    # pyre-ignore[7]
    return wrapper


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
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"The size of dataset \(\d+\) is not divisible by the "
                    r"world size \(\d+\)\. Some samples are never visited\."
                ),
                category=UserWarning,
            )
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

    def test_deterministic_iter_stable_across_epochs(self) -> None:
        """Deterministic sampler produces the same sequence on every iteration."""
        N = 30
        world_size = 4
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"The size of dataset \(\d+\) is not divisible by the "
                    r"world size \(\d+\)\. Some samples are never visited\."
                ),
                category=UserWarning,
            )
            for rank in range(world_size):
                sampler = DistributedDeterministicSampler(
                    N, rank=rank, world_size=world_size
                )
                first_epoch = list(sampler)
                for _ in range(5):
                    self.assertEqual(list(sampler), first_epoch)

    def test_deterministic_keep_tail_covers_all_indices(self) -> None:
        """Retaining the tail covers each deterministic index exactly once."""
        N = 26
        world_size = 4
        all_indices: list[int] = []
        per_rank_counts: list[int] = []
        for rank in range(world_size):
            sampler = DistributedDeterministicSampler(
                N,
                rank=rank,
                world_size=world_size,
                ddp_drop_last_distributed_round=False,
            )
            indices = list(sampler)
            self.assertEqual(indices, list(range(rank, N, world_size)))
            self.assertEqual(len(sampler), len(indices))
            per_rank_counts.append(len(indices))
            all_indices.extend(indices)

        self.assertEqual(per_rank_counts, [7, 7, 6, 6])
        self.assertEqual(sorted(all_indices), list(range(N)))
        self.assertEqual(len(all_indices), len(set(all_indices)))

    def test_deterministic_default_drops_tail(self) -> None:
        """Default deterministic sampler preserves floor/drop tail behavior."""
        N = 26
        world_size = 4
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            all_indices = [
                idx
                for rank in range(world_size)
                for idx in DistributedDeterministicSampler(
                    N, rank=rank, world_size=world_size
                )
            ]

        self.assertEqual(len(all_indices), 24)
        self.assertEqual(sorted(all_indices), list(range(24)))
        self.assertNotIn(24, all_indices)
        self.assertNotIn(25, all_indices)

    def test_deterministic_keep_tail_allows_fewer_samples_than_ranks(
        self,
    ) -> None:
        """Retaining the tail permits datasets smaller than world size."""
        N = 3
        world_size = 5
        all_indices: list[int] = []
        per_rank_counts: list[int] = []
        for rank in range(world_size):
            sampler = DistributedDeterministicSampler(
                N,
                rank=rank,
                world_size=world_size,
                ddp_drop_last_distributed_round=False,
            )
            indices = list(sampler)
            self.assertEqual(indices, list(range(rank, N, world_size)))
            self.assertEqual(len(sampler), len(indices))
            per_rank_counts.append(len(indices))
            all_indices.extend(indices)

        self.assertEqual(per_rank_counts, [1, 1, 1, 0, 0])
        self.assertEqual(sorted(all_indices), list(range(N)))
        self.assertEqual(len(all_indices), len(set(all_indices)))

    def test_deterministic_keep_tail_suppresses_tail_warning(self) -> None:
        """Retaining the tail does not warn about unvisited tail samples."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            DistributedDeterministicSampler(
                26, rank=0, world_size=4, ddp_drop_last_distributed_round=False
            )

        messages = [str(w.message) for w in caught]
        self.assertFalse(
            any("Some samples are never visited" in message for message in messages)
        )


class TestDistributedSamplerRandom(unittest.TestCase):
    def test_shuffle(self) -> None:
        """shuffling makes sampler generates different indices."""
        N = 640
        rank = 3
        world_size = 8

        previous: list[int] = []
        for epoch in range(100):
            sampler = DistributedRandomSampler(N, rank=rank, world_size=world_size)
            sampler.shuffle(seed=epoch)

            indices = list(sampler)
            print(f"{indices=}")
            self.assertNotEqual(indices, previous)
            previous = indices

    def test_shuffle_epoch_loop(self) -> None:
        """shuffle(seed=epoch) produces different sequences each epoch."""
        N = 640
        world_size = 8

        for rank in range(world_size):
            sampler = DistributedRandomSampler(N, rank=rank, world_size=world_size)
            previous: list[int] = []
            for epoch in range(10):
                sampler.shuffle(seed=epoch)
                indices = list(sampler)
                self.assertEqual(len(indices), N // world_size)
                self.assertNotEqual(indices, previous)
                previous = indices

    def test_shuffle_is_stateless(self) -> None:
        """shuffle(seed) output depends only on the seed, not on prior iteration history."""
        N = 640
        world_size = 8

        for rank in range(world_size):
            sampler = DistributedRandomSampler(N, rank=rank, world_size=world_size)
            for i in range(10):
                sampler.shuffle(seed=i)
                list(sampler)
            sampler.shuffle(seed=5)
            after_many = list(sampler)

            fresh = DistributedRandomSampler(N, rank=rank, world_size=world_size)
            fresh.shuffle(seed=5)
            from_fresh = list(fresh)

            self.assertEqual(after_many, from_fresh)

    def test_shuffle_epoch_loop_mutual_exclusive(self) -> None:
        """All ranks together cover the full dataset for each epoch seed."""
        N = 640
        world_size = 8

        samplers = [
            DistributedRandomSampler(N, rank=rank, world_size=world_size)
            for rank in range(world_size)
        ]

        for epoch in range(10):
            c = Counter()
            for sampler in samplers:
                sampler.shuffle(seed=epoch)
                c.update(sampler)

            self.assertEqual(c.total(), N)
            self.assertEqual(len(c.keys()), N)
            self.assertEqual(set(c.keys()), set(range(N)))
            self.assertTrue(all(v == 1 for v in c.values()))

    @parameterized.expand(
        [
            (None,),
            (1,),
        ]
    )
    def test_repeat(self, w: int | None) -> None:
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

    def test_keep_tail_uses_all_draws(self) -> None:
        """Retaining the tail preserves the full random draw count."""
        N = 26
        world_size = 4
        c = Counter()
        per_rank_counts: list[int] = []
        for rank in range(world_size):
            sampler = DistributedRandomSampler(
                N,
                rank=rank,
                world_size=world_size,
                ddp_drop_last_distributed_round=False,
            )
            indices = list(sampler)
            self.assertEqual(len(sampler), len(indices))
            per_rank_counts.append(len(indices))
            c.update(indices)

        self.assertEqual(per_rank_counts, [7, 7, 6, 6])
        self.assertEqual(c.total(), N)
        self.assertEqual(len(c.keys()), N)
        self.assertEqual(set(c.keys()), set(range(N)))
        self.assertTrue(all(v == 1 for v in c.values()))

    def test_keep_tail_allows_fewer_samples_than_ranks(self) -> None:
        """Retaining the tail permits datasets smaller than world size."""
        N = 3
        world_size = 5
        c = Counter()
        per_rank_counts: list[int] = []
        for rank in range(world_size):
            sampler = DistributedRandomSampler(
                N,
                rank=rank,
                world_size=world_size,
                ddp_drop_last_distributed_round=False,
            )
            indices = list(sampler)
            self.assertEqual(len(sampler), len(indices))
            per_rank_counts.append(len(indices))
            c.update(indices)

        self.assertEqual(per_rank_counts, [1, 1, 1, 0, 0])
        self.assertEqual(c.total(), N)
        self.assertEqual(len(c.keys()), N)
        self.assertEqual(set(c.keys()), set(range(N)))
        self.assertTrue(all(v == 1 for v in c.values()))

    def test_default_random_sampler_drops_tail(self) -> None:
        """Default random sampler preserves floor/drop tail behavior."""
        N = 26
        world_size = 4
        c = Counter()
        for rank in range(world_size):
            sampler = DistributedRandomSampler(N, rank=rank, world_size=world_size)
            self.assertEqual(len(sampler), N // world_size)
            c.update(sampler)

        self.assertEqual(c.total(), 24)
        self.assertEqual(len(c.keys()), 24)
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

    def test_keep_tail_allows_fewer_draws_than_ranks(self) -> None:
        """Retaining the tail permits draw counts smaller than world size."""
        N = 10
        num_draws = 3
        world_size = 5
        c = Counter()
        per_rank_counts: list[int] = []
        for rank in range(world_size):
            sampler = DistributedRandomSampler(
                N,
                rank=rank,
                world_size=world_size,
                num_draws=num_draws,
                ddp_drop_last_distributed_round=False,
            )
            indices = list(sampler)
            self.assertEqual(len(sampler), len(indices))
            per_rank_counts.append(len(indices))
            c.update(indices)

        self.assertEqual(per_rank_counts, [1, 1, 1, 0, 0])
        self.assertEqual(c.total(), num_draws)
        self.assertEqual(len(c.keys()), num_draws)
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
    @_ignore_fork_warning
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


# ---------------------------------------------------------------------------
# Sampler contract: identical output is preserved across the subprocess
# boundary, *regardless of how* the sampler implements its determinism.
#
# Today the cross-process equivalence happens to fall out of NumPy's seeded
# ``Generator`` being a pure function of the stored seed -- but that is an
# implementation detail. These tests pin the observable contract directly so
# that an implementation change which breaks it (e.g. switching to a global RNG,
# or capturing process-local / non-picklable state) is caught regardless of how
# determinism is achieved.
# ---------------------------------------------------------------------------
_SAMPLER_N = 24
_SAMPLER_SEED = 7
_TIMEOUT = 60.0


def _build_random_sampler() -> Iterable[int]:
    """A plain (unshuffled) random sampler -- same sequence every epoch."""
    return DistributedRandomSampler(
        _SAMPLER_N, rank=0, world_size=1, seed=_SAMPLER_SEED
    )


def _build_weighted_sampler() -> Iterable[int]:
    """A weighted random sampler (draws with replacement)."""
    return DistributedRandomSampler(
        _SAMPLER_N,
        rank=0,
        world_size=1,
        weights=[1.0] * _SAMPLER_N,
        num_draws=_SAMPLER_N,
        seed=_SAMPLER_SEED,
    )


def _build_deterministic_sampler() -> Iterable[int]:
    """A deterministic (round-robin) sampler -- no randomness at all."""
    return DistributedDeterministicSampler(_SAMPLER_N, rank=0, world_size=1)


def _build_shuffled_sampler() -> Iterable[int]:
    """A random sampler that reshuffles itself each epoch via ``embed_shuffle``."""
    return embed_shuffle(
        DistributedRandomSampler(_SAMPLER_N, rank=0, world_size=1, seed=_SAMPLER_SEED)
    )


_BUILDERS: dict[str, Callable[[], Iterable[int]]] = {
    "random": _build_random_sampler,
    "weighted": _build_weighted_sampler,
    "deterministic": _build_deterministic_sampler,
    "shuffled": _build_shuffled_sampler,
}


def _collect_epochs(source: Iterable[int], num_epochs: int) -> list[list[int]]:
    """Iterate ``source`` ``num_epochs`` times, returning one index list per epoch."""
    return [list(source) for _ in range(num_epochs)]


def _perturb_global_rngs() -> None:
    """Advance the global RNGs to a state unrelated to any sampler seed.

    A sampler whose output depends only on its own (seeded) state must be
    completely unaffected by this. Only ``numpy`` (the legacy global RNG) and
    stdlib ``random`` are perturbed -- those are the global generators a sampler
    could plausibly start drawing from.
    """
    random.seed(0x0BADC0DE)
    [random.random() for _ in range(64)]
    # numpy's stub types ``seed`` as a sequence/array; pass a single-element seq.
    np.random.seed([0x1234])
    np.random.rand(64)


def _available_start_methods() -> list[str]:
    return [m for m in ("fork", "spawn") if m in mp.get_all_start_methods()]


class TestDistributedSamplerSubprocessContract(unittest.TestCase):
    @parameterized.expand([("random",), ("weighted",), ("deterministic",)])
    def test_output_invariant_to_global_rng_state(self, builder_name: str) -> None:
        """A sampler's sequence does not depend on ambient global RNG state.

        Catches a regression where the sampler starts drawing from a global RNG
        (``random`` / ``numpy``) -- which would also silently diverge across a
        process boundary, since the global state differs there."""
        builder = _BUILDERS[builder_name]
        reference = list(builder())

        # Perturbed global RNGs must not change a freshly built sampler's output.
        _perturb_global_rngs()
        self.assertEqual(list(builder()), reference, msg=f"fresh build {builder_name}")

        # Nor perturbation between construction and iteration of the same object.
        sampler = builder()
        _perturb_global_rngs()
        self.assertEqual(list(sampler), reference, msg=f"same object {builder_name}")

    def test_shuffle_progression_invariant_to_global_rng_state(self) -> None:
        """Per-epoch reshuffle (``embed_shuffle``) depends only on the epoch
        counter, not on ambient global RNG state."""
        num_epochs = 4
        reference = _collect_epochs(_build_shuffled_sampler(), num_epochs)
        # The reshuffle must reorder across epochs (otherwise the test is vacuous).
        self.assertNotEqual(reference[0], reference[1])

        src = _build_shuffled_sampler()
        got = []
        for _ in range(num_epochs):
            _perturb_global_rngs()
            got.append(list(src))
        self.assertEqual(got, reference)

    @parameterized.expand(
        [
            (builder_name, num_epochs)
            for builder_name in ("random", "weighted", "deterministic", "shuffled")
            for num_epochs in (1, 3)
        ]
    )
    def test_output_survives_pickle_roundtrip(
        self, builder_name: str, num_epochs: int
    ) -> None:
        """Pickling then unpickling a sampler -- exactly what ``spawn`` /
        ``forkserver`` do to move it into a worker -- preserves its full
        multi-epoch sequence. Catches reliance on process-local or non-picklable
        state."""
        builder = _BUILDERS[builder_name]
        reference = _collect_epochs(builder(), num_epochs)

        restored = pickle.loads(pickle.dumps(builder()))
        got = _collect_epochs(restored, num_epochs)
        self.assertEqual(
            got, reference, msg=f"builder={builder_name} num_epochs={num_epochs}"
        )

    @parameterized.expand(
        [
            (builder_name, start_method, num_epochs)
            for builder_name in ("random", "weighted", "deterministic", "shuffled")
            for start_method in _available_start_methods()
            for num_epochs in (1, 3)
        ]
    )
    @_ignore_fork_warning
    def test_sampler_matches_in_subprocess(
        self, builder_name: str, start_method: str, num_epochs: int
    ) -> None:
        """Iterating a sampler in a subprocess yields the identical per-epoch
        sequence as iterating it in-process, for both ``fork`` and ``spawn`` and
        for single and multiple epochs."""
        builder = _BUILDERS[builder_name]
        reference = _collect_epochs(builder(), num_epochs)

        src = iterate_in_subprocess(builder, mp_context=start_method, timeout=_TIMEOUT)
        try:
            got = _collect_epochs(src, num_epochs)
        finally:
            finalizer = getattr(src, "_finalizer", None)
            if finalizer is not None:
                finalizer()
        self.assertEqual(
            got,
            reference,
            msg=f"builder={builder_name} start_method={start_method} "
            f"num_epochs={num_epochs}",
        )
