# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Deterministic seeding of global RNGs for subprocess-based execution.

When a pipeline runs in a worker subprocess (created by
:py:func:`~spdl.pipeline.run_pipeline_in_subprocess`, including the threaded
"MTP" arrangement), the global RNGs of ``random`` / ``numpy`` / ``torch`` would
otherwise diverge from single-process multi-threading execution:

* With the ``fork`` start method the subprocess inherits a *copy* of the parent's
  RNG state, so its draws happen to continue the parent's stream.
* With ``spawn`` / ``forkserver`` the subprocess seeds its RNGs from OS entropy,
  so results are not reproducible across program runs.

To make execution consistent without any user opt-in, the subprocess boundary
seeds the global RNGs deterministically from a base seed captured in the main
process. A given base seed always yields the same streams (reproducible). The
base seed is drawn from the main process' stdlib ``random`` module, so seeding
the main process (e.g. ``random.seed(k)``) makes the subprocess draws
reproducible across program runs as well -- the same contract as
:py:class:`torch.utils.data.DataLoader`.

``numpy`` and ``torch`` are reseeded only when they are **already imported** by
the running program: the core ``spdl.pipeline`` package is pure Python and must
not import (let alone depend on) them, and reseeding a library the user never
loaded would be pointless. SPDL's own samplers
(:py:class:`~spdl.source.DistributedRandomSampler` and friends) build a private
:py:class:`numpy.random.Generator` from their stored seed on each iteration, so
they are unaffected and already produce identical sequences in every mode.

This deliberately covers only the **global** RNGs. In particular,
``numpy.random.seed`` reseeds NumPy's *legacy* global ``RandomState`` (the one
behind ``numpy.random.rand`` etc.) but has no effect on a
``numpy.random.Generator`` returned by ``numpy.random.default_rng()`` — those are
independent objects the library does not own, so user code holding an unseeded
one must seed it itself. Likewise, hash randomization (``PYTHONHASHSEED``) and
entropy sources (``os.urandom``, ``secrets``, ``uuid.uuid4``) are out of reach.
The user-facing contract is documented on
:py:func:`~spdl.pipeline.run_pipeline_in_subprocess`.
"""

import logging
import random
import sys
from types import ModuleType

__all__ = [
    "_draw_base_seed",
    "_seed_global_rngs",
]

_LG: logging.Logger = logging.getLogger(__name__)

_MASK64: int = (1 << 64) - 1
# FNV-1a 64-bit constants.
_FNV_OFFSET: int = 0xCBF29CE484222325
_FNV_PRIME: int = 0x100000001B3


def _mix(*values: int) -> int:
    """Combine integers into a 64-bit seed (order-sensitive, well-dispersed, numpy-free)."""
    h = _FNV_OFFSET
    for value in values:
        v = value & _MASK64
        for _ in range(8):
            h ^= v & 0xFF
            h = (h * _FNV_PRIME) & _MASK64
            v >>= 8
    return h


def _imported(name: str) -> ModuleType | None:
    """Return the module ``name`` only if user code already imported it, else ``None``.

    Reads ``sys.modules`` without importing, so a library the program never
    loaded is left untouched (and no hard dependency is introduced).
    """
    return sys.modules.get(name)


def _draw_base_seed() -> int:
    """Draw a 64-bit base seed from the main process' stdlib ``random`` global.

    Reproducible across program runs only if the main process seeds ``random``
    (e.g. ``random.seed(k)``); otherwise it varies per run, exactly like the base
    seed of :py:class:`torch.utils.data.DataLoader`.
    """
    return random.getrandbits(64)


def _seed_global_rngs(base_seed: int) -> None:
    """Deterministically seed the global RNGs of ``random`` and any imported RNG library.

    ``random`` is always seeded. ``numpy`` and ``torch`` are seeded only when they
    are already present in :py:data:`sys.modules` (i.e. the program imported them);
    they are never imported by this function.
    """
    random.seed(_mix(base_seed, 0))

    if (np := _imported("numpy")) is not None:
        # ``numpy.random.seed`` accepts a 32-bit value.
        np.random.seed(_mix(base_seed, 1) & 0xFFFFFFFF)

    if (torch := _imported("torch")) is not None:
        torch.manual_seed(_mix(base_seed, 2))
