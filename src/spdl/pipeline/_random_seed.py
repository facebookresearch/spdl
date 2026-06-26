# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Copy the main process' global RNG state into a worker subprocess.

When a pipeline runs in a worker subprocess (created by
:py:func:`~spdl.pipeline.run_pipeline_in_subprocess`, including the threaded
"MTP" arrangement), RNG-dependent work inside the pipeline (e.g. data
augmentation that calls ``random`` / ``numpy`` / ``torch``) would otherwise
behave differently than in single-process multi-threading:

* With the ``fork`` start method the subprocess inherits a *copy* of the
  parent's RNG state, but only of the forking thread's view at fork time.
* With ``spawn`` / ``forkserver`` the subprocess seeds its RNGs from OS entropy,
  unrelated to the parent.

So if the user seeds the global RNGs in the main process (``random.seed(...)``,
``numpy.random.seed(...)``, ``torch.manual_seed(...)``) and then runs the
pipeline in a subprocess, that seeding would be lost or inconsistent.

This module captures the global RNG state in the main process at build time and
restores it in the subprocess before iteration, so the subprocess continues from
exactly the state the main process was in -- seamlessly, regardless of the start
method and with no user opt-in.

Only RNG libraries the program has **already imported** are touched: the core
``spdl.pipeline`` package is pure Python and must not import (or depend on)
``numpy`` / ``torch``. State is captured per library, gated on
:py:data:`sys.modules`, and a per-library restore initializer is shipped to the
subprocess only when that library is in use. Unpickling a captured ``numpy`` /
``torch`` state object re-imports the library in the subprocess, so it is
guaranteed available when its restore initializer runs.

SPDL's own samplers (:py:class:`~spdl.source.DistributedRandomSampler` and
friends) build a private :py:class:`numpy.random.Generator` from their stored
seed on each iteration, so they are independent of this global state and already
produce identical sequences in every mode.

This copies the **global** generators only. A :py:class:`numpy.random.Generator`
from :py:func:`numpy.random.default_rng` is an independent object and is not
affected; ``torch`` CUDA device RNG state is not copied (only the CPU
generator); and hash randomization (``PYTHONHASHSEED``) / entropy sources
(``os.urandom``, ``secrets``, ``uuid.uuid4``) are out of reach. The user-facing
contract is documented on :py:func:`~spdl.pipeline.run_pipeline_in_subprocess`.

When the worker is a **subinterpreter** rather than a subprocess (via
:py:func:`~spdl.pipeline.run_pipeline_in_subinterpreter`), the restore
initializers are passed as a call argument and so must be *shareable* across
interpreters. Only the stdlib ``random`` state qualifies -- it is a plain tuple
-- so capture is restricted to it via ``shareable_only``. The ``numpy`` / ``torch``
state objects are not shareable, and those libraries cannot even be imported in a
subinterpreter, so they are necessarily left out of that path.
"""

import random
import sys
from collections.abc import Callable
from functools import partial
from typing import Any

__all__ = [
    "_capture_rng_initializers",
]


def _imported(name: str) -> Any:
    """Return the module ``name`` if the program already imported it, else ``None``.

    Reads :py:data:`sys.modules` without importing, so a library the program
    never loaded is left untouched (and no dependency is introduced).
    """
    return sys.modules.get(name)


def _restore_random_state(state: object) -> None:
    """Restore the stdlib :py:mod:`random` global state (runs in the subprocess)."""
    random.setstate(state)  # pyre-ignore[6]: round-tripped from random.getstate()


def _restore_numpy_state(state: object) -> None:
    """Restore NumPy's legacy global RNG state (runs in the subprocess).

    ``numpy`` is guaranteed importable here: unpickling ``state`` (which holds a
    NumPy array) already imported it into :py:data:`sys.modules`.
    """
    _imported("numpy").random.set_state(state)


def _restore_torch_state(state: object) -> None:
    """Restore PyTorch's CPU RNG state (runs in the subprocess).

    ``torch`` is guaranteed importable here: unpickling ``state`` (a tensor)
    already imported it into :py:data:`sys.modules`.
    """
    _imported("torch").set_rng_state(state)


def _capture_rng_initializers(
    *, shareable_only: bool = False
) -> list[Callable[[], None]]:
    """Capture the current global RNG state of every already-imported RNG library.

    Runs in the main process. Returns a list of zero-argument initializers, each
    binding a captured state, that restore that exact state when run in the
    worker. ``numpy`` and ``torch`` are included only when the program has already
    imported them; stdlib ``random`` is always included.

    When ``shareable_only`` is set, only the stdlib ``random`` initializer is
    returned. Its captured state is a plain tuple of ints, which is shareable
    across subinterpreters; the ``numpy`` / ``torch`` state objects (a NumPy array
    and a tensor) are not, so they cannot ride along the subinterpreter call
    boundary and are dropped.
    """
    initializers: list[Callable[[], None]] = [
        partial(_restore_random_state, random.getstate()),
    ]
    if shareable_only:
        return initializers

    if (np := _imported("numpy")) is not None:
        initializers.append(partial(_restore_numpy_state, np.random.get_state()))

    if (torch := _imported("torch")) is not None:
        initializers.append(partial(_restore_torch_state, torch.get_rng_state()))

    return initializers
