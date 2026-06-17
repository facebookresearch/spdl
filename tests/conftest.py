# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test-suite diagnostics.

These tests spawn subprocesses and shut down pipelines, which has historically
hung on Windows (no ``fork()``, different shared-memory and process-termination
semantics, no POSIX signals). When that happens in CI the job stalls until the
global timeout with no clue as to *where* it is stuck.

``faulthandler.dump_traceback_later`` periodically prints the stack of every
thread, so a hang leaves a trail of tracebacks pointing at the exact blocking
call (a ``Queue.get``, ``Process.join``, pipe read, lock acquire, ...). It is
armed on Windows, where the hangs occur; the periodic dump is harmless
elsewhere but only adds noise, so we keep it scoped.
"""

import faulthandler
import sys

# Dump tracebacks on fatal errors (segfaults etc.) everywhere.
faulthandler.enable(file=sys.stderr)

if sys.platform == "win32":
    # Dump all thread stacks every 90s while a test is running. The per-test
    # pytest-timeout (set on the CI command line) fires after this, so a hung
    # test produces at least one full stack dump before it is killed.
    faulthandler.dump_traceback_later(90, repeat=True, file=sys.stderr)
