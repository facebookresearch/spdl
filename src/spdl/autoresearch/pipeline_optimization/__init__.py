# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SPDL pipeline optimization implementation of autoresearch.

Provides the complete workflow for automatically optimizing SPDL data
loading pipelines. Two public entry points:

- :py:func:`create_workflow` is a
  :py:data:`~spdl.autoresearch.core.WorkflowFactory`. Pass
  ``--workflow spdl.autoresearch.pipeline_optimization:create_workflow``
  to the framework dispatcher (or rely on the bundled binary's default).
- :py:func:`main` is the legacy supervisor-CLI entry point that the
  ``autoresearch`` Buck binary currently dispatches to. Subsequent
  refactor steps replace this with the framework dispatcher.
"""

from ._cli import main
from ._factory import create_workflow

__all__ = ["create_workflow", "main"]
