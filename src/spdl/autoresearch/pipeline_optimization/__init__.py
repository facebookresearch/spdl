# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SPDL pipeline optimization implementation of autoresearch.

Provides the complete workflow for automatically optimizing SPDL data
loading pipelines. The single public entry point is
:py:func:`create_workflow`, a
:py:data:`~spdl.autoresearch.core.WorkflowFactory` that the framework
dispatcher resolves via ``--workflow
spdl.autoresearch.pipeline_optimization:create_workflow`` (or the bundled
``autoresearch_with_pipeline_opt`` binary's default).
"""

from ._factory import create_workflow

__all__ = ["create_workflow"]
