# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Framework for automated, LLM-driven experiment workflows.

Provides reusable infrastructure for running automated research loops:

- :py:mod:`~spdl.autoresearch.core`: a domain-neutral async work scheduler
  and the :py:class:`~spdl.autoresearch.core.WorkflowProtocol` /
  :py:class:`~spdl.autoresearch.core.WorkflowSpec` contracts that pluggable
  workflows implement.
- :py:mod:`~spdl.autoresearch.pipeline_optimization`:
  concrete workflow implementation for SPDL data loading pipeline optimization.

The framework dispatcher (private) lives under ``spdl.autoresearch._app``.
Users interact with it through the ``spdl-autoresearch`` CLI rather than by
importing these helpers directly; importing from ``spdl.autoresearch._app``
is reserved for framework code and tests.
"""

__all__: list[str] = []
