# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Framework dispatcher for the pluggable autoresearch CLI.

The ``spdl-autoresearch`` binary delegates to :py:func:`_main.main` in
this package. The dispatcher resolves a workflow factory specifier
(``module.path:factory``), parses framework-level CLI arguments, and
either launches an interactive supervisor agent or drives the engine
directly. Workflow-specific argv (everything after ``--``) is forwarded
to the factory, which builds a :py:class:`~spdl.autoresearch.core.WorkflowSpec`.
"""

__all__: list[str] = []
