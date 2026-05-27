#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Entry point for the bare ``spdl-autoresearch`` Buck binary.

Thin shim that delegates to the framework dispatcher in
``spdl.autoresearch._app._main``. The framework dispatcher requires a
``--workflow module.path:factory`` argument (or short name registered
under the ``spdl.autoresearch.workflows`` entry-points group); the
bundled ``autoresearch_with_pipeline_opt`` Buck binary uses a sibling
shim that auto-injects the pipeline-optimization workflow as the
default.
"""

from __future__ import annotations

from spdl.autoresearch._app._main import main

if __name__ == "__main__":
    main()
