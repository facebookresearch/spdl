#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Entry point for ``spdl autoresearch``.

Thin wrapper that delegates to the SPDL pipeline optimization
implementation in ``spdl.autoresearch.pipeline_optimization``.
"""

from __future__ import annotations

from spdl.autoresearch.pipeline_optimization import main

if __name__ == "__main__":
    main()
