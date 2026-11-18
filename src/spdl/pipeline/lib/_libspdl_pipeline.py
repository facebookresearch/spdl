# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Python interface importable before the optional Rust binding is built."""

from ._buffer import _export, _export_owned, _release_ticket

__all__ = ["_export", "_export_owned", "_release_ticket"]
