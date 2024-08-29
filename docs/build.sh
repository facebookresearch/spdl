#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

cd "$(dirname "$0")"
rm -rf _build source/generated

mkdir -p _build/breathe/doxygen/libspdl/xml/

python3 -m sphinx.cmd.build -v source _build
