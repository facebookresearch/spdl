#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROMPT="$(cat "$SCRIPT_DIR/prompts/launch.md")"

FB_PROMPT="$SCRIPT_DIR/prompts/fb/launch.md"
if [ -f "$FB_PROMPT" ]; then
  PROMPT="$PROMPT

$(cat "$FB_PROMPT")"
fi

if [ $# -gt 0 ]; then
  exec claude --system-prompt "$PROMPT" "$*"
else
  exec claude --system-prompt "$PROMPT"
fi
