#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Source this file, then call retry_with_backoff command [args...].
retry_with_backoff() {
    local attempt status delay
    for attempt in 1 2 3 4; do
        if "$@"; then
            return 0
        else
            status=$?
        fi
        if (( attempt == 4 )); then
            echo "Command failed after ${attempt} attempts: $*" >&2
            return "${status}"
        fi
        delay=$((5 << (attempt - 1)))
        echo "Attempt ${attempt} failed; retrying in ${delay}s: $*" >&2
        sleep "${delay}" || return $?
    done
}
