#!/usr/bin/env bash

# Used by Windows CI build jobs.

set -x

curl -LsSf https://astral.sh/uv/install.sh | sh
set -ex

py_ver="${1}"
if [[ "${2}" == 'ft' ]]; then
    py_ver="${py_ver}t"
fi

export PATH="${PATH}:${HOME}/.local/bin/"

uv python pin "${py_ver}"
uv python list --only-installed
uv venv
uv build --no-python-downloads --all-packages --wheel
mv ./dist ./package
