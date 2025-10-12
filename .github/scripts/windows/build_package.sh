#!/usr/bin/env bash

# Used by Windows CI build jobs.

set -x

py_ver="${1}"
if [[ "${2}" == 'ft' ]]; then
    py_ver="${py_ver}t"
fi
cuda_ver="${3}"

curl -LsSf https://astral.sh/uv/install.sh | sh
set -ex

export PATH="${PATH}:${HOME}/.local/bin/"

uv python pin "${py_ver}"
uv python list --only-installed
uv venv

cuda_dir="/c/opt/cuda"
.github/scripts/install_cuda_toolkit.py \
       --base-dir "${cuda_dir}" \
       --cuda-version "${cuda_ver}"

export PATH="${PATH}:${cuda_dir}/bin"
export CUDA_PATH="${cuda_dir}"

uv build --no-python-downloads --all-packages --wheel
mv ./dist ./package
