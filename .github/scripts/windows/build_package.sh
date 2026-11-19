#!/usr/bin/env bash

# Used by Windows CI build jobs.
set -ex

source .github/scripts/helper.sh
install_uv curl
export PATH="${PATH}:${HOME}/.local/bin/"

setup_build_python "${1}" "${2}"
source .venv/Scripts/activate

cuda_dir="/c/opt/cuda"
configure_cuda_toolkit "${cuda_dir}" "${3}" \
    --component cuda_cudart \
    --component cuda_nvcc \
    --component cuda_crt \
    --component libnvvm \
    --component cuda_cccl
export PATH="${PATH}:${cuda_dir}/bin"
export CUDA_PATH="${cuda_dir}"

uv build --no-python-downloads --all-packages --wheel
mv ./dist ./package
