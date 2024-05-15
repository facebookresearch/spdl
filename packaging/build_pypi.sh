#!/usr/bin/env bash

set -eux

python_version="${1}"
use_cuda="${2}"
use_tracing="${3}"

packages=(
    patch
    conda-forge::cxx-compiler
)

if [ ${use_cuda} ]; then
    packages+=(nvidia/label/cuda-12.0.0::cuda-toolkit)
fi

conda create --name build_env -y python="${python_version}" "${packages[@]}"

set +x
conda init
eval "$($(which conda) shell.bash hook)"
conda activate build_env
set -x

pip3 install build cmake ninja

export USE_CUDA="${use_cuda}"
export USE_NVCODEC="${use_cuda}"
export USE_TRACING="${use_tracing}"
python3 -m build --no-isolation --wheel
