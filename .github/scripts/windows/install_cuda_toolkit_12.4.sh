#!/usr/bin/env bash

# Note:
# In the future we can use `redistrib_12.3.2.json` files to figure out the files automatically
# Convert this to Python?

set -e

base_url="https://developer.download.nvidia.com/compute/cuda/redist"
base_dir="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4"

routes=(
  "cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.4.127-archive"
  "cuda_nvcc/windows-x86_64/cuda_nvcc-windows-x86_64-12.4.131-archive"
)

set -x

mkdir -p "${base_dir}"
for route in "${routes[@]}"; do
    component="${route##*/}"

    curl -LO "${base_url}/${route}.zip"
    unzip "${component}.zip" -d "${base_dir}"
    rm "${component}.zip"

    cp -r "${base_dir}/${component}/"* "${base_dir}/"
done
