#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Configure and build native tests without installing SPDL or a Python binding.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ffmpeg_version=${1:-8}
build_root=${2:-"${repo_root}/build/libspdl-tests"}
parallel=${CMAKE_BUILD_PARALLEL_LEVEL:-2}

case "${ffmpeg_version}" in
  4|5|6|7|8|9) ;;
  *) echo "Expected an FFmpeg major version (4 through 9), got: ${ffmpeg_version}" >&2; exit 2 ;;
esac

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the native-test Conda environment before running this script." >&2
  exit 2
fi
runtime_version=$(ffmpeg -version | sed -n '1s/^ffmpeg version \([0-9]*\).*/\1/p')
if [[ "${runtime_version}" != "${ffmpeg_version}" ]]; then
  echo "FFmpeg ${ffmpeg_version} tests need the matching full runtime; active CLI is ${runtime_version:-unknown}." >&2
  echo "Install ffmpeg=${ffmpeg_version}.*=gpl_* in a separate Conda environment." >&2
  exit 2
fi

mkdir -p "${build_root}"
build_root=$(cd "${build_root}" && pwd)

# Reuse the same pinned dependency stage as wheel builds. No C++ modules are
# used, so a separate clang-scan-deps installation is unnecessary.
cmake -S "${repo_root}/src/third_party" -B "${build_root}/deps-build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_SCAN_FOR_MODULES=OFF \
  -DCMAKE_INSTALL_PREFIX="${build_root}/deps" \
  -DSPDL_BUILD_TESTS=ON \
  -DBUILD_TESTING=OFF
cmake --build "${build_root}/deps-build" --target install --parallel "${parallel}"

cmake -S "${repo_root}/src" -B "${build_root}/ffmpeg${ffmpeg_version}" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_SCAN_FOR_MODULES=OFF \
  -DCMAKE_PREFIX_PATH="${build_root}/deps" \
  -DSPDL_BUILD_PYTHON_BINDING=OFF \
  -DSPDL_BUILD_TESTS=ON \
  -DSPDL_TEST_FFMPEG_ROOT="${CONDA_PREFIX}" \
  -DSPDL_USE_FFMPEG_VERSION="${ffmpeg_version}"
cmake --build "${build_root}/ffmpeg${ffmpeg_version}" \
  --target libspdl_tests --parallel "${parallel}"

ctest --test-dir "${build_root}/ffmpeg${ffmpeg_version}" \
  --output-on-failure --no-tests=error --parallel "${parallel}" \
  --output-junit "${build_root}/ctest-ffmpeg${ffmpeg_version}.xml"
