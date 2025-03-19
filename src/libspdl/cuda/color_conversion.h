/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/cuda/buffer.h>
#include <libspdl/cuda/types.h>

#include <vector>

namespace spdl::cuda {

CUDABufferPtr nv12_to_planar_rgba(
    const std::vector<CUDABuffer>& frames,
    const CUDAConfig& cfg,
    int matrix_coefficients = 1);

CUDABufferPtr nv12_to_planar_bgra(
    const std::vector<CUDABuffer>& frames,
    const CUDAConfig& cfg,
    int matrix_coefficients = 1);

}; // namespace spdl::cuda
