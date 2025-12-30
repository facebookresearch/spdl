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

/// Convert batched NV12 frames (3D buffer) to planar RGB format on GPU.
///
/// This is an optimized version that takes a pre-allocated 3D buffer containing
/// multiple NV12 frames, reducing memory allocation overhead.
///
/// @param nv12_batch 3D buffer with shape [num_frames, height*1.5, width].
/// @param cfg CUDA configuration including device and stream.
/// @param matrix_coefficients Color matrix coefficients for conversion
/// (default: BT.709).
/// @param sync If true, synchronizes the stream before returning (default:
/// true).
/// @return CUDA buffer containing planar RGB data with shape [num_frames, 3,
/// height, width].
CUDABufferPtr nv12_to_planar_rgb(
    const CUDABuffer& nv12_batch,
    const CUDAConfig& cfg,
    int matrix_coefficients = 1,
    bool sync = true);

/// Convert batched NV12 frames (3D buffer) to planar BGR format on GPU.
///
/// This is an optimized version that takes a pre-allocated 3D buffer containing
/// multiple NV12 frames, reducing memory allocation overhead.
///
/// @param nv12_batch 3D buffer with shape [num_frames, height*1.5, width].
/// @param cfg CUDA configuration including device and stream.
/// @param matrix_coefficients Color matrix coefficients for conversion
/// (default: BT.709).
/// @param sync If true, synchronizes the stream before returning (default:
/// true).
/// @return CUDA buffer containing planar BGR data with shape [num_frames, 3,
/// height, width].
CUDABufferPtr nv12_to_planar_bgr(
    const CUDABuffer& nv12_batch,
    const CUDAConfig& cfg,
    int matrix_coefficients = 1,
    bool sync = true);

} // namespace spdl::cuda
