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

/// Convert NV12 frames to planar RGB format on GPU.
///
/// Performs color space conversion from NV12 (YUV 4:2:0 with interleaved UV)
/// to planar RGB format using CUDA.
///
/// @param frames Vector of NV12 frame buffers to convert.
/// @param cfg CUDA configuration including device and stream.
/// @param matrix_coefficients Color matrix coefficients for conversion
/// (default: BT.709).
/// @param sync If true, synchronizes the stream before returning (default:
/// true).
/// @return CUDA buffer containing planar RGB data.
CUDABufferPtr nv12_to_planar_rgb(
    const std::vector<CUDABuffer>& frames,
    const CUDAConfig& cfg,
    int matrix_coefficients = 1,
    bool sync = true);

/// Convert NV12 frames to planar BGR format on GPU.
///
/// Performs color space conversion from NV12 (YUV 4:2:0 with interleaved UV)
/// to planar BGR format using CUDA.
///
/// @param frames Vector of NV12 frame buffers to convert.
/// @param cfg CUDA configuration including device and stream.
/// @param matrix_coefficients Color matrix coefficients for conversion
/// (default: BT.709).
/// @param sync If true, synchronizes the stream before returning (default:
/// true).
/// @return CUDA buffer containing planar BGR data.
CUDABufferPtr nv12_to_planar_bgr(
    const std::vector<CUDABuffer>& frames,
    const CUDAConfig& cfg,
    int matrix_coefficients = 1,
    bool sync = true);

/// Convert batched NV12 frames (3D buffer) to planar RGB format on GPU.
///
/// This is an optimized version that takes a pre-allocated 3D buffer containing
/// multiple NV12 frames, reducing memory allocation overhead.
///
/// @param nv12_batch 3D buffer with shape [max_frames, height*1.5, width].
/// @param num_frames Actual number of frames to convert (may be <= max_frames).
/// @param cfg CUDA configuration including device and stream.
/// @param matrix_coefficients Color matrix coefficients for conversion
/// (default: BT.709).
/// @param sync If true, synchronizes the stream before returning (default:
/// true).
/// @return CUDA buffer containing planar RGB data with shape [num_frames, 3,
/// height, width].
CUDABufferPtr nv12_to_planar_rgb_batched(
    const CUDABuffer& nv12_batch,
    size_t num_frames,
    const CUDAConfig& cfg,
    int matrix_coefficients = 1,
    bool sync = true);

/// Convert batched NV12 frames (3D buffer) to planar BGR format on GPU.
///
/// This is an optimized version that takes a pre-allocated 3D buffer containing
/// multiple NV12 frames, reducing memory allocation overhead.
///
/// @param nv12_batch 3D buffer with shape [max_frames, height*1.5, width].
/// @param num_frames Actual number of frames to convert (may be <= max_frames).
/// @param cfg CUDA configuration including device and stream.
/// @param matrix_coefficients Color matrix coefficients for conversion
/// (default: BT.709).
/// @param sync If true, synchronizes the stream before returning (default:
/// true).
/// @return CUDA buffer containing planar BGR data with shape [num_frames, 3,
/// height, width].
CUDABufferPtr nv12_to_planar_bgr_batched(
    const CUDABuffer& nv12_batch,
    size_t num_frames,
    const CUDAConfig& cfg,
    int matrix_coefficients = 1,
    bool sync = true);

} // namespace spdl::cuda
