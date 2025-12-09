/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <libspdl/core/buffer.h>
#include <libspdl/core/types.h>
#include <libspdl/cuda/buffer.h>

#include <vector>

namespace spdl::cuda {

/// Transfer CPU buffer to CUDA device.
///
/// Copies data from CPU memory to GPU memory.
///
/// @param buffer CPU buffer to transfer.
/// @param cfg CUDA configuration including device and stream.
/// @return CUDA buffer containing the transferred data.
CUDABufferPtr transfer_buffer(core::CPUBufferPtr buffer, const CUDAConfig& cfg);

/// Transfer generic array from CPU to CUDA device.
///
/// Copies data from a generic CPU array (e.g., NumPy array) to GPU memory.
///
/// @param shape Dimensions of the array.
/// @param elem_class Element class (Int, UInt, or Float).
/// @param depth Size of each element in bytes.
/// @param ptr Pointer to CPU data.
/// @param cfg CUDA configuration including device and stream.
/// @return CUDA buffer containing the transferred data.
CUDABufferPtr transfer_buffer(
    const std::vector<size_t>& shape,
    core::ElemClass elem_class,
    size_t depth,
    void* ptr,
    const CUDAConfig& cfg);

/// Transfer data from CUDA device to CPU.
///
/// Copies data from GPU memory to a new CPU buffer.
///
/// @param shape Dimensions of the array.
/// @param elem_class Element class (Int, UInt, or Float).
/// @param depth Size of each element in bytes.
/// @param ptr Pointer to GPU data.
/// @return CPU buffer containing the transferred data.
spdl::core::CPUBufferPtr transfer_buffer(
    const std::vector<size_t>& shape,
    core::ElemClass elem_class,
    size_t depth,
    const void* ptr);

/// Copy data from GPU to CPU storage.
///
/// @param src Pointer to source GPU data.
/// @param shape Dimensions of the data.
/// @return CPU storage containing the copied data.
spdl::core::CPUStorage cp_to_cpu(
    const void* src,
    const std::vector<size_t>& shape);
} // namespace spdl::cuda
