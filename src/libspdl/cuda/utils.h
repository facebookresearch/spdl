/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/cuda/types.h>

namespace spdl::cuda {

/// Initialize CUDA context.
///
/// Sets up the CUDA runtime environment. Should be called before
/// using CUDA operations.
void init_cuda();

/// Synchronize a CUDA stream.
///
/// Blocks until all operations in the stream have completed.
///
/// @param handle CUDA configuration containing the stream to synchronize.
void synchronize_stream(const CUDAConfig& handle);

} // namespace spdl::cuda
