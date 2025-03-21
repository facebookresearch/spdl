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

// Initialize CUDA context
void init_cuda();

void synchronize_stream(const CUDAConfig& handle);

} // namespace spdl::cuda
