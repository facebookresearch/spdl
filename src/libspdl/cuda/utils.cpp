/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/cuda/utils.h>

#include "libspdl/cuda/detail/utils.h"

// Make sure CUDA_API_PER_THREAD_DEFAULT_STREAM is defined by compiler
// so that it is applied globally
#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#error CUDA_API_PER_THREAD_DEFAULT_STREAM must be defined.
#endif

#include <cuda.h>

#include <glog/logging.h>

namespace spdl::cuda {

void init_cuda() {
  int c;
  if (cuDeviceGetCount(&c) == CUDA_SUCCESS) {
    VLOG(5) << "CUDA context has been already initialized.";
  } else {
    VLOG(5) << "Initializing CUDA context.";
    // Note: You can make cuInit fail by setting CUDA_VISIBLE_DEVICES to empty
    // string.
    if (CUresult res = cuInit(0); res != CUDA_SUCCESS) {
      CHECK_CU(res, "`cuInit(0)` failed.");
    }
  }
}

} // namespace spdl::cuda
