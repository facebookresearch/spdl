/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libspdl/cuda/nvdec/detail/buffer.h"

#include "libspdl/core/detail/tracing.h"
#include "libspdl/cuda/detail/utils.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <fmt/core.h>

namespace spdl::cuda::detail {

FrameBuffer::FrameBuffer(
    size_t num_frames,
    size_t width,
    size_t height,
    const CUDAConfig& cfg)
    : shape_({num_frames, height + height / 2, width}), cfg_(cfg) {}

bool FrameBuffer::empty() const {
  return queue_.empty();
}

void FrameBuffer::push(void* src_ptr, size_t pitch) {
  if (!current_) {
    current_ = cuda_buffer(shape_, cfg_);
    idx_ = 0;
  }

  // get the next buffer point
  auto h2 = shape_[1], width = shape_[2];
  size_t offset = idx_ * h2 * width;
  auto* dst_ptr = (uint8_t*)current_->data() + offset;

  // Perform copy
  CUDA_MEMCPY2D cfg{
      .srcXInBytes = 0,
      .srcY = 0,
      .srcMemoryType = CU_MEMORYTYPE_DEVICE,
      .srcHost = nullptr,
      .srcDevice = (CUdeviceptr)src_ptr,
      .srcArray = nullptr,
      .srcPitch = pitch,

      .dstXInBytes = 0,
      .dstY = 0,
      .dstMemoryType = CU_MEMORYTYPE_DEVICE,
      .dstHost = nullptr,
      .dstDevice = (CUdeviceptr)dst_ptr,
      .dstArray = nullptr,
      .dstPitch = width,

      .WidthInBytes = width,
      .Height = h2,
  };

  TRACE_EVENT("nvdec", "cuMemcpy2DAsync");
  auto stream = (CUstream)cfg_.stream;
  CHECK_CU(cuMemcpy2DAsync(&cfg, stream), "Failed to copy a frame.");
  CHECK_CU(cuStreamSynchronize(stream), "Failed to synchronize stream.");

  // Move the buffer if full
  ++idx_;
  if (idx_ == shape_[0]) {
    queue_.emplace_back(current_.release());
  }
}

CUDABufferPtr FrameBuffer::pop() {
  if (queue_.empty()) {
    SPDL_FAIL_INTERNAL(fmt::format("There is no buffer available."));
  }
  CUDABufferPtr ret = std::move(queue_.front());
  queue_.pop_front();
  return ret;
}

void FrameBuffer::flush() {
  if (current_ && idx_ > 0) {
    // Adjust shape to reflect actual number of frames
    current_->shape[0] = idx_;
    queue_.emplace_back(current_.release());
    idx_ = 0;
  }
}

}; // namespace spdl::cuda::detail
