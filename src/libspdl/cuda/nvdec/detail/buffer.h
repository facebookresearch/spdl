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

#include <deque>

namespace spdl::cuda::detail {

/// Batches decoded video frames into contiguous CUDA buffers.
///
/// FrameBuffer accumulates individual decoded frames from NVDEC into
/// fixed-size batched buffers. Each buffer holds a configurable number
/// of frames arranged contiguously in device memory. When a buffer fills,
/// it moves to an internal queue for retrieval.
///
/// The frame data is stored in NV12 format with layout:
/// [num_frames, height + height/2, width]
///
/// This class is non-copyable and non-movable to ensure safe buffer
/// management.
class FrameBuffer {
  std::deque<CUDABufferPtr> queue_{};
  CUDABufferPtr current_{};
  size_t idx_ = 0;

  std::vector<size_t> shape_;
  CUDAConfig cfg_;

 public:
  FrameBuffer(
      size_t num_frames,
      size_t width,
      size_t height,
      const CUDAConfig& cfg);

  // Disable copy semantics
  FrameBuffer(const FrameBuffer&) = delete;
  FrameBuffer& operator=(const FrameBuffer&) = delete;

  // Disable move semantics
  FrameBuffer(FrameBuffer&&) = delete;
  FrameBuffer& operator=(FrameBuffer&&) = delete;

  ~FrameBuffer() = default;

  /// Returns whether the buffer queue is empty.
  ///
  /// @return true if no completed buffers are available, false otherwise
  bool empty() const;

  /// Copies a decoded frame from device memory into the current batch buffer.
  ///
  /// Frames are accumulated until the batch is full (num_frames reached).
  /// When full, the buffer is moved to the queue for retrieval via pop().
  ///
  /// @param ptr Device pointer to the source frame data in NV12 format
  /// @param pitch Source memory pitch (stride) in bytes
  void push(void* ptr, size_t pitch);

  /// Retrieves and removes a completed buffer from the queue.
  ///
  /// @return A CUDABufferPtr containing a batch of frames
  /// @throws std::runtime_error if the queue is empty
  CUDABufferPtr pop();

  /// Flushes the current buffer to the queue, adjusting its shape to reflect
  /// the actual number of frames pushed.
  ///
  /// If the current buffer is empty (no frames pushed), does nothing.
  /// After flushing, the current buffer is reset.
  void flush();
};

} // namespace spdl::cuda::detail
