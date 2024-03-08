#pragma once

#include <libspdl/core/storage.h>
#include <libspdl/core/types.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// Buffer
////////////////////////////////////////////////////////////////////////////////

/// Abstract base buffer class (to be exposed to Python)
/// Represents contiguous array memory.
struct Buffer {
  ///
  /// Shape of buffer
  std::vector<size_t> shape;
  ///
  /// Type of unit element
  ElemClass elem_class = ElemClass::UInt;
  ///
  /// Size of unit element
  size_t depth = sizeof(uint8_t);
  ///
  /// Whether channel last (NHWC for image/video, NC for audio)
  bool channel_last = false;

  ///
  /// The actual data.
  std::shared_ptr<Storage> storage;

  Buffer(
      std::vector<size_t> shape,
      bool channel_last,
      ElemClass elem_class,
      size_t depth,
      Storage* storage);
  virtual ~Buffer() = default;

  ///
  /// Returns the pointer to the head of the data buffer.
  void* data();

  ///
  /// True if the data is on a CUDA device.
  virtual bool is_cuda() const = 0;
};

///
/// Contiguous array data on CPU memory.
struct CPUBuffer : Buffer {
  CPUBuffer(
      const std::vector<size_t> shape,
      bool channel_last,
      ElemClass elem_class,
      size_t depth,
      CPUStorage* storage);

  /// Always false.
  bool is_cuda() const override;
};

/// Contiguous array data on a CUDA device.
/// This class is used to hold data decoded with FFmpeg hardware acceleration.
struct CUDABuffer : Buffer {
#ifndef SPDL_USE_CUDA
  void fail() const;
#else
  CUDABuffer(
      std::vector<size_t> shape,
      bool channel_last,
      ElemClass elem_class,
      size_t depth,
      CUDAStorage* storage);

  bool is_cuda() const override;
  uintptr_t get_cuda_stream() const;
#endif
};

/// Contiguous array data on a CUDA device.
/// This class is used to hold data decoded with NVDEC.
struct CUDABuffer2DPitch {
#ifndef SPDL_USE_NVDEC
  void fail() const;
#else
  ///
  /// information to track the stateo f memory
  size_t max_frames;
  ///
  /// If this is image or video.
  bool is_image = false;

  ///
  /// Shape of the data.
  bool channel_last = false;
  /// ``n`` keeps track of how many frames are written.
  /// ``n`` < max_frames;
  /// ``n`` is updated by writer.
  size_t n{0}, c{0}, h{0}, w{0}, bpp{0};
  size_t width_in_bytes{0};

  ///
  /// Data pointer set by CUDA API
  CUdeviceptr p{0};
  ///
  /// Pitch size, set by CUDA API
  size_t pitch{0};

  CUDABuffer2DPitch(size_t max_frames, bool is_image = false);
  ~CUDABuffer2DPitch();

  /// Allocate the memory big enough to hold data for ``(max_frames, c, h, w)``
  /// The actual data size depends on ``bpp`` and ``pitch``.
  void allocate(size_t c, size_t h, size_t w, size_t bpp, bool channel_last);
  ///
  /// Get the shape of the data.
  std::vector<size_t> get_shape() const;
  ///
  /// Get the pointer to the head of the next frame.
  uint8_t* get_next_frame();
#endif
};

////////////////////////////////////////////////////////////////////////////////
// Factory functions
////////////////////////////////////////////////////////////////////////////////

/// Create ``CPUBuffer``.
std::unique_ptr<CPUBuffer> cpu_buffer(
    const std::vector<size_t> shape,
    bool channel_last = false,
    ElemClass elem_class = ElemClass::UInt,
    size_t size = sizeof(uint8_t));

#ifdef SPDL_USE_CUDA
///
/// Create ``CUDABuffer``.
std::unique_ptr<CUDABuffer> cuda_buffer(
    const std::vector<size_t> shape,
    CUstream stream,
    bool channel_last = false);
#endif
} // namespace spdl::core
