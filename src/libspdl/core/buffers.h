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
// buffer class to be exposed to python
struct Buffer {
  std::vector<size_t> shape;
  ElemClass elem_class = ElemClass::UInt;
  size_t depth = sizeof(uint8_t);
  bool channel_last = false;

  std::shared_ptr<Storage> storage;

  Buffer(
      std::vector<size_t> shape,
      bool channel_last,
      ElemClass elem_class,
      size_t depth,
      Storage* storage);
  virtual ~Buffer() = default;
  void* data();
  virtual bool is_cuda() const = 0;
};

struct CPUBuffer : Buffer {
  CPUBuffer(
      const std::vector<size_t> shape,
      bool channel_last,
      ElemClass elem_class,
      size_t depth,
      CPUStorage* storage);

  bool is_cuda() const override;
};

#ifdef SPDL_USE_CUDA
struct CUDABuffer : Buffer {
  CUDABuffer(
      std::vector<size_t> shape,
      bool channel_last,
      ElemClass elem_class,
      size_t depth,
      CUDAStorage* storage);

  bool is_cuda() const override;
  uintptr_t get_cuda_stream() const;
};
#endif

#ifdef SPDL_USE_NVDEC
struct CUDABuffer2DPitch {
  // const until we introduce RGB and other picture formats
  const bool channel_last = false;

  // information to track the stateo f memory
  size_t max_frames;

  // Shape of the data.
  // n will be updated each time a frame is written
  size_t n{0}, c{0}, h{0}, w{0}, bpp{0};
  size_t width_in_bytes{0};
  size_t dst_h{0}; // The next height position to be written
  // n and dst_h are updated by writer.

  // Data pointer and pitch size, set by CUDA API
  CUdeviceptr p{0};
  size_t pitch{0};

  CUDABuffer2DPitch(size_t max_frames);
  ~CUDABuffer2DPitch();

  void allocate(size_t c, size_t h, size_t w, size_t bpp);
  std::vector<size_t> get_shape() const;
};
#endif

////////////////////////////////////////////////////////////////////////////////
// Factory functions
////////////////////////////////////////////////////////////////////////////////
std::unique_ptr<CPUBuffer> cpu_buffer(
    const std::vector<size_t> shape,
    bool channel_last = false,
    ElemClass elem_class = ElemClass::UInt,
    size_t size = sizeof(uint8_t));

#ifdef SPDL_USE_CUDA
std::unique_ptr<CUDABuffer> cuda_buffer(
    const std::vector<size_t> shape,
    CUstream stream,
    bool channel_last = false);
#endif
} // namespace spdl::core
