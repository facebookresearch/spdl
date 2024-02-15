#pragma once

#include <libspdl/core/storage.h>
#include <libspdl/core/types.h>

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
