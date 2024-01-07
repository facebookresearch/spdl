#pragma once

#include <libspdl/storage.h>
#include <libspdl/types.h>

#include <memory>
#include <variant>
#include <vector>

namespace spdl {

// buffer class to be exposed to python
struct Buffer {
  using StorageVariants =
#ifdef SPDL_USE_CUDA
      std::variant<Storage, CUDAStorage>;
#else
      std::variant<Storage>;
#endif

  std::vector<size_t> shape;
  ElemClass elem_class = ElemClass::UInt;
  size_t depth = sizeof(uint8_t);
  bool channel_last = false;

  std::shared_ptr<StorageVariants> storage;

  Buffer(
      const std::vector<size_t> shape,
      bool channel_last,
      ElemClass elem_class,
      size_t depth,
      Storage&& storage);
#ifdef SPDL_USE_CUDA
  Buffer(
      const std::vector<size_t> shape,
      bool channel_last,
      CUDAStorage&& storage);
#endif

  void* data();
  bool is_cuda() const;

#ifdef SPDL_USE_CUDA
  uintptr_t get_cuda_stream() const;
#endif
};

Buffer cpu_buffer(
    const std::vector<size_t> shape,
    bool channel_last = false,
    ElemClass elem_class = ElemClass::UInt,
    size_t size = sizeof(uint8_t));

#ifdef SPDL_USE_CUDA
Buffer cuda_buffer(
    const std::vector<size_t> shape,
    CUstream stream,
    bool channel_last = false);
#endif
} // namespace spdl
