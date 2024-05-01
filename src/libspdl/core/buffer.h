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

struct Buffer;
struct CPUBuffer;
struct CUDABuffer;

using BufferPtr = std::unique_ptr<Buffer>;
using CPUBufferPtr = std::unique_ptr<CPUBuffer>;
using CUDABufferPtr = std::unique_ptr<CUDABuffer>;

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
  /// The actual data.
  std::shared_ptr<Storage> storage;

  Buffer(
      std::vector<size_t> shape,
      ElemClass elem_class,
      size_t depth,
      Storage* storage);
  virtual ~Buffer() = default;

  ///
  /// Returns the pointer to the head of the data buffer.
  void* data();

  virtual bool is_cuda() const = 0;
};

///
/// Contiguous array data on CPU memory.
struct CPUBuffer : public Buffer {
  CPUBuffer(
      const std::vector<size_t> shape,
      ElemClass elem_class,
      size_t depth,
      CPUStorage* storage);

  bool is_cuda() const override {
    return false;
  }
};

///
/// Contiguous array data on a CUDA device.
struct CUDABuffer : Buffer {
#ifdef SPDL_USE_CUDA
  int device_index;

  CUDABuffer(
      std::vector<size_t> shape,
      ElemClass elem_class,
      size_t depth,
      CUDAStorage* storage,
      int device_index);

  uintptr_t get_cuda_stream() const;

#endif
  bool is_cuda() const override {
    return true;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Factory functions
////////////////////////////////////////////////////////////////////////////////

/// Create ``CPUBuffer``.
CPUBufferPtr cpu_buffer(
    const std::vector<size_t> shape,
    ElemClass elem_class = ElemClass::UInt,
    size_t depth = sizeof(uint8_t));

#ifdef SPDL_USE_CUDA
///
/// Create ``CUDABuffer``.
CUDABufferPtr cuda_buffer(
    const std::vector<size_t> shape,
    CUstream stream,
    int device_index,
    ElemClass elem_class = ElemClass::UInt,
    size_t depth = sizeof(uint8_t));

CUDABufferPtr cuda_buffer(
    const std::vector<size_t> shape,
    uintptr_t stream,
    int device_index,
    ElemClass elem_class,
    size_t depth,
    const cuda_allocator_fn& allocator,
    const cuda_deleter_fn& deleter);

#endif

} // namespace spdl::core
