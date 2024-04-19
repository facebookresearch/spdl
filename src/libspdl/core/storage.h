#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

#ifdef SPDL_USE_CUDA
#include <cuda.h>
#endif

namespace spdl::core {

struct Storage {
  virtual void* data() const = 0;
  virtual ~Storage() = default;
};

struct CPUStorage : Storage {
  void* data_ = nullptr;

  void* data() const override;

  CPUStorage() = default;
  CPUStorage(size_t size);

  CPUStorage(const CPUStorage&) = delete;
  CPUStorage& operator=(const CPUStorage&) = delete;

  CPUStorage(CPUStorage&&) noexcept;
  CPUStorage& operator=(CPUStorage&&) noexcept;

  ~CPUStorage();
};

using cuda_allocator_fn = std::function<uintptr_t(int, int, uintptr_t)>;
using cuda_deleter_fn = std::function<void(uintptr_t)>;

struct CUDAStorage : Storage {
#ifdef SPDL_USE_CUDA
  void* data_ = nullptr;
  CUstream stream = 0;

  cuda_deleter_fn deleter;

  void* data() const override;

  CUDAStorage() = default;
  CUDAStorage(size_t size, CUstream stream);
  CUDAStorage(
      size_t size,
      int device,
      uintptr_t stream,
      const cuda_allocator_fn& allocator,
      cuda_deleter_fn deleter);

  CUDAStorage(const CUDAStorage&) = delete;
  CUDAStorage& operator=(const CUDAStorage&) = delete;

  CUDAStorage(CUDAStorage&&) noexcept;
  CUDAStorage& operator=(CUDAStorage&&) noexcept;

  ~CUDAStorage();
#endif
};

} // namespace spdl::core
