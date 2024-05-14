#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>

#ifdef SPDL_USE_CUDA
#include <cuda.h>
#endif

namespace spdl::core {

struct Storage {
  virtual void* data() const = 0;
  virtual ~Storage() = default;
};

class CPUStorage : Storage {
  void* data_ = nullptr;

 public:
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
using cuda_allocator = std::pair<cuda_allocator_fn, cuda_deleter_fn>;

class CUDAStorage : Storage {
#ifdef SPDL_USE_CUDA
  void* data_ = nullptr;

 public:
  CUstream stream = 0;

  cuda_deleter_fn deleter;

  void* data() const override;

  CUDAStorage() = default;
  CUDAStorage(size_t size, int device, CUstream stream);
  CUDAStorage(
      size_t size,
      int device,
      uintptr_t stream,
      const std::optional<cuda_allocator>& allocator = std::nullopt);

  CUDAStorage(const CUDAStorage&) = delete;
  CUDAStorage& operator=(const CUDAStorage&) = delete;

  CUDAStorage(CUDAStorage&&) noexcept;
  CUDAStorage& operator=(CUDAStorage&&) noexcept;

  ~CUDAStorage();
#endif
};

} // namespace spdl::core
