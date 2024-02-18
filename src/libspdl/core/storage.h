#pragma once

#include <cstddef>
#include <cstdint>

#if defined(SPDL_USE_CUDA) || defined(SPDL_USE_NVDEC)
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

#ifdef SPDL_USE_CUDA
struct CUDAStorage : Storage {
  void* data_ = nullptr;
  CUstream stream = 0;

  void* data() const override;

  CUDAStorage() = default;
  CUDAStorage(size_t size, CUstream stream);

  CUDAStorage(const CUDAStorage&) = delete;
  CUDAStorage& operator=(const CUDAStorage&) = delete;

  CUDAStorage(CUDAStorage&&) noexcept;
  CUDAStorage& operator=(CUDAStorage&&) noexcept;

  ~CUDAStorage();
};
#endif

} // namespace spdl::core
