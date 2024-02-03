#pragma once

#include <cstddef>
#include <cstdint>

#ifdef SPDL_USE_CUDA
#include <cuda.h>
#endif

namespace spdl {

struct Storage {
  void* data = nullptr;

  Storage() = default;
  Storage(size_t size);

  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;

  Storage(Storage&&) noexcept;
  Storage& operator=(Storage&&) noexcept;

  ~Storage();
};

#ifdef SPDL_USE_CUDA
struct CUDAStorage {
  void* data = nullptr;
  CUstream stream = 0;

  CUDAStorage() = default;
  CUDAStorage(size_t size, CUstream stream);

  CUDAStorage(const CUDAStorage&) = delete;
  CUDAStorage& operator=(const CUDAStorage&) = delete;

  CUDAStorage(CUDAStorage&&) noexcept;
  CUDAStorage& operator=(CUDAStorage&&) noexcept;

  ~CUDAStorage();
};
#endif

} // namespace spdl
