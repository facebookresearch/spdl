#pragma once

#include <cstddef>
#include <cstdint>

namespace spdl {

struct Storage {
  uint8_t* data = nullptr;

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

  CUDAStorage() = default;
  CUDAStorage(size_t size);

  CUDAStorage(const CUDAStorage&) = delete;
  CUDAStorage& operator=(const CUDAStorage&) = delete;

  CUDAStorage(CUDAStorage&&) noexcept;
  CUDAStorage& operator=(CUDAStorage&&) noexcept;

  ~CUDAStorage();
};
#endif

} // namespace spdl
