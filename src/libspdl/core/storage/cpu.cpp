#include <libspdl/core/storage.h>

#include "libspdl/core/detail/tracing.h"
#ifdef SPDL_USE_CUDA
#include "libspdl/core/detail/cuda.h"
#else
#include "libspdl/core/detail/logging.h"
#endif

#include <glog/logging.h>

#include <utility>

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// Storage
////////////////////////////////////////////////////////////////////////////////
CPUStorage::CPUStorage(size_t size, bool pin_memory) {
  TRACE_EVENT(
      "decoding",
      "CPUStorage::CPUStorage",
      perfetto::Flow::ProcessScoped(reinterpret_cast<uintptr_t>(this)));
  if (pin_memory) {
#ifndef SPDL_USE_CUDA
    LOG(WARNING)
        << "`pin_memory` requires SPDL with CUDA support. Falling back to CPU memory.";
#else
    CHECK_CUDA(
        cudaHostAlloc(&data_, size, cudaHostAllocDefault),
        "Failed to allocate pinned memory.");
    memory_pinned = true;
    return;
#endif
  }
  data_ = operator new(size);
}
CPUStorage::CPUStorage(CPUStorage&& other) noexcept {
  *this = std::move(other);
}
CPUStorage& CPUStorage::operator=(CPUStorage&& other) noexcept {
  using std::swap;
  swap(data_, other.data_);
  return *this;
}
CPUStorage::~CPUStorage() {
  if (data_) {
    TRACE_EVENT(
        "decoding",
        "CPUStorage::~CPUStorage",
        perfetto::Flow::ProcessScoped(reinterpret_cast<uintptr_t>(this)));
    if (memory_pinned) {
#ifndef SPDL_USE_CUDA
      LOG(WARNING) << "SPDL is not compiled with CUDA support, and "
                      "`memory_pinned` attribute should not be true.";
#else
      auto status = cudaFreeHost(data_);
      if (status != cudaSuccess) {
        LOG(ERROR) << fmt::format(
            "Failed to free CUDA memory ({}: {})",
            cudaGetErrorName(status),
            cudaGetErrorString(status));
      }
#endif
    } else {
      operator delete(data_);
    }
  }
}

void* CPUStorage::data() const {
  return data_;
}
} // namespace spdl::core
