#include <libspdl/core/storage.h>

#include "libspdl/core/detail/cuda.h"
#include "libspdl/core/detail/tracing.h"

#include <fmt/core.h>
#include <folly/logging/xlog.h>

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// CUDAStorage
////////////////////////////////////////////////////////////////////////////////
CUDAStorage::CUDAStorage(size_t size, CUstream stream_) : stream(stream_) {
  TRACE_EVENT("decoding", "cudaMalloc");
  CHECK_CUDA(cudaMalloc(&data_, size), "Failed to allocate CUDA memory");
}

CUDAStorage::CUDAStorage(
    size_t size,
    int device,
    uintptr_t stream_,
    const cuda_allocator_fn& allocator,
    cuda_deleter_fn deleter_)
    : stream(static_cast<CUstream>((void*)stream_)),
      deleter(std::move(deleter_)) {
  TRACE_EVENT("decoding", "custom_cuda_allocator_fn");
  data_ = reinterpret_cast<void*>(allocator(size, device, stream_));
}

CUDAStorage::CUDAStorage(CUDAStorage&& other) noexcept {
  *this = std::move(other);
}

CUDAStorage& CUDAStorage::operator=(CUDAStorage&& other) noexcept {
  using std::swap;
  swap(data_, other.data_);
  swap(stream, other.stream);
  swap(deleter, other.deleter);
  return *this;
}
CUDAStorage::~CUDAStorage() {
  if (data_) {
    TRACE_EVENT("decoding", "CUDAStorage::~CUDAStorage");
    XLOG(DBG9) << "Freeing CUDA memory " << data_;
    if (deleter) {
      deleter(reinterpret_cast<uintptr_t>(data_));
    } else {
      auto status = cudaFree(data_);
      if (status != cudaSuccess) {
        XLOG(CRITICAL) << fmt::format(
            "Failed to free CUDA memory ({}: {})",
            cudaGetErrorName(status),
            cudaGetErrorString(status));
      }
    }
  }
}

void* CUDAStorage::data() const {
  return data_;
}

} // namespace spdl::core
