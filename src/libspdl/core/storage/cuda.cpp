#include <libspdl/core/storage.h>

#include "libspdl/core/detail/cuda.h"

#include <fmt/core.h>
#include <folly/logging/xlog.h>

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// CUDAStorage
////////////////////////////////////////////////////////////////////////////////
CUDAStorage::CUDAStorage(size_t size, CUstream stream_) : stream(stream_) {
  XLOG(DBG9) << fmt::format("Allocating CUDA memory ({} bytes)", size);
  CHECK_CUDA(
      cudaMallocAsync(&data_, size, 0), "Failed to allocate CUDA memory");
  XLOG(DBG9) << fmt::format("Allocation queued {}", data_);
}
CUDAStorage::CUDAStorage(CUDAStorage&& other) noexcept {
  *this = std::move(other);
}
CUDAStorage& CUDAStorage::operator=(CUDAStorage&& other) noexcept {
  using std::swap;
  swap(data_, other.data_);
  swap(stream, other.stream);
  return *this;
}
CUDAStorage::~CUDAStorage() {
  if (data_) {
    XLOG(DBG9) << "Freeing CUDA memory " << data_;
    CHECK_CUDA(cudaFreeAsync(data_, 0), "Failed to free CUDA memory");
  }
}

void* CUDAStorage::data() const {
  return data_;
}

} // namespace spdl::core
