#include <libspdl/core/storage.h>

#include <libspdl/core/detail/tracing.h>
#ifdef SPDL_USE_CUDA
#include <libspdl/core/detail/cuda.h>
#endif

#include <fmt/core.h>
#include <folly/logging/xlog.h>

namespace spdl::core {

CPUStorage::CPUStorage(void* data) : data_(data) {}
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
    TRACE_EVENT("decoding", "operator delete");
    operator delete(data_);
  }
}

void* CPUStorage::data() const {
  return data_;
}

#ifdef SPDL_USE_CUDA
CUDAStorage::CUDAStorage(void* data, CUstream stream_)
    : data_(data), stream(stream_) {}

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
    TRACE_EVENT("decoding", "cudaFreeAsync");
    CHECK_CUDA(cudaFreeAsync(data_, 0), "Failed to free CUDA memory");
  }
}

void* CUDAStorage::data() const {
  return data_;
}
#endif

} // namespace spdl::core
