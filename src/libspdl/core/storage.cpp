#include <libspdl/core/storage.h>

#include <libspdl/core/detail/tracing.h>
#ifdef SPDL_USE_CUDA
#include <libspdl/core/detail/cuda.h>
#endif

#include <fmt/core.h>
#include <folly/logging/xlog.h>

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// Storage
////////////////////////////////////////////////////////////////////////////////
namespace {
void* _get_buffer(size_t size) {
  TRACE_EVENT("decoding", "storage::_get_buffer");
  return operator new(size);
}
} // namespace

CPUStorage::CPUStorage(size_t size) : data_(_get_buffer(size)) {}
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
    operator delete(data_);
  }
}

void* CPUStorage::data() const {
  return data_;
}

////////////////////////////////////////////////////////////////////////////////
// CUDAStorage
////////////////////////////////////////////////////////////////////////////////
#ifdef SPDL_USE_CUDA
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
#endif

} // namespace spdl::core
