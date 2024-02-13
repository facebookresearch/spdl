#include <libspdl/core/storage.h>

#include <libspdl/core/detail/tracing.h>
#ifdef SPDL_USE_CUDA
#include <libspdl/core/detail/cuda.h>
#endif

#include <fmt/core.h>
#include <folly/logging/xlog.h>

namespace spdl::core {
namespace {
void* _get_buffer(size_t size) {
  TRACE_EVENT("decoding", "storage::_get_buffer");
  return operator new(size);
}
} // namespace

Storage::Storage(size_t size) : data(_get_buffer(size)) {}
Storage::Storage(Storage&& other) noexcept {
  *this = std::move(other);
}
Storage& Storage::operator=(Storage&& other) noexcept {
  using std::swap;
  swap(data, other.data);
  return *this;
}
Storage::~Storage() {
  if (data) {
    operator delete(data);
  }
}

#ifdef SPDL_USE_CUDA
CUDAStorage::CUDAStorage(size_t size, CUstream stream_) : stream(stream_) {
  XLOG(DBG9) << fmt::format("Allocating CUDA memory {} ({} bytes)", data, size);
  CHECK_CUDA(cudaMallocAsync(&data, size, 0), "Failed to allocate CUDA memory");
  XLOG(DBG9) << fmt::format("Allocation queued {}", data);
}
CUDAStorage::CUDAStorage(CUDAStorage&& other) noexcept {
  *this = std::move(other);
}
CUDAStorage& CUDAStorage::operator=(CUDAStorage&& other) noexcept {
  using std::swap;
  swap(data, other.data);
  swap(stream, other.stream);
  return *this;
}
CUDAStorage::~CUDAStorage() {
  if (data) {
    XLOG(DBG9) << "Freeing CUDA memory " << data;
    CHECK_CUDA(cudaFreeAsync(data, 0), "Failed to free CUDA memory");
  }
}
#endif

} // namespace spdl::core
