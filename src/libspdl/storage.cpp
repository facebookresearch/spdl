#include <libspdl/storage.h>
#ifdef SPDL_USE_CUDA
#include <libspdl/detail/cuda.h>
#endif

#include <fmt/core.h>
#include <folly/logging/xlog.h>

namespace spdl {

Storage::Storage(size_t size) : data(operator new(size)) {}
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
  CUDA_CHECK(cudaMallocAsync(&data, size, 0));
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
    CUDA_CHECK(cudaFreeAsync(data, 0));
  }
}
#endif

} // namespace spdl
