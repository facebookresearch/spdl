#include <libspdl/core/storage.h>

#include "libspdl/core/detail/tracing.h"

#include <utility>

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// Storage
////////////////////////////////////////////////////////////////////////////////
CPUStorage::CPUStorage(size_t size) {
  TRACE_EVENT(
      "decoding",
      "CPUStorage::CPUStorage",
      perfetto::Flow::ProcessScoped(reinterpret_cast<uintptr_t>(this)));
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
    operator delete(data_);
  }
}

void* CPUStorage::data() const {
  return data_;
}
} // namespace spdl::core
