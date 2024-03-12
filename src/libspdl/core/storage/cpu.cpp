#include <libspdl/core/storage.h>

#include "libspdl/core/detail/tracing.h"

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
} // namespace spdl::core
