#include <libspdl/core/adaptor.h>

#include <fmt/core.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

namespace nb = nanobind;

namespace spdl::core {

void register_adaptors(nb::module_& m) {
  nb::class_<SourceAdaptor>(m, "SourceAdaptor").def(nb::init<>());

  nb::class_<MMapAdaptor>(m, "MMapAdaptor").def(nb::init<>());

  nb::class_<BytesAdaptor>(m, "BytesAdaptor").def(nb::init<>());
}
} // namespace spdl::core
