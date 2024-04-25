#include <libspdl/core/future.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace spdl::core {

void register_future(nb::module_& m) {
  auto _Future = nb::class_<Future>(m, "Future");

  _Future.def("cancelled", &Future::cancelled).def("cancel", &Future::cancel);
}
} // namespace spdl::core
