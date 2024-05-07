#include <libspdl/coro/future.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace spdl::coro {

void register_future(nb::module_& m) {
  nb::class_<Future>(m, "Future")
      .def("cancelled", &Future::cancelled)
      .def("cancel", &Future::cancel);
}
} // namespace spdl::coro
