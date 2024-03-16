#include <libspdl/core/future.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace spdl::core {

void register_future(py::module& m) {
  auto _Future = py::class_<Future, FuturePtr>(m, "Future", py::module_local());

  _Future.def("rethrow", &Future::rethrow);
}
} // namespace spdl::core
