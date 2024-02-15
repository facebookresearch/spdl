#include <libspdl/core/utils.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace spdl::core {

void register_tracing(py::module& m) {
  auto _TracingSession =
      py::class_<TracingSession>(m, "TracingSession", py::module_local());

  _TracingSession.def("init", &TracingSession::init)
      .def("config", &TracingSession::config)
      .def("start", &TracingSession::start)
      .def("stop", &TracingSession::stop);

  m.def("init_tracing", init_tracing);
}

} // namespace spdl::core
