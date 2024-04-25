#include <libspdl/core/utils.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

namespace nb = nanobind;

namespace spdl::core {

void register_tracing(nb::module_& m) {
  auto _TracingSession = nb::class_<TracingSession>(m, "TracingSession");

  _TracingSession.def("init", &TracingSession::init)
      .def("config", &TracingSession::config)
      .def("start", &TracingSession::start)
      .def("stop", &TracingSession::stop);

  m.def("init_tracing", &init_tracing);
  m.def("trace_counter", &trace_counter<int>);
  m.def("trace_counter", &trace_counter<double>);
  m.def("trace_event_begin", &trace_event_begin);
  m.def("trace_event_end", &trace_event_end);

  m.def(
      "trace_default_demux_executor_queue_size",
      &trace_default_demux_executor_queue_size);
  m.def(
      "trace_default_decode_executor_queue_size",
      &trace_default_decode_executor_queue_size);
}

} // namespace spdl::core
