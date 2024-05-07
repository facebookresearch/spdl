#include <libspdl/coro/executor.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

namespace spdl::coro {
void register_executor(nb::module_& m) {
  nb::class_<ThreadPoolExecutor>(m, "ThreadPoolExecutor")
      .def(
          nb::init<size_t, const std::string&>(),
          nb::arg("num_threads"),
          nb::arg("thread_name_prefix"))
      .def("get_task_queue_size", &ThreadPoolExecutor::get_task_queue_size);

  m.def(
      "trace_default_demux_executor_queue_size",
      &trace_default_demux_executor_queue_size);
  m.def(
      "trace_default_decode_executor_queue_size",
      &trace_default_decode_executor_queue_size);
}
} // namespace spdl::coro
