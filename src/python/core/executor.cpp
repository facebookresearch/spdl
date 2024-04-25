#include <libspdl/core/executor.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

namespace spdl::core {
void register_executor(nb::module_& m) {
  auto _ThreadPoolExecutor =
      nb::class_<ThreadPoolExecutor>(m, "ThreadPoolExecutor");

  _ThreadPoolExecutor
      .def(
          nb::init<size_t, const std::string&>(),
          nb::arg("num_threads"),
          nb::arg("thread_name_prefix"))
      .def("get_task_queue_size", &ThreadPoolExecutor::get_task_queue_size);
}
} // namespace spdl::core
