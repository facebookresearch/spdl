#include <libspdl/core/executor.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace spdl::core {
void register_executor(py::module& m) {
  auto _ThreadPoolExecutor =
      py::class_<ThreadPoolExecutor, ThreadPoolExecutorPtr>(
          m, "ThreadPoolExecutor", py::module_local());

  _ThreadPoolExecutor
      .def(
          py::init<size_t, const std::string&>(),
          py::arg("num_threads"),
          py::arg("thread_name_prefix"))
      .def("get_task_queue_size", &ThreadPoolExecutor::get_task_queue_size);
}
} // namespace spdl::core
