#include <libspdl/core/executor.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace spdl::core {
void register_executor(py::module& m) {
  auto _ThreadPoolExecutor =
      py::class_<ThreadPoolExecutor, ThreadPoolExecutorPtr>(
          m, "ThreadPoolExecutor", py::module_local());

  _ThreadPoolExecutor.def(
      py::init<size_t, const std::string&, int>(),
      py::arg("num_threads"),
      py::arg("thread_name_prefix"),
      py::arg("throttle_interval") = 0);
}
} // namespace spdl::core
