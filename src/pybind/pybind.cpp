#include <glog/logging.h>
#include <libspdl/processors.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifndef SPDL_FFMPEG_EXT_NAME
#error SPDL_FFMPEG_EXT_NAME must be defined.
#endif


namespace google {
namespace glog_internal_namespace_ {
bool IsGoogleLoggingInitialized();
} // namespace glog_internal_namespace_
} // namespace google

void init_logging(const std::string& name) {
#if !defined(_MSC_VER)
  // This trick can only be used on UNIX platforms
  if (!::google::glog_internal_namespace_::IsGoogleLoggingInitialized())
#endif
  {
    ::google::InitGoogleLogging(name.c_str());
#if !defined(_MSC_VER)
    // This is never defined on Windows
    ::google::InstallFailureSignalHandler();
#endif
  }
}

namespace spdl {
namespace {

namespace py = pybind11;

PYBIND11_MODULE(SPDL_FFMPEG_EXT_NAME, m) {
  m.def("init_logging", &init_logging);
  py::class_<Engine>(m, "Engine", py::module_local())
      .def(py::init([](size_t num_io_threads,
                       size_t num_decoding_threads,
                       size_t frame_queue_size) {
        return new Engine{
            num_io_threads, num_decoding_threads, frame_queue_size};
      }))
      .def(
          "enqueue",
          [](Engine& self,
             const std::string& url,
             const std::vector<double>& timestamps) {
            self.enqueue({url, timestamps});
          })
      .def("dequeue", &Engine::dequeue);
}
} // namespace
} // namespace spdl
