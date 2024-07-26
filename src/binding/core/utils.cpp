#include <libspdl/core/types.h>
#include <libspdl/core/utils.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#ifdef SPDL_LOG_API_USAGE
#include <c10/util/Logging.h>
#endif

namespace nb = nanobind;

namespace spdl::core {

void register_utils(nb::module_& m) {
  m.def("get_ffmpeg_log_level", &get_ffmpeg_log_level);
  m.def("set_ffmpeg_log_level", &set_ffmpeg_log_level);
  m.def("register_avdevices", &register_avdevices);
  m.def("get_ffmpeg_filters", &get_ffmpeg_filters);

  m.def("is_cuda_available", []() {
    nb::gil_scoped_release g;
    return is_cuda_available();
  });
  m.def("is_nvcodec_available", []() {
    nb::gil_scoped_release g;
    return is_nvcodec_available();
  });

  m.def("init_glog", [](char const* name) {
    nb::gil_scoped_release g;
    init_glog(name);
  });

  m.def("log_api_usage", [](const std::string& name) {
#ifdef SPDL_LOG_API_USAGE
    C10_LOG_API_USAGE_ONCE(name);
#endif
  });
}

} // namespace spdl::core
