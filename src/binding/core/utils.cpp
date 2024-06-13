#include <libspdl/core/types.h>
#include <libspdl/core/utils.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace spdl::core {

void register_utils(nb::module_& m) {
  m.def("get_ffmpeg_log_level", &get_ffmpeg_log_level);
  m.def("set_ffmpeg_log_level", &set_ffmpeg_log_level);
  m.def("register_avdevices", &register_avdevices);

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
}

} // namespace spdl::core
