#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <libspdl/buffers.h>
#include <libspdl/processors.h>
#include <libspdl/types.h>
#include <libspdl/utils.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstddef>

#ifndef SPDL_FFMPEG_EXT_NAME
#error SPDL_FFMPEG_EXT_NAME must be defined.
#endif

namespace py = pybind11;

namespace spdl {
namespace {

////////////////////////////////////////////////////////////////////////////////
// Array Interface
////////////////////////////////////////////////////////////////////////////////
py::dict get_array_interface(Buffer& b) {
  py::dict ret;
  ret["version"] = 3;
  ret["shape"] = py::tuple(py::cast(b.shape));
  ret["typestr"] = std::string("|u1");
  ret["data"] = std::tuple<size_t, bool>{(uintptr_t)b.data(), false};
  ret["strides"] = py::none();
  ret["descr"] = std::vector<std::tuple<std::string, std::string>>{{"", "|u1"}};
  return ret;
}

#ifdef SPDL_USE_CUDA
py::dict get_cuda_array_interface(Buffer& b) {
  py::dict ret;
  ret["version"] = 2;
  ret["shape"] = py::tuple(py::cast(b.shape));
  ret["typestr"] = std::string("|u1");
  ret["data"] = std::tuple<size_t, bool>{(uintptr_t)b.data(), false};
  ret["strides"] = py::none();
  ret["stream"] = b.get_cuda_stream();
  return ret;
}
#endif

////////////////////////////////////////////////////////////////////////////////
// FOLLY INITIALIZATION
////////////////////////////////////////////////////////////////////////////////
struct DoublePtr {
  char **p, **p_orig;
  DoublePtr(int argc) : p(new char*[argc]), p_orig(p) {}
  DoublePtr(const DoublePtr&) = delete;
  DoublePtr& operator=(const DoublePtr&) = delete;
  DoublePtr(DoublePtr&&) noexcept = delete;
  DoublePtr& operator=(DoublePtr&&) noexcept = delete;
  ~DoublePtr() {
    delete[] p_orig;
  }
};

folly::Init* FOLLY_INIT = nullptr;

void delete_folly_init() {
  delete FOLLY_INIT;
}

std::vector<std::string> init_folly_init(
    const std::string& prog,
    const std::vector<std::string>& orig_args) {
  int nargs = 1 + orig_args.size();
  DoublePtr args(nargs);
  args.p[0] = const_cast<char*>(prog.c_str());
  for (int i = 1; i < nargs; ++i) {
    args.p[i] = const_cast<char*>(orig_args[i - 1].c_str());
  }

  FOLLY_INIT = new folly::Init{&nargs, &args.p};
  Py_AtExit(delete_folly_init);

  std::vector<std::string> ret;
  for (int i = 0; i < nargs; ++i) {
    ret.emplace_back(args.p[i]);
  }
  return ret;
}

////////////////////////////////////////////////////////////////////////////////
// Misc
////////////////////////////////////////////////////////////////////////////////
std::optional<std::tuple<int, int>> get_frame_rate(
    const py::object& frame_rate) {
  if (frame_rate == py::none()) {
    return std::nullopt;
  }
  py::object Fraction = py::module_::import("fractions").attr("Fraction");
  py::object r = Fraction(frame_rate);
  return {{r.attr("numerator").cast<int>(), r.attr("denominator").cast<int>()}};
}
////////////////////////////////////////////////////////////////////////////////
// Registeration
////////////////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(SPDL_FFMPEG_EXT_NAME, m) {
  m.def("init_folly", &init_folly_init);
  m.def("get_ffmpeg_log_level", &get_ffmpeg_log_level);
  m.def("set_ffmpeg_log_level", &set_ffmpeg_log_level);
  m.def("clear_ffmpeg_cuda_context_cache", &clear_ffmpeg_cuda_context_cache);
  m.def(
      "create_cuda_context",
      &create_cuda_context,
      py::arg("index"),
      py::arg("use_primary_context") = false);

  py::class_<Frames>(m, "Frames", py::module_local())
      .def("to_buffer", &Frames::to_buffer)
      .def("__len__", [](const Frames& self) { return self.frames.size(); })
      .def(
          "__getitem__",
          [](const Frames& self, const py::slice& slice) {
            py::ssize_t start = 0, stop = 0, step = 0, len = 0;
            if (!slice.compute(
                    self.frames.size(), &start, &stop, &step, &len)) {
              throw py::error_already_set();
            }
            return self.slice(
                static_cast<int>(start),
                static_cast<int>(stop),
                static_cast<int>(step));
          })
      .def_property_readonly("width", &Frames::get_width)
      .def_property_readonly("height", &Frames::get_height)
      .def_property_readonly("sample_rate", &Frames::get_sample_rate)
      .def_property_readonly("num_samples", &Frames::get_num_samples);

  py::class_<Buffer>(m, "Buffer", py::module_local())
      .def_property_readonly(
          "channel_last", [](const Buffer& self) { return self.channel_last; })
      .def("is_cuda", &Buffer::is_cuda)
      .def(
          "get_array_interface",
          [](Buffer& self) { return get_array_interface(self); })
#ifdef SPDL_USE_CUDA
      .def(
          "get_cuda_array_interface",
          [](Buffer& self) { return get_cuda_array_interface(self); })
#endif
      ;

  m.def(
      "decode_video",
      [](const std::string& src,
         const std::vector<double>& timestamps,
         const std::optional<std::string>& format = std::nullopt,
         const std::optional<OptionDict>& format_options = std::nullopt,
         const int buffer_size = 8096,
         const std::optional<std::string>& decoder = std::nullopt,
         const std::optional<OptionDict>& decoder_options = std::nullopt,
         const int cuda_device_index = -1,
         const py::object& frame_rate = py::none(),
         const std::optional<int>& width = std::nullopt,
         const std::optional<int>& height = std::nullopt,
         const std::optional<std::string>& pix_fmt = std::nullopt) {
        return decode_video(
            src,
            timestamps,
            get_video_filter_description(
                get_frame_rate(frame_rate), width, height, pix_fmt),
            {format, format_options, buffer_size},
            {decoder, decoder_options, cuda_device_index});
      },
      py::arg("src"),
      py::arg("timestamps"),
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = 8096,
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("cuda_device_index") = -1,
      py::arg("frame_rate") = py::none(),
      py::arg("width") = py::none(),
      py::arg("height") = py::none(),
      py::arg("pix_fmt") = py::none());
}
} // namespace
} // namespace spdl
