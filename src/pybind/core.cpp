#include <libspdl/core/buffers.h>
#include <libspdl/core/conversion.h>
#include <libspdl/core/decoding.h>
#include <libspdl/core/interface/basic.h>
#include <libspdl/core/interface/mmap.h>
#include <libspdl/core/utils.h>

#include <fmt/core.h>
#include <folly/init/Init.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifndef SPDL_FFMPEG_EXT_NAME
#error SPDL_FFMPEG_EXT_NAME must be defined.
#endif

namespace py = pybind11;

namespace spdl::core {
namespace {

////////////////////////////////////////////////////////////////////////////////
// Array Interface
////////////////////////////////////////////////////////////////////////////////
std::string get_typestr(Buffer& b) {
  const auto key = [&]() {
    switch (b.elem_class) {
      case ElemClass::UInt:
        return "u";
      case ElemClass::Int:
        return "i";
      case ElemClass::Float:
        return "f";
    }
  }();
  return fmt::format("|{}{}", key, b.depth);
}

py::dict get_array_interface(Buffer& b) {
  auto typestr = get_typestr(b);
  py::dict ret;
  ret["version"] = 3;
  ret["shape"] = py::tuple(py::cast(b.shape));
  ret["typestr"] = typestr;
  ret["data"] = std::tuple<size_t, bool>{(uintptr_t)b.data(), false};
  ret["strides"] = py::none();
  ret["descr"] =
      std::vector<std::tuple<std::string, std::string>>{{"", typestr}};
  return ret;
}

#ifdef SPDL_USE_CUDA
py::dict get_cuda_array_interface(Buffer& b) {
  auto typestr = get_typestr(b);
  py::dict ret;
  ret["version"] = 2;
  ret["shape"] = py::tuple(py::cast(b.shape));
  ret["typestr"] = typestr;
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
  if (frame_rate.is(py::none())) {
    return std::nullopt;
  }
  py::object Fraction = py::module_::import("fractions").attr("Fraction");
  py::object r = Fraction(frame_rate);
  return {{r.attr("numerator").cast<int>(), r.attr("denominator").cast<int>()}};
}

class PySourceAdoptor : public SourceAdoptor {
 public:
  using SourceAdoptor::SourceAdoptor;

  void* get(const std::string& url) override {
    PYBIND11_OVERLOAD_PURE(void*, SourceAdoptor, get, url);
  }
};

} // namespace

void register_pybind(py::module& m) {
  ////////////////////////////////////////////////////////////////////////////////
  // Class registerations
  ////////////////////////////////////////////////////////////////////////////////
  // Make sure classes are module_local.

  auto _Buffer = py::class_<Buffer>(m, "Buffer", py::module_local());
  auto _FrameContainer =
      py::class_<FrameContainer>(m, "FrameContainer", py::module_local());

  // SourceAdoptor is used by external libraries to provide customized source.
  // This registeration is global.
  // To reduce the possibilty of name colision, suffixing with `_SPDL_GLOBAL`.
  auto _SourceAdoptor = py::
      class_<SourceAdoptor, PySourceAdoptor, std::shared_ptr<SourceAdoptor>>(
          m, "SourceAdoptor_SPDL_GLOBAL");

  auto _BasicAdoptor =
      py::class_<BasicAdoptor, SourceAdoptor, std::shared_ptr<BasicAdoptor>>(
          m, "BasicAdoptor", py::module_local());

  auto _MMapAdoptor =
      py::class_<MMapAdoptor, SourceAdoptor, std::shared_ptr<MMapAdoptor>>(
          m, "MMapAdoptor", py::module_local());

  m.def("init_folly", &init_folly_init);
  m.def("get_ffmpeg_log_level", &get_ffmpeg_log_level);
  m.def("set_ffmpeg_log_level", &set_ffmpeg_log_level);
  m.def("clear_ffmpeg_cuda_context_cache", &clear_ffmpeg_cuda_context_cache);
  m.def(
      "create_cuda_context",
      &create_cuda_context,
      py::arg("index"),
      py::arg("use_primary_context") = false);

  _FrameContainer
      .def(
          "__len__",
          [](const FrameContainer& self) { return self.frames.size(); })
      .def(
          "__getitem__",
          [](const FrameContainer& self, const py::slice& slice) {
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
      .def("is_cuda", &FrameContainer::is_cuda)
      .def_property_readonly(
          "media_type",
          [](const FrameContainer& self) -> std::string {
            switch (self.type) {
              case MediaType::Audio:
                return "audio";
              case MediaType::Video:
                return "video";
            }
          })
      .def_property_readonly("format", &FrameContainer::get_format)
      .def_property_readonly("num_planes", &FrameContainer::get_num_planes)
      .def_property_readonly("width", &FrameContainer::get_width)
      .def_property_readonly("height", &FrameContainer::get_height)
      .def_property_readonly("sample_rate", &FrameContainer::get_sample_rate)
      .def_property_readonly("num_samples", &FrameContainer::get_num_samples);

  _Buffer
      .def_property_readonly(
          "channel_last", [](const Buffer& self) { return self.channel_last; })
      .def_property_readonly(
          "ndim", [](const Buffer& self) { return self.shape.size(); })
      .def_property_readonly(
          "shape", [](const Buffer& self) { return self.shape; })
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

  m.def("convert_frames", &convert_frames);

  _SourceAdoptor.def("get", &SourceAdoptor::get);

  _BasicAdoptor.def(
      py::init<
          const std::optional<std::string>&,
          const std::optional<std::string>&,
          const std::optional<OptionDict>&>(),
      py::arg("prefix") = py::none(),
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none());

  _MMapAdoptor.def(
      py::init<
          const std::optional<std::string>&,
          const std::optional<std::string>&,
          const std::optional<OptionDict>&,
          int>(),
      py::arg("prefix") = py::none(),
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = 8096);

  m.def(
      "decode_video",
      [](const std::string& src,
         const std::vector<std::tuple<double, double>>& timestamps,
         std::shared_ptr<SourceAdoptor> adoptor,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const int cuda_device_index,
         const py::object& frame_rate,
         const std::optional<int>& width,
         const std::optional<int>& height,
         const std::optional<std::string>& pix_fmt) {
        return decode_video(
            src,
            adoptor,
            timestamps,
            get_video_filter_description(
                get_frame_rate(frame_rate), width, height, pix_fmt),
            {decoder, decoder_options, cuda_device_index});
      },
      py::arg("src"),
      py::arg("timestamps"),
      py::arg("adoptor") = nullptr,
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("cuda_device_index") = -1,
      py::arg("frame_rate") = py::none(),
      py::arg("width") = py::none(),
      py::arg("height") = py::none(),
      py::arg("pix_fmt") = py::none());

  m.def(
      "decode_video",
      [](const std::string& src,
         const std::vector<std::tuple<double, double>>& timestamps,
         std::shared_ptr<SourceAdoptor> adoptor,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const int cuda_device_index,
         const std::string& filter_desc) {
        return decode_video(
            src,
            adoptor,
            timestamps,
            filter_desc,
            {decoder, decoder_options, cuda_device_index});
      },
      py::arg("src"),
      py::arg("timestamps"),
      py::arg("adoptor") = nullptr,
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("cuda_device_index") = -1,
      py::arg("filter_desc") = std::string());

  m.def(
      "decode_audio",
      [](const std::string& src,
         const std::vector<std::tuple<double, double>>& timestamps,
         std::shared_ptr<SourceAdoptor> adoptor,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const std::optional<int>& sample_rate,
         const std::optional<int>& num_channels,
         const std::optional<std::string>& sample_fmt) {
        return decode_audio(
            src,
            adoptor,
            timestamps,
            get_audio_filter_description(sample_rate, num_channels, sample_fmt),
            {decoder, decoder_options});
      },
      py::arg("src"),
      py::arg("timestamps"),
      py::arg("adoptor") = nullptr,
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("sample_rate") = py::none(),
      py::arg("num_channels") = py::none(),
      py::arg("sample_fmt") = py::none());

  m.def(
      "decode_audio",
      [](const std::string& src,
         const std::vector<std::tuple<double, double>>& timestamps,
         std::shared_ptr<SourceAdoptor> adoptor,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const std::string& filter_desc) {
        return decode_audio(
            src, adoptor, timestamps, filter_desc, {decoder, decoder_options});
      },
      py::arg("src"),
      py::arg("timestamps"),
      py::arg("adoptor") = nullptr,
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("filter_desc") = std::string());
}
} // namespace spdl::core
