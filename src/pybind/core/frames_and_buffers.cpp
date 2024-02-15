#include <libspdl/core/buffers.h>
#include <libspdl/core/conversion.h>
#include <libspdl/core/frames.h>

#include <fmt/core.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace spdl::core {
namespace {
////////////////////////////////////////////////////////////////////////////////
// Array interface supplements
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
py::dict get_cuda_array_interface(CUDABuffer& b) {
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
// Trampoline class for registering abstract DecodedFrames
////////////////////////////////////////////////////////////////////////////////
class PyDecodedFrames : public DecodedFrames {
 public:
  using DecodedFrames::DecodedFrames;

  std::string get_media_format() const override {
    PYBIND11_OVERLOAD_PURE(std::string, DecodedFrames, get_media_format);
  }
  std::string get_media_type() const override {
    PYBIND11_OVERLOAD_PURE(std::string, DecodedFrames, get_media_type);
  }
};

class PyBuffer : public Buffer {
 public:
  using Buffer::Buffer;

  bool is_cuda() const override {
    PYBIND11_OVERLOAD_PURE(bool, Buffer, is_cuda);
  }
};
} // namespace

void register_frames_and_buffers(py::module& m) {
  auto _Buffer = py::class_<Buffer, PyBuffer, std::shared_ptr<Buffer>>(
      m, "Buffer", py::module_local());

  auto _CPUBuffer = py::class_<CPUBuffer>(m, "CPUBuffer", py::module_local());

#ifdef SPDL_USE_CUDA
  auto _CUDABuffer =
      py::class_<CUDABuffer>(m, "CUDABuffer", py::module_local());
#endif

  auto _DecodedFrames = py::
      class_<DecodedFrames, PyDecodedFrames, std::shared_ptr<DecodedFrames>>(
          m, "DecodedFrames", py::module_local());

  auto _FFmpegFrames =
      py::class_<FFmpegFrames>(m, "FFmpegFrames", py::module_local());

  auto _FFmpegAudioFrames =
      py::class_<FFmpegAudioFrames>(m, "FFmpegAudioFrames", py::module_local());

  auto _FFmpegVideoFrames =
      py::class_<FFmpegVideoFrames>(m, "FFmpegVideoFrames", py::module_local());

  _CPUBuffer
      .def_property_readonly(
          "channel_last",
          [](const CPUBuffer& self) { return self.channel_last; })
      .def_property_readonly(
          "ndim", [](const CPUBuffer& self) { return self.shape.size(); })
      .def_property_readonly(
          "shape", [](const CPUBuffer& self) { return self.shape; })
      .def_property_readonly("is_cuda", &CPUBuffer::is_cuda)
      .def("get_array_interface", [](CPUBuffer& self) {
        return get_array_interface(self);
      });

#ifdef SPDL_USE_CUDA
  _CUDABuffer
      .def_property_readonly(
          "channel_last",
          [](const CUDABuffer& self) { return self.channel_last; })
      .def_property_readonly(
          "ndim", [](const CUDABuffer& self) { return self.shape.size(); })
      .def_property_readonly(
          "shape", [](const CUDABuffer& self) { return self.shape; })
      .def_property_readonly("is_cuda", &CUDABuffer::is_cuda)
      .def(
          "get_cuda_array_interface",
          [](CUDABuffer& self) { return get_cuda_array_interface(self); });
#endif

  _FFmpegAudioFrames
      .def_property_readonly(
          "media_format", &FFmpegAudioFrames::get_media_format)
      .def_property_readonly("media_type", &FFmpegAudioFrames::get_media_type)
      .def_property_readonly("is_cuda", &FFmpegAudioFrames::is_cuda)
      .def_property_readonly("num_frames", &FFmpegAudioFrames::get_num_frames)
      .def_property_readonly("sample_rate", &FFmpegAudioFrames::get_sample_rate)
      .def_property_readonly(
          "num_channels", &FFmpegAudioFrames::get_num_channels)
      .def("__len__", &FFmpegAudioFrames::get_num_frames);

  _FFmpegVideoFrames
      .def_property_readonly(
          "media_format", &FFmpegVideoFrames::get_media_format)
      .def_property_readonly("media_type", &FFmpegVideoFrames::get_media_type)
      .def_property_readonly("is_cuda", &FFmpegVideoFrames::is_cuda)
      .def_property_readonly("num_frames", &FFmpegVideoFrames::get_num_frames)
      .def_property_readonly("num_planes", &FFmpegVideoFrames::get_num_planes)
      .def_property_readonly("width", &FFmpegVideoFrames::get_width)
      .def_property_readonly("height", &FFmpegVideoFrames::get_height)
      .def("__len__", &FFmpegVideoFrames::get_num_frames)
      .def(
          "__getitem__",
          [](const FFmpegVideoFrames& self, const py::slice& slice) {
            py::ssize_t start = 0, stop = 0, step = 0, len = 0;
            if (!slice.compute(
                    self.frames.size(), &start, &stop, &step, &len)) {
              throw py::error_already_set();
            }
            return self.slice(
                static_cast<int>(start),
                static_cast<int>(stop),
                static_cast<int>(step));
          });

  m.def("convert_frames", &convert_audio_frames);
  m.def("convert_frames", &convert_video_frames);
}
} // namespace spdl::core
