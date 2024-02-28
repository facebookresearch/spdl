#include <libspdl/core/buffers.h>
#include <libspdl/core/conversion.h>
#include <libspdl/core/frames.h>
#include <libspdl/core/types.h>

#include <fmt/core.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace spdl::core {
namespace {
////////////////////////////////////////////////////////////////////////////////
// Array interface supplements
////////////////////////////////////////////////////////////////////////////////
std::string get_typestr(const ElemClass elem_class, size_t depth) {
  const auto key = [&]() {
    switch (elem_class) {
      case ElemClass::UInt:
        return "u";
      case ElemClass::Int:
        return "i";
      case ElemClass::Float:
        return "f";
    }
  }();
  return fmt::format("|{}{}", key, depth);
}

py::dict get_array_interface(Buffer& b) {
  auto typestr = get_typestr(b.elem_class, b.depth);
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
  auto typestr = get_typestr(b.elem_class, b.depth);
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

#ifdef SPDL_USE_NVDEC
py::dict get_cuda_array_interface(CUDABuffer2DPitch& b) {
  auto typestr = get_typestr(ElemClass::UInt, 1);
  py::dict ret;
  ret["version"] = 2;
  ret["shape"] = py::tuple(py::cast(b.get_shape()));
  ret["typestr"] = typestr;
  ret["data"] = std::tuple<size_t, bool>{(uintptr_t)b.p, false};
  auto hp = b.h * b.pitch;
  ret["strides"] = b.is_image
      ? py::tuple(py::cast(
            b.channel_last ? std::vector<size_t>{b.pitch, b.c * b.bpp, b.bpp}
                           : std::vector<size_t>{hp, b.pitch, b.bpp}))
      : py::tuple(py::cast(
            b.channel_last
                ? std::vector<size_t>{hp, b.pitch, b.c * b.bpp, b.bpp}
                : std::vector<size_t>{b.c * hp, hp, b.pitch, b.bpp}));
  ret["stream"] = py::none();
  return ret;
}
#endif

template <typename T>
std::string get_type_string(const T& self) {
  switch (self.get_media_type()) {
    case MediaType::Audio:
      return "audio";
    case MediaType::Video:
      return "video";
    case MediaType::Image:
      return "image";
  }
}

} // namespace

void register_frames_and_buffers(py::module& m) {
  auto _Buffer = py::class_<Buffer>(m, "Buffer", py::module_local());

  auto _CPUBuffer = py::class_<CPUBuffer>(m, "CPUBuffer", py::module_local());

#ifdef SPDL_USE_CUDA
  auto _CUDABuffer =
      py::class_<CUDABuffer>(m, "CUDABuffer", py::module_local());
#endif

#ifdef SPDL_USE_NVDEC
  auto _CUDABuffer2DPitch =
      py::class_<CUDABuffer2DPitch, std::shared_ptr<CUDABuffer2DPitch>>(
          m, "CUDABuffer2DPitch", py::module_local());
#endif

  auto _DecodedFrames =
      py::class_<DecodedFrames>(m, "DecodedFrames", py::module_local());

  auto _FFmpegAudioFrames = py::class_<FFmpegAudioFrames, DecodedFrames>(
      m, "FFmpegAudioFrames", py::module_local());

  auto _FFmpegVideoFrames = py::class_<FFmpegVideoFrames, DecodedFrames>(
      m, "FFmpegVideoFrames", py::module_local());

  auto _FFmpegImageFrames = py::class_<FFmpegImageFrames, DecodedFrames>(
      m, "FFmpegImageFrames", py::module_local());

#ifdef SPDL_USE_NVDEC
  auto _NvDecVideoFrames = py::class_<NvDecVideoFrames, DecodedFrames>(
      m, "NvDecVideoFrames", py::module_local());
#endif

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
      .def("get_cuda_array_interface", [](CUDABuffer& self) {
        return get_cuda_array_interface(self);
      });
#endif

#ifdef SPDL_USE_NVDEC
  _CUDABuffer2DPitch
      .def_property_readonly(
          "channel_last",
          [](const CUDABuffer2DPitch& self) { return self.channel_last; })
      .def_property_readonly(
          "ndim",
          [](const CUDABuffer2DPitch& self) { return self.get_shape().size(); })
      .def_property_readonly("shape", &CUDABuffer2DPitch::get_shape)
      .def_property_readonly(
          "is_cuda", [](const CUDABuffer2DPitch& self) { return true; })
      .def("get_cuda_array_interface", [](CUDABuffer2DPitch& self) {
        return get_cuda_array_interface(self);
      });
#endif

  _FFmpegAudioFrames
      .def_property_readonly("media_type", &get_type_string<FFmpegAudioFrames>)
      .def_property_readonly("is_cuda", &FFmpegAudioFrames::is_cuda)
      .def_property_readonly("num_frames", &FFmpegAudioFrames::get_num_frames)
      .def_property_readonly("sample_rate", &FFmpegAudioFrames::get_sample_rate)
      .def_property_readonly(
          "num_channels", &FFmpegAudioFrames::get_num_channels)
      .def("__len__", &FFmpegAudioFrames::get_num_frames);

  _FFmpegVideoFrames
      .def_property_readonly("media_type", &get_type_string<FFmpegVideoFrames>)
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
          })
      .def("__getitem__", [](const FFmpegVideoFrames& self, int i) {
        return self.slice(i);
      });

  _FFmpegImageFrames
      .def_property_readonly("media_type", &get_type_string<FFmpegImageFrames>)
      .def_property_readonly("is_cuda", &FFmpegImageFrames::is_cuda)
      .def_property_readonly("num_planes", &FFmpegImageFrames::get_num_planes)
      .def_property_readonly("width", &FFmpegImageFrames::get_width)
      .def_property_readonly("height", &FFmpegImageFrames::get_height);

#ifdef SPDL_USE_NVDEC
  _NvDecVideoFrames
      .def_property_readonly("media_type", &get_type_string<NvDecVideoFrames>)
      .def_property_readonly("is_cuda", &NvDecVideoFrames::is_cuda)
      .def_property_readonly(
          "device_index",
          [](const NvDecVideoFrames& self) { return self.device_index; })
      .def_property_readonly(
          "channel_last",
          [](const NvDecVideoFrames& self) {
            return self.buffer->channel_last;
          })
      .def_property_readonly(
          "ndim",
          [](const NvDecVideoFrames& self) {
            return self.buffer->get_shape().size();
          })
      .def_property_readonly(
          "shape",
          [](const NvDecVideoFrames& self) { return self.buffer->get_shape(); })
      .def(
          "get_cuda_array_interface",
          [](NvDecVideoFrames& self) {
            return get_cuda_array_interface(*self.buffer);
          })
      .def("__len__", [](const NvDecVideoFrames& self) {
        return self.buffer->get_shape()[0];
      });
#endif

  m.def("convert_frames", &convert_audio_frames);
  m.def("convert_frames", &convert_video_frames);
  m.def("convert_frames", &convert_image_frames);
  m.def("convert_frames", &convert_batch_image_frames);
  m.def("convert_frames", &convert_nvdec_video_frames);

  m.def("convert_to_cpu_buffer", &convert_audio_frames);
  m.def("convert_to_cpu_buffer", &convert_video_frames_to_cpu_buffer);
  m.def("convert_to_cpu_buffer", &convert_image_frames_to_cpu_buffer);
  m.def("convert_to_cpu_buffer", &convert_batch_image_frames_to_cpu_buffer);
}
} // namespace spdl::core
