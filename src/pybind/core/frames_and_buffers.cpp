#include <libspdl/core/buffer.h>
#include <libspdl/core/conversion.h>
#include <libspdl/core/frames.h>
#include <libspdl/core/types.h>

#include <fmt/core.h>

#include <pybind11/functional.h>
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

} // namespace

void register_frames_and_buffers(py::module& m) {
  auto _Buffer = py::class_<Buffer, BufferPtr>(m, "Buffer", py::module_local());

  auto _CPUBuffer = py::class_<CPUBuffer>(m, "CPUBuffer", py::module_local());

  auto _CUDABuffer =
      py::class_<CUDABuffer>(m, "CUDABuffer", py::module_local());

  auto _CUDABuffer2DPitch =
      py::class_<CUDABuffer2DPitch, std::shared_ptr<CUDABuffer2DPitch>>(
          m, "CUDABuffer2DPitch", py::module_local());

  auto _FFmpegAudioFrames = py::class_<FFmpegAudioFrames, FFmpegAudioFramesPtr>(
      m, "FFmpegAudioFrames", py::module_local());

  auto _FFmpegVideoFrames = py::class_<FFmpegVideoFrames, FFmpegVideoFramesPtr>(
      m, "FFmpegVideoFrames", py::module_local());

  auto _FFmpegImageFrames = py::class_<FFmpegImageFrames, FFmpegImageFramesPtr>(
      m, "FFmpegImageFrames", py::module_local());

  auto _NvDecVideoFrames = py::class_<NvDecVideoFrames, NvDecVideoFramesPtr>(
      m, "NvDecVideoFrames", py::module_local());

  auto _NvDecImageFrames = py::class_<NvDecImageFrames, NvDecImageFramesPtr>(
      m, "NvDecImageFrames", py::module_local());

  _CPUBuffer
      .def_property_readonly(
          "channel_last",
          [](const CPUBuffer& self) { return self.channel_last; })
      .def_property_readonly(
          "ndim", [](const CPUBuffer& self) { return self.shape.size(); })
      .def_property_readonly(
          "shape", [](const CPUBuffer& self) { return self.shape; })
      .def_property_readonly("is_cuda", [](const CPUBuffer&) { return false; })
      .def_property_readonly("__array_interface__", [](CPUBuffer& self) {
        return get_array_interface(self);
      });

#ifdef SPDL_USE_CUDA
#define IF_CUDABUFFER_ENABLED(x) x
#else
#define IF_CUDABUFFER_ENABLED(x)                                         \
  [](const CUDABuffer&) {                                                \
    throw std::runtime_error("SPDL is not compiled with CUDA support."); \
  }
#endif

  _CUDABuffer
      .def_property_readonly(
          "channel_last", IF_CUDABUFFER_ENABLED([](const CUDABuffer& self) {
            return self.channel_last;
          }))
      .def_property_readonly(
          "ndim", IF_CUDABUFFER_ENABLED([](const CUDABuffer& self) {
            return self.shape.size();
          }))
      .def_property_readonly(
          "shape", IF_CUDABUFFER_ENABLED([](const CUDABuffer& self) {
            return self.shape;
          }))
      .def_property_readonly("is_cuda", [](const CUDABuffer&) { return true; })
      .def_property_readonly(
          "__cuda_array_interface__",
          IF_CUDABUFFER_ENABLED(
              [](CUDABuffer& self) { return get_cuda_array_interface(self); }));

#ifdef SPDL_USE_NVDEC
#define IF_CUDABUFFER2_ENABLED(x) x
#else
#define IF_CUDABUFFER2_ENABLED(x)                                         \
  [](const CUDABuffer2DPitch&) {                                          \
    throw std::runtime_error("SPDL is not compiled with NVDEC support."); \
  }
#endif

  _CUDABuffer2DPitch
      .def_property_readonly(
          "channel_last",
          IF_CUDABUFFER2_ENABLED(
              [](const CUDABuffer2DPitch& self) { return self.channel_last; }))
      .def_property_readonly(
          "ndim", IF_CUDABUFFER2_ENABLED([](const CUDABuffer2DPitch& self) {
            return self.get_shape().size();
          }))
      .def_property_readonly(
          "shape", IF_CUDABUFFER2_ENABLED(&CUDABuffer2DPitch::get_shape))
      .def_property_readonly(
          "is_cuda", IF_CUDABUFFER2_ENABLED([](const CUDABuffer2DPitch& self) {
            return true;
          }))
      .def_property_readonly(
          "__cuda_array_interface__",
          IF_CUDABUFFER2_ENABLED([](CUDABuffer2DPitch& self) {
            return get_cuda_array_interface(self);
          }));

  _FFmpegAudioFrames
      .def_property_readonly(
          "is_cuda", [](const FFmpegAudioFrames&) { return false; })
      .def_property_readonly("num_frames", &FFmpegAudioFrames::get_num_frames)
      .def_property_readonly("sample_rate", &FFmpegAudioFrames::get_sample_rate)
      .def_property_readonly(
          "num_channels", &FFmpegAudioFrames::get_num_channels)
      .def_property_readonly(
          "format", &FFmpegAudioFrames::get_media_format_name)
      .def("__len__", &FFmpegAudioFrames::get_num_frames)
      .def("__repr__", [](const FFmpegAudioFrames& self) {
        return fmt::format(
            "FFmpegAudioFrames<num_frames={}, sample_format={}, sample_rate={}, num_channels={}>",
            self.get_media_format_name(),
            self.get_num_frames(),
            self.get_sample_rate(),
            self.get_num_channels());
      });

  _FFmpegVideoFrames
      .def_property_readonly("is_cuda", &FFmpegVideoFrames::is_cuda)
      .def_property_readonly("num_frames", &FFmpegVideoFrames::get_num_frames)
      .def_property_readonly("num_planes", &FFmpegVideoFrames::get_num_planes)
      .def_property_readonly("width", &FFmpegVideoFrames::get_width)
      .def_property_readonly("height", &FFmpegVideoFrames::get_height)
      .def_property_readonly(
          "format", &FFmpegVideoFrames::get_media_format_name)
      .def("__len__", &FFmpegVideoFrames::get_num_frames)
      .def(
          "__getitem__",
          [](const FFmpegVideoFrames& self, const py::slice& slice) {
            py::ssize_t start = 0, stop = 0, step = 0, len = 0;
            if (!slice.compute(
                    self.get_num_frames(), &start, &stop, &step, &len)) {
              throw py::error_already_set();
            }
            return self.slice(
                static_cast<int>(start),
                static_cast<int>(stop),
                static_cast<int>(step));
          })
      .def(
          "__getitem__",
          [](const FFmpegVideoFrames& self, int i) { return self.slice(i); })
      .def("__repr__", [](const FFmpegVideoFrames& self) {
        return fmt::format(
            "FFmpegVideoFrames<num_frames={}, pixel_format={}, num_planes={}, width={}, height={}, is_cuda={}>",
            self.get_num_frames(),
            self.get_media_format_name(),
            self.get_num_planes(),
            self.get_width(),
            self.get_height(),
            self.is_cuda());
      });

  _FFmpegImageFrames
      .def_property_readonly("is_cuda", &FFmpegImageFrames::is_cuda)
      .def_property_readonly("num_planes", &FFmpegImageFrames::get_num_planes)
      .def_property_readonly("width", &FFmpegImageFrames::get_width)
      .def_property_readonly("height", &FFmpegImageFrames::get_height)
      .def_property_readonly(
          "format", &FFmpegImageFrames::get_media_format_name)
      .def("__repr__", [](const FFmpegImageFrames& self) {
        return fmt::format(
            "FFmpegImageFrames<pixel_format={}, num_planes={}, width={}, height={}, is_cuda={}>",
            self.get_media_format_name(),
            self.get_num_planes(),
            self.get_width(),
            self.get_height(),
            self.is_cuda());
      });

#ifdef SPDL_USE_NVDEC
#define IF_NVDECVIDEOFRAMES_ENABLED(x) x
#else
#define IF_NVDECVIDEOFRAMES_ENABLED(x)                                    \
  [](const NvDecVideoFrames&) {                                           \
    throw std::runtime_error("SPDL is not compiled with NVDEC support."); \
  }
#endif

  // TODO: Add __repr__
  _NvDecVideoFrames
      .def_property_readonly(
          "is_cuda", IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecVideoFrames&) {
            return true;
          }))
      .def_property_readonly(
          "channel_last",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecVideoFrames& self) {
            return self.buffer->channel_last;
          }))
      .def_property_readonly(
          "ndim", IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecVideoFrames& self) {
            return self.buffer->get_shape().size();
          }))
      .def_property_readonly(
          "shape",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecVideoFrames& self) {
            return self.buffer->get_shape();
          }))
      .def_property_readonly(
          "format",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecVideoFrames& self) {
            return self.get_media_format_name();
          }))
      .def_property_readonly(
          "__cuda_array_interface__",
          IF_NVDECVIDEOFRAMES_ENABLED([](NvDecVideoFrames& self) {
            return get_cuda_array_interface(*self.buffer);
          }))
      .def(
          "__len__",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecVideoFrames& self) {
            return self.buffer->get_shape()[0];
          }));

  // TODO: Add __repr__
  _NvDecImageFrames
      .def_property_readonly(
          "is_cuda", IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecImageFrames&) {
            return true;
          }))
      .def_property_readonly(
          "channel_last",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecImageFrames& self) {
            return self.buffer->channel_last;
          }))
      .def_property_readonly(
          "ndim", IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecImageFrames& self) {
            return self.buffer->get_shape().size();
          }))
      .def_property_readonly(
          "shape",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecImageFrames& self) {
            return self.buffer->get_shape();
          }))
      .def_property_readonly(
          "format",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecImageFrames& self) {
            return self.get_media_format_name();
          }))
      .def_property_readonly(
          "__cuda_array_interface__",
          IF_NVDECVIDEOFRAMES_ENABLED([](NvDecImageFrames& self) {
            return get_cuda_array_interface(*self.buffer);
          }))
      .def(
          "__len__",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecImageFrames& self) {
            return self.buffer->get_shape()[0];
          }));

  m.def("convert_to_buffer", &convert_audio_frames);
  m.def("convert_to_buffer", &convert_visual_frames<MediaType::Video>);
  m.def("convert_to_buffer", &convert_visual_frames<MediaType::Image>);
  m.def("convert_to_buffer", &convert_batch_image_frames);
  m.def("convert_to_buffer", &convert_nvdec_frames<MediaType::Video>);
  m.def("convert_to_buffer", &convert_nvdec_frames<MediaType::Image>);
  m.def("convert_to_buffer", &convert_nvdec_batch_image_frames);

  m.def("convert_to_cpu_buffer", &convert_audio_frames);
  m.def(
      "convert_to_cpu_buffer",
      &convert_visual_frames_to_cpu_buffer<MediaType::Video>);
  m.def(
      "convert_to_cpu_buffer",
      &convert_visual_frames_to_cpu_buffer<MediaType::Image>);
  m.def("convert_to_cpu_buffer", &convert_batch_image_frames_to_cpu_buffer);

  m.def(
      "convert_to_cpu_buffer_async",
      &convert_frames_to_cpu_buffer_async<MediaType::Audio>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("index") = py::none(),
      py::arg("executor") = nullptr);
  m.def(
      "convert_to_cpu_buffer_async",
      &convert_frames_to_cpu_buffer_async<MediaType::Video>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("index") = py::none(),
      py::arg("executor") = nullptr);
  m.def(
      "convert_to_cpu_buffer_async",
      &convert_frames_to_cpu_buffer_async<MediaType::Image>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("index") = py::none(),
      py::arg("executor") = nullptr);
}
} // namespace spdl::core
