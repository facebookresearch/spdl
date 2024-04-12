#include "libspdl/core/detail/nvdec/converter.h"

#include "libspdl/core/detail/cuda.h"
#include "libspdl/core/detail/nvdec/color_conversion.h"
#include "libspdl/core/detail/nvdec/utils.h"
#include "libspdl/core/detail/tracing.h"

namespace spdl::core::detail {
namespace {

class NV12Passthrough : public Converter {
 public:
  using Converter::Converter;
  void convert(uint8_t* src_ptr, unsigned int src_pitch) override {
    auto cfg = CUDA_MEMCPY2D{
        .srcXInBytes = 0,
        .srcY = 0,
        .srcMemoryType = CU_MEMORYTYPE_DEVICE,
        .srcHost = nullptr,
        .srcDevice = (CUdeviceptr)src_ptr,
        .srcArray = nullptr,
        .srcPitch = src_pitch,

        .dstXInBytes = 0,
        .dstY = 0,
        .dstMemoryType = CU_MEMORYTYPE_DEVICE,
        .dstHost = nullptr,
        .dstDevice = (CUdeviceptr)buffer->get_next_frame(),
        .dstArray = nullptr,
        .dstPitch = buffer->pitch,

        .WidthInBytes = buffer->width_in_bytes,
        .Height = buffer->h,
    };
    TRACE_EVENT("nvdec", "cuMemcpy2DAsync");
    CHECK_CU(
        cuMemcpy2DAsync(&cfg, stream),
        "Failed to copy Y plane from decoder output surface.");
  }
};

using ColorConversionFunc =
    void (*)(CUstream, uint8_t*, int, uint8_t*, int, int, int, int);

template <ColorConversionFunc Fn>
class NV12ToRGB : public Converter {
  unsigned char matrix_coeff;

 public:
  NV12ToRGB(
      CUstream stream,
      CUDABuffer2DPitch* buffer,
      unsigned char matrix_coeff_)
      : Converter(stream, buffer), matrix_coeff(matrix_coeff_){};
  void convert(uint8_t* src_ptr, unsigned int src_pitch) override {
    Fn(stream,
       src_ptr,
       src_pitch,
       buffer->get_next_frame(),
       buffer->pitch,
       buffer->w,
       buffer->h,
       matrix_coeff);
  };
};
} // namespace

Converter::Converter(CUstream s, CUDABuffer2DPitch* b) : stream(s), buffer(b) {}

std::unique_ptr<Converter> get_converter(
    CUstream stream,
    CUDABuffer2DPitch* buffer,
    const CUVIDDECODECREATEINFO* param,
    unsigned char matrix_coeff,
    const std::optional<std::string>& pix_fmt) {
  // NV12
  if (param->OutputFormat == cudaVideoSurfaceFormat_NV12) {
    // Source memory layout
    //
    // <---pitch--->
    // <-width->
    // ┌────────┬───┐  ▲
    // │ YYYYYY │   │ height
    // │ YYYYYY │   │  ▼
    // ├────────┤   │  ▲
    // │ UVUVUV |   │ height / 2
    // └────────┴───┘  ▼
    //

    if (!pix_fmt) {
      bool channel_last = false;
      size_t c = 1, bpp = 1;
      auto w = param->ulTargetWidth, h = param->ulTargetHeight;
      buffer->allocate(c, h + h / 2, w, bpp, channel_last);
      return std::unique_ptr<Converter>(new NV12Passthrough{stream, buffer});
    }
    auto pix_fmt_val = pix_fmt.value();
    if (pix_fmt_val == "rgba") {
      bool channel_last = false;
      size_t c = 4, bpp = 1;
      auto w = param->ulTargetWidth, h = param->ulTargetHeight;
      buffer->allocate(c, h, w, bpp, channel_last);
      return std::unique_ptr<Converter>(
          new NV12ToRGB<nv12_to_planar_rgba>{stream, buffer, matrix_coeff});
    }
    if (pix_fmt_val == "bgra") {
      bool channel_last = false;
      size_t c = 4, bpp = 1;
      auto w = param->ulTargetWidth, h = param->ulTargetHeight;
      buffer->allocate(c, h, w, bpp, channel_last);
      return std::unique_ptr<Converter>(
          new NV12ToRGB<nv12_to_planar_bgra>{stream, buffer, matrix_coeff});
    }
    SPDL_FAIL(fmt::format(
        "Unsupported pixel format: {}. Supported formats are 'rgba', 'bgra'.",
        pix_fmt_val));
  }

  SPDL_FAIL(fmt::format(
      "Conversion is not implemented for {}",
      get_surface_format_name(param->OutputFormat)));
}

} // namespace spdl::core::detail
