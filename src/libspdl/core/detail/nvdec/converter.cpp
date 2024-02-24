#include <libspdl/core/detail/nvdec/converter.h>

#include <libspdl/core/detail/cuda.h>
#include <libspdl/core/detail/nvdec/utils.h>
#include <libspdl/core/detail/tracing.h>

namespace spdl::core::detail {

CUDA_MEMCPY2D
nv12pass(CUDABuffer2DPitch* buffer, uint8_t* src_ptr, unsigned int src_pitch) {
  return CUDA_MEMCPY2D{
      .Height = buffer->h,
      .WidthInBytes = buffer->width_in_bytes,
      .dstArray = nullptr,
      .dstDevice = (CUdeviceptr)buffer->get_next_frame(),
      .dstHost = nullptr,
      .dstMemoryType = CU_MEMORYTYPE_DEVICE,
      .dstPitch = buffer->pitch,
      .dstXInBytes = 0,
      .dstY = 0,
      .srcArray = nullptr,
      .srcDevice = (CUdeviceptr)src_ptr,
      .srcHost = nullptr,
      .srcMemoryType = CU_MEMORYTYPE_DEVICE,
      .srcPitch = src_pitch,
      .srcXInBytes = 0,
      .srcY = 0,
  };
}

Converter::Converter(CUstream s, CUDABuffer2DPitch* b) : stream(s), buffer(b) {}

void NV12Passthrough::convert(uint8_t* src_ptr, unsigned int src_pitch) {
  auto cfg = nv12pass(buffer, src_ptr, src_pitch);
  TRACE_EVENT("nvdec", "cuMemcpy2DAsync");
  CHECK_CU(
      cuMemcpy2DAsync(&cfg, stream),
      "Failed to copy Y plane from decoder output surface.");
};

std::unique_ptr<Converter> get_converter(
    CUstream stream,
    CUDABuffer2DPitch* buffer,
    const CUVIDDECODECREATEINFO* param) {
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

    bool channel_last = false;
    size_t c = 1, bpp = 1;
    auto w = param->ulTargetWidth;
    auto h = param->ulTargetHeight;
    buffer->allocate(c, h + h / 2, w, bpp, channel_last);
    return std::unique_ptr<Converter>(new NV12Passthrough{stream, buffer});
  }

  SPDL_FAIL(fmt::format(
      "Conversion is not implemented for {}",
      get_surface_format_name(param->OutputFormat)));
}

} // namespace spdl::core::detail
