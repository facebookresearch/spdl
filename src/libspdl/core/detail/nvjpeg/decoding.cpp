#include <libspdl/core/decoding.h>
#include <libspdl/core/detail/cuda.h>
#include <libspdl/core/detail/logging.h>

#include "libspdl/core/detail/nvjpeg/utils.h"
#include "libspdl/core/detail/tracing.h"

#ifdef SPDL_USE_NPPI
#include "libspdl/core/detail/npp/resize.h"
#endif

#include <fmt/format.h>
#include <folly/logging/xlog.h>

namespace spdl::core::detail {
namespace {

std::tuple<size_t, bool> get_shape(nvjpegOutputFormat_t out_fmt) {
  switch (out_fmt) {
    // TODO: Support NVJPEG_OUTPUT_YUV?
    case NVJPEG_OUTPUT_RGB:
      [[fallthrough]];
    case NVJPEG_OUTPUT_BGR:
      return {3, false};
    case NVJPEG_OUTPUT_RGBI:
      [[fallthrough]];
    case NVJPEG_OUTPUT_BGRI:
      return {3, true};
    case NVJPEG_OUTPUT_Y:
      return {1, false};
    default:
      // It should be already handled by `get_nvjpeg_output_format`
      SPDL_FAIL_INTERNAL(
          fmt::format("Unexpected output format: {}", to_string(out_fmt)));
  }
}

struct SizeMeta {
  size_t width;
  size_t height;
  size_t num_channels;
  bool interleaved;
};

std::tuple<CUDABufferPtr, SizeMeta> get_output(
    nvjpegOutputFormat_t out_fmt,
    size_t height,
    size_t width,
    const CUDAConfig& cuda_config) {
  auto [num_channels, interleaved] = get_shape(out_fmt);

  auto buffer = interleaved
      ? cuda_buffer({height, width, num_channels}, cuda_config)
      : cuda_buffer({num_channels, height, width}, cuda_config);

  return {
      std::move(buffer),
      SizeMeta{
          .width = width,
          .height = height,
          .num_channels = num_channels,
          .interleaved = interleaved}};
}

void wrap_buffer(CUDABufferPtr& buffer, SizeMeta meta, nvjpegImage_t& image) {
  auto ptr = static_cast<uint8_t*>(buffer->data());
  auto pitch = meta.interleaved ? meta.width * meta.num_channels : meta.width;
  for (int c = 0; c < meta.num_channels; c++) {
    image.channel[c] = ptr;
    image.pitch[c] = pitch;
    ptr += pitch * meta.height;
  }
}

std::tuple<CUDABufferPtr, SizeMeta, nvjpegImage_t> decode(
    std::string_view data,
    nvjpegOutputFormat_t fmt,
    const CUDAConfig& cuda_config) {
  auto nvjpeg = get_nvjpeg();

  // Note: Creation/destruction of nvjpegJpegState_t is thread-safe, however,
  // looking at the trace, it appears that they have internal locking mechanism
  // which make these operations as slow as several hudreds milliseconds in
  // multithread situation. So we use thread local.
  thread_local auto jpeg_state = get_nvjpeg_jpeg_state(nvjpeg);

  int num_components;
  nvjpegChromaSubsampling_t subsampling;
  thread_local int widths[NVJPEG_MAX_COMPONENT];
  thread_local int heights[NVJPEG_MAX_COMPONENT];
  {
    TRACE_EVENT("decoding", "nvjpegGetImageInfo");
    CHECK_NVJPEG(
        nvjpegGetImageInfo(
            nvjpeg,
            (const unsigned char*)data.data(),
            data.size(),
            &num_components,
            &subsampling,
            widths,
            heights),
        "Failed to fetch image information.");
  }

  auto [buffer, meta] = get_output(fmt, heights[0], widths[0], cuda_config);
  nvjpegImage_t image;
  wrap_buffer(buffer, meta, image);

  // Note: backend is not used by NVJPEG API when using nvjpegDecode().
  //
  // https://docs.nvidia.com/cuda/nvjpeg/index.html#decode-apisingle-phase
  // >> From CUDA 11 onwards, nvjpegDecode() picks the best available back-end
  // >> for a given image, user no longer has control on this. If there is a
  // >> need to select the back-end, then consider using nvjpegDecodeJpeg.
  // >> This is a new API added in CUDA 11 which allows user to control the
  // >> back-end.
  {
    TRACE_EVENT("decoding", "nvjpegDecode");
    CHECK_NVJPEG(
        nvjpegDecode(
            nvjpeg,
            jpeg_state.get(),
            (const unsigned char*)data.data(),
            data.size(),
            fmt,
            &image,
            (CUstream_st*)cuda_config.stream),
        "Failed to decode an image.");
  }
  return {std::move(buffer), meta, image};
}

} // namespace

CUDABufferPtr decode_image_nvjpeg(
    const std::string_view& data,
    const CUDAConfig cuda_config,
    int scale_width,
    int scale_height,
    const std::string& pix_fmt) {
  auto fmt = get_nvjpeg_output_format(pix_fmt);

  ensure_cuda_initialized();
  set_cuda_primary_context(cuda_config.device_index);

  auto [buffer, src_meta, decoded] = decode(data, fmt, cuda_config);

  if (scale_width > 0 && scale_height > 0) {
    auto [buffer2, meta2] =
        get_output(fmt, scale_height, scale_width, cuda_config);
    nvjpegImage_t resized;
    wrap_buffer(buffer2, meta2, resized);

    resize_npp(
        fmt,
        decoded,
        src_meta.width,
        src_meta.height,
        resized,
        scale_width,
        scale_height);

    return std::move(buffer2);
  }

  return std::move(buffer);
}

} // namespace spdl::core::detail
