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

std::tuple<std::vector<size_t>, bool>
get_shape(nvjpegOutputFormat_t out_fmt, size_t height, size_t width) {
  switch (out_fmt) {
    // TODO: Support NVJPEG_OUTPUT_YUV?
    case NVJPEG_OUTPUT_RGB:
      [[fallthrough]];
    case NVJPEG_OUTPUT_BGR:
      return {{3, height, width}, false};
    case NVJPEG_OUTPUT_RGBI:
      [[fallthrough]];
    case NVJPEG_OUTPUT_BGRI:
      return {{height, width, 3}, true};
    case NVJPEG_OUTPUT_Y:
      return {{1, height, width}, false};
    default:
      // It should be already handled by `get_nvjpeg_output_format`
      SPDL_FAIL_INTERNAL(
          fmt::format("Unexpected output format: {}", to_string(out_fmt)));
  }
}

std::tuple<CUDABufferPtr, nvjpegImage_t> get_output(
    nvjpegOutputFormat_t out_fmt,
    size_t height,
    size_t width,
    const CUDAConfig& cuda_config) {
  auto [shape, interleaved] = get_shape(out_fmt, height, width);

  auto buffer = cuda_buffer(
      shape,
      cuda_config.device_index,
      cuda_config.stream,
      cuda_config.allocator);

  auto ptr = static_cast<uint8_t*>(buffer->data());

  nvjpegImage_t output;
  auto num_channels = interleaved ? shape[2] : shape[0];
  auto pitch = interleaved ? width * num_channels : width;
  for (int c = 0; c < num_channels; c++) {
    output.channel[c] = ptr;
    output.pitch[c] = pitch;
    ptr += pitch * height;
  }

  return {std::move(buffer), output};
}

std::tuple<CUDABufferPtr, nvjpegImage_t, int, int> decode(
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

  auto [buffer, image] = get_output(fmt, heights[0], widths[0], cuda_config);

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
  return {std::move(buffer), image, widths[0], heights[0]};
}

} // namespace

CUDABufferPtr decode_image_nvjpeg(
    const std::string_view& data,
    const CUDAConfig cuda_config,
    int scale_width,
    int scale_height,
    const std::string& pix_fmt) {
  cudaStream_t cuda_stream = 0;

  auto out_fmt = get_nvjpeg_output_format(pix_fmt);

  ensure_cuda_initialized();
  set_cuda_primary_context(cuda_config.device_index);

  auto [buffer, decoded, src_width, src_height] =
      decode(data, out_fmt, cuda_config);

  if (scale_width > 0 && scale_height > 0) {
    auto [buffer2, resized] =
        get_output(out_fmt, scale_height, scale_width, cuda_config);

    resize_npp(
        out_fmt,
        decoded,
        src_width,
        src_height,
        resized,
        scale_width,
        scale_height);

    return std::move(buffer2);
  }

  return std::move(buffer);
}

} // namespace spdl::core::detail
