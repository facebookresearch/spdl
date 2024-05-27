#include <libspdl/core/decoding.h>
#include <libspdl/core/detail/cuda.h>
#include <libspdl/core/detail/logging.h>

#include "libspdl/core/detail/nvjpeg/utils.h"
#include "libspdl/core/detail/tracing.h"

#include <fmt/format.h>
#include <folly/logging/xlog.h>

namespace spdl::core::detail {
namespace {

std::tuple<CUDABufferPtr, nvjpegImage_t> get_output(
    nvjpegOutputFormat_t out_fmt,
    size_t height,
    size_t width,
    int cuda_device_index,
    cudaStream_t cuda_stream,
    const std::optional<cuda_allocator>& cuda_allocator) {
  auto [shape, interleaved] = [&]() -> std::tuple<std::vector<size_t>, bool> {
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
  }();

  auto buffer = cuda_buffer(
      shape, cuda_device_index, (uintptr_t)cuda_stream, cuda_allocator);

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
} // namespace

CUDABufferPtr decode_image_nvjpeg(
    const std::string_view& data,
    int cuda_device_index,
    const std::string& pix_fmt,
    const std::optional<cuda_allocator>& cuda_allocator) {
  cudaStream_t cuda_stream = 0;

  auto out_fmt = get_nvjpeg_output_format(pix_fmt);

  ensure_cuda_initialized();
  set_cuda_primary_context(cuda_device_index);

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

  auto [buffer, output] = get_output(
      out_fmt,
      heights[0],
      widths[0],
      cuda_device_index,
      cuda_stream,
      cuda_allocator);

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
            out_fmt,
            &output,
            cuda_stream),
        "Failed to decode an image.");
  }
  return std::move(buffer);
}

} // namespace spdl::core::detail
