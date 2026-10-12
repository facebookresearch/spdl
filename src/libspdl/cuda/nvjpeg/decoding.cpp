/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/cuda/nvjpeg/decoding.h>

#include <libspdl/cuda/buffer.h>
#include <libspdl/cuda/types.h>
#include "libspdl/cuda/detail/utils.h"

#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"
#include "libspdl/cuda/nvjpeg/detail/utils.h"

#ifdef SPDL_USE_NPPI
#include "libspdl/cuda/npp/detail/resize.h"
#include "libspdl/cuda/npp/detail/utils.h"
#endif

#include <fmt/format.h>

namespace spdl::cuda {
namespace {

cudaStream_t as_cuda_stream(uintptr_t stream) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  return reinterpret_cast<cudaStream_t>(stream);
}

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
          fmt::format(
              "Unexpected output format: {}", detail::to_string(out_fmt)));
  }
}

bool validate_resize_dimensions(int scale_width, int scale_height) {
  const bool has_width = scale_width > 0;
  const bool has_height = scale_height > 0;
  if (has_width != has_height) {
    SPDL_FAIL("`scale_width` and `scale_height` must both be positive.");
  }
  return has_width;
}

std::tuple<CUDABufferPtr, detail::NVJPEGImageLayout> get_output(
    nvjpegOutputFormat_t out_fmt,
    size_t height,
    size_t width,
    const CUDAConfig& cuda_config,
    std::optional<size_t> batch_size = std::nullopt) {
  auto [num_channels, interleaved] = get_shape(out_fmt);

  auto buffer = [&](const size_t ch, bool interleaved_2) {
    return batch_size
        ? (interleaved_2
               ? cuda_buffer({*batch_size, height, width, ch}, cuda_config)
               : cuda_buffer({*batch_size, ch, height, width}, cuda_config))
        : (interleaved_2 ? cuda_buffer({height, width, ch}, cuda_config)
                         : cuda_buffer({ch, height, width}, cuda_config));
  }(num_channels, interleaved);

  return {
      std::move(buffer),
      detail::NVJPEGImageLayout{
          .width = width,
          .height = height,
          .num_channels = num_channels,
          .interleaved = interleaved}};
}

std::tuple<CUDABufferPtr, detail::NVJPEGImageLayout, nvjpegImage_t> decode(
    std::string_view data,
    nvjpegOutputFormat_t fmt,
    const CUDAConfig& cuda_config) {
  auto nvjpeg = detail::get_nvjpeg();

  // Note: Creation/destruction of nvjpegJpegState_t is thread-safe, however,
  // looking at the trace, it appears that they have internal locking mechanism
  // which make these operations as slow as several hudreds milliseconds in
  // multithread situation. So we use thread local.
  thread_local auto jpeg_state = detail::get_nvjpeg_jpeg_state(nvjpeg);

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
  nvjpegImage_t image{};
  detail::wrap_nvjpeg_image(buffer->data(), meta, image);

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
            as_cuda_stream(cuda_config.stream)),
        "Failed to decode an image.");
  }
  return {std::move(buffer), meta, image};
}

} // namespace

CUDABufferPtr decode_image_nvjpeg(
    const std::string_view& data,
    const CUDAConfig& cuda_config,
    int scale_width,
    int scale_height,
    const std::string& pix_fmt,
    bool sync) {
  const bool resize = validate_resize_dimensions(scale_width, scale_height);
  auto fmt = detail::get_nvjpeg_output_format(pix_fmt);

  detail::CUDAContextPushGuard context_guard{cuda_config.device_index};

  auto [buffer, src_meta, decoded] = decode(data, fmt, cuda_config);

  if (resize) {
#ifndef SPDL_USE_NPPI
    SPDL_FAIL(
        "Image resizing while decoding with NVJPEG reqreuires SPDL to be compiled with NPPI support.");
#else
    CUDABufferPtr buffer2;
    detail::NVJPEGImageLayout meta2{};
    detail::CUDAStreamSyncOnExceptionGuard cleanup_guard{cuda_config.stream};
    std::tie(buffer2, meta2) =
        get_output(fmt, scale_height, scale_width, cuda_config);
    if (!buffer2) {
      SPDL_FAIL_INTERNAL("NVJPEG output allocation returned a null buffer.");
    }
    nvjpegImage_t resized{};
    detail::wrap_nvjpeg_image(buffer2->data(), meta2, resized);

    detail::resize_npp(
        fmt,
        decoded,
        (int)src_meta.width,
        (int)src_meta.height,
        resized,
        scale_width,
        scale_height,
        cuda_config.stream,
        cuda_config.device_index,
        sync);

    if (!sync) {
      detail::retain_cuda_storage_dependencies(
          buffer2->storage, {buffer->storage});
    }

    return buffer2;
#endif
  }

  if (sync) {
    CHECK_CUDA(
        cudaStreamSynchronize(as_cuda_stream(cuda_config.stream)),
        "Failed to synchronize stream after NVJPEG decoding.");
  }

  return std::move(buffer);
}

CUDABufferPtr decode_image_nvjpeg(
    const std::vector<std::string_view>& dataset,
    const CUDAConfig& cuda_config,
    int scale_width,
    int scale_height,
    const std::string& pix_fmt,
    bool sync) {
  const auto batch_size = dataset.size();
  if (batch_size == 0) {
    SPDL_FAIL("No input is provided.");
  }
  // Batch decoding always produces one uniformly-sized output allocation and
  // always runs the NPP resize path. Unlike single-image decoding, it therefore
  // has no no-resize mode and requires explicit positive output dimensions.
  if (!validate_resize_dimensions(scale_width, scale_height)) {
    SPDL_FAIL("Both `scale_width` and `scale_height` must be specified.");
  }

#ifndef SPDL_USE_NPPI
  SPDL_FAIL(
      "Image resizing while decoding with NVJPEG reqreuires SPDL to be compiled with NPPI support.");
#else
  auto fmt = detail::get_nvjpeg_output_format(pix_fmt);

  detail::CUDAContextPushGuard context_guard{cuda_config.device_index};
  const NppStreamContext npp_context = detail::get_npp_stream_context(
      cuda_config.stream, cuda_config.device_index);

  auto [out_buffer, out_meta] =
      get_output(fmt, scale_height, scale_width, cuda_config, batch_size);
  if (!out_buffer) {
    SPDL_FAIL_INTERNAL("NVJPEG output allocation returned a null buffer.");
  }
  nvjpegImage_t out_wrapper{};
  std::vector<CUDAStoragePtr> source_storages;
  source_storages.reserve(batch_size);
  detail::CUDAStreamSyncOnExceptionGuard cleanup_guard{cuda_config.stream};

  for (size_t i = 0; i < batch_size; ++i) {
    auto [src_buffer, src_meta, decoded] = decode(dataset[i], fmt, cuda_config);
    source_storages.emplace_back(std::move(src_buffer->storage));

    detail::wrap_nvjpeg_image(out_buffer->data(), out_meta, out_wrapper, i);
    detail::resize_npp(
        fmt,
        decoded,
        (int)src_meta.width,
        (int)src_meta.height,
        out_wrapper,
        scale_width,
        scale_height,
        npp_context,
        false);
  }

  if (sync) {
    CHECK_CUDA(
        cudaStreamSynchronize(as_cuda_stream(cuda_config.stream)),
        "Failed to synchronize stream after batch NVJPEG decoding.");
  } else {
    // Intentionally copy this vector. If dependency-owner allocation throws,
    // the caller must retain the sole source-storage references until
    // cleanup_guard synchronizes the stream during unwinding.
    detail::retain_cuda_storage_dependencies(
        out_buffer->storage, source_storages);
  }

  return std::move(out_buffer);
#endif
}

} // namespace spdl::cuda
