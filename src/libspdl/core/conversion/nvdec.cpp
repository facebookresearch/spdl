#include <libspdl/core/conversion.h>

#include "libspdl/core/detail/executor.h"
#include "libspdl/core/detail/ffmpeg/conversion.h"
#include "libspdl/core/detail/future.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#ifdef SPDL_USE_NVCODEC
#include <libspdl/core/detail/cuda.h>
#endif

extern "C" {
#include <libavutil/frame.h>
}

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// Batch Image
////////////////////////////////////////////////////////////////////////////////
namespace {
#ifdef SPDL_USE_NVCODEC
bool same_shape(const std::vector<size_t>& a, const std::vector<size_t>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (int i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

template <MediaType media_type>
void check_consistency(const std::vector<NvDecFramesPtr<media_type>>& frames) {
  auto numel = frames.size();
  if (numel == 0) {
    SPDL_FAIL("No frame to convert to buffer.");
  }
  auto& f0 = frames[0];
  auto pix_fmt = static_cast<AVPixelFormat>(f0->media_format);
  int device_index = f0->buffer->device_index;
  auto shape = f0->buffer->get_shape();
  for (auto& f : frames) {
    if (auto shape_ = f->buffer->get_shape(); !same_shape(shape, shape_)) {
      SPDL_FAIL(fmt::format(
          "Cannot convert the frames as the frames do not have the same size."));
    }
    if (static_cast<AVPixelFormat>(f->media_format) != pix_fmt) {
      SPDL_FAIL(fmt::format(
          "Cannot convert the frames as the frames do not have the same pixel format."));
    }
    if (device_index != f->buffer->device_index) {
      SPDL_FAIL(fmt::format(
          "Cannot convert the frames as the frames are not on the same device index."));
    }
  }
}
#endif

CUDABuffer2DPitchPtr convert_nvdec_batch_image_frames(
    const std::vector<NvDecImageFramesPtr>& batch_frames) {
#ifndef SPDL_USE_NVCODEC
  SPDL_FAIL("SPDL is not compiled with NVDEC support.");
#else
  TRACE_EVENT("decoding", "core::convert_nvdec_batch_image_frames");
  check_consistency(batch_frames);
  auto& buf0 = batch_frames[0]->buffer;

  detail::set_cuda_primary_context(buf0->device_index);
  auto ret = std::make_shared<CUDABuffer2DPitch>(
      buf0->device_index, batch_frames.size());
  ret->allocate(buf0->c, buf0->h, buf0->w, buf0->bpp, buf0->channel_last);

  cudaStream_t stream = 0;
  for (auto& frame : batch_frames) {
    auto& buf = frame->buffer;
    CHECK_CUDA(
        cudaMemcpy2DAsync(
            ret->get_next_frame(),
            ret->pitch,
            (void*)buf->p,
            buf->pitch,
            buf->width_in_bytes,
            buf->h,
            cudaMemcpyDefault,
            stream),
        "Failed to launch cudaMemcpy2DAsync.");
    ret->n += 1;
  }
  CHECK_CUDA(
      cudaStreamSynchronize(stream),
      "Failed to synchronize the stream after copying the data.");
  return ret;
#endif
}
} // namespace

FuturePtr async_batch_convert_nvdec_frames(
    std::function<void(CUDABuffer2DPitchPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::vector<NvDecImageFramesPtr>&& frames,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke(
      [frames =
           std::move(frames)]() -> folly::coro::Task<CUDABuffer2DPitchPtr> {
        co_return convert_nvdec_batch_image_frames(std::move(frames));
      });
  return detail::execute_task_with_callback(
      std::move(task),
      set_result,
      notify_exception,
      detail::get_demux_executor_high_prio(executor));
}

} // namespace spdl::core
