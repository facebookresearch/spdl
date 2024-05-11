#include <libspdl/coro/conversion.h>

#include "libspdl/coro/detail/executor.h"
#include "libspdl/coro/detail/future.h"

#include <libspdl/core/conversion.h>
#include <libspdl/core/cuda.h>

#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <fmt/core.h>

extern "C" {
#include <libavutil/frame.h>
}

namespace spdl::coro {

using spdl::core::InternalError;

////////////////////////////////////////////////////////////////////////////////
// Video/Image
////////////////////////////////////////////////////////////////////////////////
namespace {

template <MediaType media_type>
void check_consistency(const std::vector<AVFrame*>& frames)
  requires(media_type != MediaType::Audio)
{
  auto numel = frames.size();
  if (numel == 0) {
    SPDL_FAIL("No frame to convert to buffer.");
  }
  if constexpr (media_type == MediaType::Image) {
    if (numel != 1) {
      SPDL_FAIL_INTERNAL(fmt::format(
          "There must be exactly one frame to convert to buffer. Found: {}",
          numel));
    }
  }
  auto pix_fmt = static_cast<AVPixelFormat>(frames[0]->format);
  int height = frames[0]->height, width = frames[0]->width;
  for (auto* f : frames) {
    if (f->height != height || f->width != width) {
      SPDL_FAIL(fmt::format(
          "Cannot convert the frames as the frames do not have the same size. "
          "Reference WxH = {}x{}, found {}x{}.",
          height,
          width,
          f->height,
          f->width));
    }
    if (static_cast<AVPixelFormat>(f->format) != pix_fmt) {
      SPDL_FAIL(fmt::format(
          "Cannot convert the frames as the frames do not have the same pixel format."));
    }
  }
}

template <MediaType media_type>
CPUBufferPtr convert_video(const std::vector<AVFrame*>& frames)
  requires(media_type != MediaType::Audio)
{
  check_consistency<media_type>(frames);
  if (static_cast<AVPixelFormat>(frames[0]->format) == AV_PIX_FMT_CUDA) {
    SPDL_FAIL_INTERNAL("FFmpeg-native CUDA frames are not supported.");
  }
  auto buf = spdl::core::convert_video_frames_cpu(frames);
  if constexpr (media_type == MediaType::Image) {
    buf->shape.erase(buf->shape.begin()); // Trim the first dim
  }
  return buf;
}

template <MediaType media_type>
CPUBufferPtr convert_vision_frames(const FFmpegFramesPtr<media_type> frames)
  requires(media_type != MediaType::Audio)
{
  TRACE_EVENT(
      "decoding",
      "core::convert_vision_frames",
      perfetto::Flow::ProcessScoped(frames->get_id()));
  return convert_video<media_type>(frames->get_frames());
}
} // namespace

template <MediaType media_type>
FuturePtr async_convert_frames(
    std::function<void(CPUBufferPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    FFmpegFramesPtr<media_type> frames,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke(
      [=](FFmpegFramesPtr<media_type>&& frm)
          -> folly::coro::Task<CPUBufferPtr> {
        co_return convert_vision_frames<media_type>(std::move(frm));
      },
      std::move(frames));
  return detail::execute_task_with_callback(
      std::move(task),
      set_result,
      notify_exception,
      detail::get_demux_executor_high_prio(executor));
}

template FuturePtr async_convert_frames<MediaType::Video>(
    std::function<void(CPUBufferPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    FFmpegFramesPtr<MediaType::Video> frames,
    ThreadPoolExecutorPtr executor);

template FuturePtr async_convert_frames<MediaType::Image>(
    std::function<void(CPUBufferPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    FFmpegFramesPtr<MediaType::Image> frames,
    ThreadPoolExecutorPtr executor);

template <MediaType media_type>
FuturePtr async_convert_frames_cuda(
    std::function<void(CUDABufferPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    FFmpegFramesPtr<media_type> frames,
    int cuda_device_index,
    const uintptr_t cuda_stream,
    const std::optional<cuda_allocator>& cuda_allocator,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke(
      [=](FFmpegFramesPtr<media_type>&& frm)
          -> folly::coro::Task<CUDABufferPtr> {
        co_return convert_to_cuda(
            convert_vision_frames<media_type>(std::move(frm)),
            cuda_device_index,
            cuda_stream,
            cuda_allocator);
      },
      std::move(frames));
  return detail::execute_task_with_callback(
      std::move(task),
      set_result,
      notify_exception,
      detail::get_demux_executor_high_prio(executor));
}

template FuturePtr async_convert_frames_cuda<MediaType::Video>(
    std::function<void(CUDABufferPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    FFmpegFramesPtr<MediaType::Video> frames,
    int cuda_device_index,
    const uintptr_t cuda_stream,
    const std::optional<cuda_allocator>& cuda_allocator,
    ThreadPoolExecutorPtr executor);

template FuturePtr async_convert_frames_cuda<MediaType::Image>(
    std::function<void(CUDABufferPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    FFmpegFramesPtr<MediaType::Image> frames,
    int cuda_device_index,
    const uintptr_t cuda_stream,
    const std::optional<cuda_allocator>& cuda_allocator,
    ThreadPoolExecutorPtr executor);

////////////////////////////////////////////////////////////////////////////////
// Batch Image
////////////////////////////////////////////////////////////////////////////////
namespace {
std::vector<AVFrame*> merge_frames(
    const std::vector<FFmpegFramesPtr<MediaType::Image>>& batch) {
  std::vector<AVFrame*> ret;
  ret.reserve(batch.size());
  for (auto& frame : batch) {
    if (frame->get_num_frames() != 1) {
      SPDL_FAIL_INTERNAL(
          "Unexpected number of frames are found in one of the image frames.");
    }
    ret.push_back(frame->get_frames()[0]);
  }
  return ret;
}

CPUBufferPtr convert_batch_image_frames(
    const std::vector<FFmpegFramesPtr<MediaType::Image>>& batch) {
  TRACE_EVENT("decoding", "core::convert_batch_image_frames");
  return convert_video<MediaType::Video>(merge_frames(batch));
}
} // namespace

FuturePtr async_batch_convert_frames(
    std::function<void(CPUBufferPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::vector<FFmpegFramesPtr<MediaType::Image>>&& frames,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke(
      [=](std::vector<FFmpegFramesPtr<MediaType::Image>>&& frms)
          -> folly::coro::Task<CPUBufferPtr> {
        co_return convert_batch_image_frames(frms);
      },
      // Pass the ownership of FramePtrs to executor thread, so that they are
      // deallocated there, instead of the main thread.
      std::move(frames));
  return detail::execute_task_with_callback(
      std::move(task),
      set_result,
      notify_exception,
      detail::get_demux_executor_high_prio(executor));
}

FuturePtr async_batch_convert_frames_cuda(
    std::function<void(CUDABufferPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::vector<FFmpegFramesPtr<MediaType::Image>>&& frames,
    int cuda_device_index,
    const uintptr_t cuda_stream,
    const std::optional<cuda_allocator>& cuda_allocator,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke(
      [=](std::vector<FFmpegFramesPtr<MediaType::Image>>&& frms)
          -> folly::coro::Task<CUDABufferPtr> {
        co_return convert_to_cuda(
            convert_batch_image_frames(frms),
            cuda_device_index,
            cuda_stream,
            cuda_allocator);
        ;
      },
      // Pass the ownership of FramePtrs to executor thread, so that they are
      // deallocated there, instead of the main thread.
      std::move(frames));
  return detail::execute_task_with_callback(
      std::move(task),
      set_result,
      notify_exception,
      detail::get_demux_executor_high_prio(executor));
}

} // namespace spdl::coro
