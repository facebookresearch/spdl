#include <libspdl/core/conversion.h>

#include "libspdl/core/detail/executor.h"
#include "libspdl/core/detail/ffmpeg/conversion.h"
#include "libspdl/core/detail/future.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <fmt/core.h>

extern "C" {
#include <libavutil/frame.h>
}

namespace spdl::core {

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
BufferPtr convert_video(
    const std::vector<AVFrame*>& frames,
    const std::optional<int>& cuda_device_index)
  requires(media_type != MediaType::Audio)
{
  check_consistency<media_type>(frames);
  bool is_cuda =
      static_cast<AVPixelFormat>(frames[0]->format) == AV_PIX_FMT_CUDA;
  if (is_cuda) {
    if (!cuda_device_index) {
      SPDL_FAIL_INTERNAL(
          "CUDA frames are found, but cuda device index is not set.");
    }
    return detail::convert_video_frames_cuda(frames, cuda_device_index.value());
  }
  auto buf = detail::convert_video_frames_cpu(frames);
  if constexpr (media_type == MediaType::Image) {
    buf->shape.erase(buf->shape.begin()); // Trim the first dim
  }
  return buf;
}

template <MediaType media_type>
BufferPtr convert_vision_frames(const FFmpegFramesPtr<media_type> frames)
  requires(media_type != MediaType::Audio)
{
  TRACE_EVENT(
      "decoding",
      "core::convert_vision_frames",
      perfetto::Flow::ProcessScoped(frames->get_id()));
  return convert_video<media_type>(
      frames->get_frames(), frames->cuda_device_index);
}
} // namespace

template <MediaType media_type>
FuturePtr async_convert_frames(
    std::function<void(BufferWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    FFmpegFramesWrapperPtr<media_type> frames,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke(
      [](FFmpegFramesPtr<media_type>&& frm)
          -> folly::coro::Task<BufferWrapperPtr> {
        co_return wrap(convert_vision_frames<media_type>(std::move(frm)));
      },
      // Pass the ownership of FramePtr to executor thread, so that it is
      // deallocated there, instead of the main thread.
      frames->unwrap());
  return detail::execute_task_with_callback(
      std::move(task),
      set_result,
      notify_exception,
      detail::get_demux_executor_high_prio(executor));
}

template FuturePtr async_convert_frames<MediaType::Video>(
    std::function<void(BufferWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    FFmpegFramesWrapperPtr<MediaType::Video> frames,
    ThreadPoolExecutorPtr executor);

template FuturePtr async_convert_frames<MediaType::Image>(
    std::function<void(BufferWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    FFmpegFramesWrapperPtr<MediaType::Image> frames,
    ThreadPoolExecutorPtr executor);

////////////////////////////////////////////////////////////////////////////////
// Batch Image
////////////////////////////////////////////////////////////////////////////////
namespace {
std::vector<AVFrame*> merge_frames(
    const std::vector<FFmpegImageFramesWrapperPtr>& batch) {
  std::vector<AVFrame*> ret;
  ret.reserve(batch.size());
  for (auto& frame : batch) {
    auto& ref = frame->get_frames_ref();
    if (ref->get_num_frames() != 1) {
      SPDL_FAIL_INTERNAL(
          "Unexpected number of frames are found in one of the image frames.");
    }
    ret.push_back(ref->get_frames()[0]);
  }
  return ret;
}

BufferPtr convert_batch_image_frames(
    const std::vector<FFmpegImageFramesWrapperPtr>& batch) {
  TRACE_EVENT("decoding", "core::convert_batch_image_frames");
  return convert_video<MediaType::Video>(
      merge_frames(batch), batch[0]->get_frames_ref()->cuda_device_index);
}

std::vector<FFmpegImageFramesWrapperPtr> rewrap(
    std::vector<FFmpegImageFramesWrapperPtr>&& frames) {
  std::vector<FFmpegImageFramesWrapperPtr> ret;
  ret.reserve(frames.size());
  for (auto& frame : frames) {
    ret.emplace_back(wrap<MediaType::Image, FFmpegFramesPtr>(frame->unwrap()));
  }
  return ret;
}
} // namespace

FuturePtr async_batch_convert_frames(
    std::function<void(BufferWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::vector<FFmpegImageFramesWrapperPtr> frames,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke(
      [=](std::vector<FFmpegImageFramesWrapperPtr>&& frms)
          -> folly::coro::Task<BufferWrapperPtr> {
        co_return wrap(convert_batch_image_frames(frms));
      },
      // Pass the ownership of FramePtrs to executor thread, so that they are
      // deallocated there, instead of the main thread.
      rewrap(std::move(frames)));
  return detail::execute_task_with_callback(
      std::move(task),
      set_result,
      notify_exception,
      detail::get_demux_executor_high_prio(executor));
}

} // namespace spdl::core
