#include <libspdl/core/conversion.h>

#include "libspdl/core/detail/executor.h"
#include "libspdl/core/detail/future.h"
#include "libspdl/core/detail/logging.h"

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// Async - FFmpeg
////////////////////////////////////////////////////////////////////////////////
template <MediaType media_type, bool cpu_only>
FuturePtr async_convert_frames(
    std::function<void(BufferWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    FFmpegFramesWrapperPtr<media_type> frames,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke(
      [=](FFmpegFramesWrapperPtr<media_type>&& frm)
          -> folly::coro::Task<BufferWrapperPtr> {
        if constexpr (media_type == MediaType::Audio) {
          co_return wrap(convert_audio_frames(std::move(frm)));
        } else {
          co_return wrap(
              convert_vision_frames<media_type, cpu_only>(std::move(frm)));
        }
      },
      // Pass the ownership of FramePtr to executor thread, so that it is
      // deallocated there, instead of the main thread.
      wrap<media_type, FFmpegFramesPtr>(frames->unwrap()));
  return detail::execute_task_with_callback(
      std::move(task),
      set_result,
      notify_exception,
      detail::get_demux_executor_high_prio(executor));
}

template FuturePtr async_convert_frames<MediaType::Audio, true>(
    std::function<void(BufferWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    FFmpegFramesWrapperPtr<MediaType::Audio> frames,
    ThreadPoolExecutorPtr executor);

template FuturePtr async_convert_frames<MediaType::Video, true>(
    std::function<void(BufferWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    FFmpegFramesWrapperPtr<MediaType::Video> frames,
    ThreadPoolExecutorPtr executor);

template FuturePtr async_convert_frames<MediaType::Video, false>(
    std::function<void(BufferWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    FFmpegFramesWrapperPtr<MediaType::Video> frames,
    ThreadPoolExecutorPtr executor);

template FuturePtr async_convert_frames<MediaType::Image, true>(
    std::function<void(BufferWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    FFmpegFramesWrapperPtr<MediaType::Image> frames,
    ThreadPoolExecutorPtr executor);

template FuturePtr async_convert_frames<MediaType::Image, false>(
    std::function<void(BufferWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    FFmpegFramesWrapperPtr<MediaType::Image> frames,
    ThreadPoolExecutorPtr executor);

namespace {
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

template <bool cpu_only>
FuturePtr async_batch_convert_frames(
    std::function<void(BufferWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::vector<FFmpegImageFramesWrapperPtr> frames,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke(
      [=](std::vector<FFmpegImageFramesWrapperPtr>&& frms)
          -> folly::coro::Task<BufferWrapperPtr> {
        co_return wrap(convert_batch_image_frames<cpu_only>(frms));
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

template FuturePtr async_batch_convert_frames<true>(
    std::function<void(BufferWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::vector<FFmpegImageFramesWrapperPtr> frames,
    ThreadPoolExecutorPtr executor);

template FuturePtr async_batch_convert_frames<false>(
    std::function<void(BufferWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::vector<FFmpegImageFramesWrapperPtr> frames,
    ThreadPoolExecutorPtr executor);

////////////////////////////////////////////////////////////////////////////////
// Async - NVDEC
////////////////////////////////////////////////////////////////////////////////
template <MediaType media_type>
FuturePtr async_convert_nvdec_frames(
    std::function<void(CUDABuffer2DPitchPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    NvDecFramesWrapperPtr<media_type> frames,
    ThreadPoolExecutorPtr executor) {
  auto task =
      folly::coro::co_invoke([=]() -> folly::coro::Task<CUDABuffer2DPitchPtr> {
        // CUDABuffer2DPitchPtr uses shared_ptr, and the CUDA buffer is shared
        // among Frames class and Buffer class, so unlike CPU case, we do not
        // need to deallocate here.
        co_return convert_nvdec_frames<media_type>(frames);
      });
  return detail::execute_task_with_callback(
      std::move(task),
      set_result,
      notify_exception,
      detail::get_demux_executor_high_prio(executor));
}

template FuturePtr async_convert_nvdec_frames(
    std::function<void(CUDABuffer2DPitchPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    NvDecFramesWrapperPtr<MediaType::Video> frames,
    ThreadPoolExecutorPtr executor);

template FuturePtr async_convert_nvdec_frames(
    std::function<void(CUDABuffer2DPitchPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    NvDecFramesWrapperPtr<MediaType::Image> frames,
    ThreadPoolExecutorPtr executor);

FuturePtr async_batch_convert_nvdec_frames(
    std::function<void(CUDABuffer2DPitchPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::vector<NvDecImageFramesWrapperPtr> frames,
    ThreadPoolExecutorPtr executor) {
  auto task =
      folly::coro::co_invoke([=]() -> folly::coro::Task<CUDABuffer2DPitchPtr> {
        co_return convert_nvdec_batch_image_frames(frames);
      });
  return detail::execute_task_with_callback(
      std::move(task),
      set_result,
      notify_exception,
      detail::get_demux_executor_high_prio(executor));
}

////////////////////////////////////////////////////////////////////////////////
// Async - CPU to CUDA
////////////////////////////////////////////////////////////////////////////////
FuturePtr async_convert_to_cuda(
    std::function<void(BufferWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    BufferWrapperPtr buffer,
    int cuda_device_index,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke(
      [](BufferPtr&& b,
         int device_index) -> folly::coro::Task<BufferWrapperPtr> {
#ifndef SPDL_USE_CUDA
        SPDL_FAIL("SPDL is not compiled with CUDA support.");
#else
        co_return wrap(convert_to_cuda(std::move(b), device_index));
#endif
      },
      buffer->unwrap(),
      cuda_device_index);
  return detail::execute_task_with_callback(
      std::move(task),
      set_result,
      notify_exception,
      detail::get_demux_executor_high_prio(executor));
}
} // namespace spdl::core
