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
template <MediaType media_type>
FuturePtr async_convert_frames(
    std::function<void(CPUBufferPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    FFmpegFramesPtr<media_type> frames,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke(
      [=](FFmpegFramesPtr<media_type>&& frm)
          -> folly::coro::Task<CPUBufferPtr> {
        co_return spdl::core::convert_frames<media_type>(std::move(frm));
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
            spdl::core::convert_frames<media_type>(std::move(frm)),
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
template <MediaType media_type>
FuturePtr async_batch_convert_frames(
    std::function<void(CPUBufferPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::vector<FFmpegFramesPtr<media_type>>&& frames,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke(
      [=](std::vector<FFmpegFramesPtr<media_type>>&& frms)
          -> folly::coro::Task<CPUBufferPtr> {
        co_return convert_frames(frms);
      },
      // Pass the ownership of FramePtrs to executor thread, so that they are
      // deallocated there, instead of the main thread.
      std::move(frames));
  return detail::execute_task_with_callback(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_demux_executor_high_prio(executor));
}

template FuturePtr async_batch_convert_frames(
    std::function<void(CPUBufferPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::vector<FFmpegFramesPtr<MediaType::Image>>&& frames,
    ThreadPoolExecutorPtr executor);

template <>
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
            convert_frames(frms),
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
      std::move(set_result),
      std::move(notify_exception),
      detail::get_demux_executor_high_prio(executor));
}

} // namespace spdl::coro
