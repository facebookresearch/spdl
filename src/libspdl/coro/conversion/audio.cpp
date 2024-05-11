#include <libspdl/coro/conversion.h>

#include "libspdl/coro/detail/executor.h"
#include "libspdl/coro/detail/future.h"

#include <libspdl/core/conversion.h>
#include <libspdl/core/cuda.h>

#include "libspdl/core/detail/tracing.h"

namespace spdl::coro {
CPUBufferPtr convert_audio_frames(
    const FFmpegFramesPtr<MediaType::Audio> frames) {
  TRACE_EVENT(
      "decoding",
      "core::convert_audio_frames",
      perfetto::Flow::ProcessScoped(frames->get_id()));
  return spdl::core::convert_audio_frames(frames.get());
}

template <>
FuturePtr async_convert_frames(
    std::function<void(CPUBufferPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    FFmpegFramesPtr<MediaType::Audio> frames,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke(
      [=](FFmpegFramesPtr<MediaType::Audio>&& frm)
          -> folly::coro::Task<CPUBufferPtr> {
        co_return convert_audio_frames(std::move(frm));
      },
      std::move(frames));
  return detail::execute_task_with_callback(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_demux_executor_high_prio(executor));
}

template <>
FuturePtr async_convert_frames_cuda(
    std::function<void(CUDABufferPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    FFmpegFramesPtr<MediaType::Audio> frames,
    int cuda_device_index,
    const uintptr_t cuda_stream,
    const std::optional<cuda_allocator>& cuda_allocator,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke(
      [=](FFmpegFramesPtr<MediaType::Audio>&& frm)
          -> folly::coro::Task<CUDABufferPtr> {
        co_return convert_to_cuda(
            convert_audio_frames(std::move(frm)),
            cuda_device_index,
            cuda_stream,
            cuda_allocator);
      },
      std::move(frames));
  return detail::execute_task_with_callback(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_demux_executor_high_prio(executor));
}

} // namespace spdl::coro
