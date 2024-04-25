#include <libspdl/core/conversion.h>

#include "libspdl/core/conversion/cuda.h"
#include "libspdl/core/detail/executor.h"
#include "libspdl/core/detail/ffmpeg/conversion.h"
#include "libspdl/core/detail/future.h"
#include "libspdl/core/detail/tracing.h"

namespace spdl::core {
BufferPtr convert_audio_frames(const FFmpegAudioFramesPtr frames) {
  TRACE_EVENT(
      "decoding",
      "core::convert_audio_frames",
      perfetto::Flow::ProcessScoped(frames->get_id()));
  return detail::convert_audio_frames(frames.get());
}

template <>
FuturePtr async_convert_frames(
    std::function<void(BufferPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    FFmpegAudioFramesPtr frames,
    const std::optional<int>& cuda_device_index,
    const uintptr_t cuda_stream,
    const std::optional<cuda_allocator_fn>& cuda_allocator,
    const std::optional<cuda_deleter_fn>& cuda_deleter,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke(
      [=](FFmpegAudioFramesPtr&& frm) -> folly::coro::Task<BufferPtr> {
        auto ret = convert_audio_frames(std::move(frm));
        if (cuda_device_index) {
          ret = convert_to_cuda(
              std::move(ret),
              *cuda_device_index,
              cuda_stream,
              cuda_allocator,
              cuda_deleter);
        }
        co_return std::move(ret);
      },
      std::move(frames));
  return detail::execute_task_with_callback(
      std::move(task),
      set_result,
      notify_exception,
      detail::get_demux_executor_high_prio(executor));
}

} // namespace spdl::core
