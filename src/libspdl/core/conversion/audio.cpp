#include <libspdl/core/conversion.h>

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
    std::function<void(BufferWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    FFmpegAudioFramesWrapperPtr frames,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke(
      [](FFmpegAudioFramesPtr&& frm) -> folly::coro::Task<BufferWrapperPtr> {
        co_return wrap(convert_audio_frames(std::move(frm)));
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

} // namespace spdl::core
