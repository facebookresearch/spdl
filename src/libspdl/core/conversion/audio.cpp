#include <libspdl/core/conversion.h>

#include "libspdl/core/detail/ffmpeg/conversion.h"
#include "libspdl/core/detail/tracing.h"

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// Audio
////////////////////////////////////////////////////////////////////////////////
BufferPtr convert_audio_frames(const FFmpegAudioFramesWrapperPtr frames) {
  TRACE_EVENT(
      "decoding",
      "core::convert_audio_frames",
      perfetto::Flow::ProcessScoped(frames->get_id()));
  return detail::convert_audio_frames(frames->get_frames_ref().get());
}
} // namespace spdl::core
