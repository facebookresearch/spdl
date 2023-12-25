#include <libspdl/detail/ffmpeg/ctx_utils.h>
#include <libspdl/utils.h>

#include <cstdint>
extern "C" {
#include <libavutil/log.h>
}

namespace spdl {

int get_ffmpeg_log_level() {
  return av_log_get_level();
}

void set_ffmpeg_log_level(int level) {
  av_log_set_level(level);
}

void clear_ffmpeg_cuda_context_cache() {
  detail::clear_cuda_context_cache();
}

} // namespace spdl
