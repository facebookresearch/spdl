#include <libspdl/core/detail/ffmpeg/ctx_utils.h>
#include <libspdl/core/detail/ffmpeg/filter_graph.h>
#include <libspdl/core/utils.h>

#include <cstdint>

extern "C" {
#include <libavutil/log.h>
}

namespace spdl::core {

int get_ffmpeg_log_level() {
  return av_log_get_level();
}

void set_ffmpeg_log_level(int level) {
  av_log_set_level(level);
}

void clear_ffmpeg_cuda_context_cache() {
  detail::clear_cuda_context_cache();
}

void create_cuda_context(int index, bool use_primary_context) {
  detail::create_cuda_context(index, use_primary_context);
}

std::string get_video_filter_description(
    const std::optional<std::tuple<int, int>>& frame_rate,
    const std::optional<int>& width,
    const std::optional<int>& height,
    const std::optional<std::string>& pix_fmt) {
  return detail::get_video_filter_description(
      frame_rate, width, height, pix_fmt);
}

std::string get_audio_filter_description(
    const std::optional<int>& sample_rate,
    const std::optional<int>& num_channels,
    const std::optional<std::string>& sample_fmt) {
  return detail::get_audio_filter_description(
      sample_rate, num_channels, sample_fmt);
}

} // namespace spdl::core
