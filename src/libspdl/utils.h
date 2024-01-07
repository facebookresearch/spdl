#pragma once

namespace spdl {

int get_ffmpeg_log_level();

void set_ffmpeg_log_level(int);

void clear_ffmpeg_cuda_context_cache();

void create_cuda_context(int index, bool use_primary_context = false);

std::string get_video_filter_description(
    const std::optional<std::tuple<int, int>>& frame_rate,
    const std::optional<int>& width,
    const std::optional<int>& height,
    const std::optional<std::string>& pix_fmt);

std::string get_audio_filter_description(
    const std::optional<int>& sample_rate,
    const std::optional<int>& num_channels,
    const std::optional<std::string>& sample_fmt);

} // namespace spdl
