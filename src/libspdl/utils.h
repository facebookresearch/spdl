#pragma once

namespace spdl {

int get_ffmpeg_log_level();

void set_ffmpeg_log_level(int);

void clear_ffmpeg_cuda_context_cache();

void create_cuda_context(int index, bool use_primary_context = false);

} // namespace spdl
