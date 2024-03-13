#pragma once

#include <libspdl/core/types.h>

#include <memory>
#include <optional>
#include <string>
#include <tuple>

namespace spdl::core {

int get_ffmpeg_log_level();

void set_ffmpeg_log_level(int);

void clear_ffmpeg_cuda_context_cache();

void create_cuda_context(int index, bool use_primary_context = false);

// Fetch device index from data pointer
int get_cuda_device_index(unsigned long long ptr);

std::string get_video_filter_description(
    const std::optional<Rational>& frame_rate,
    const std::optional<int>& width,
    const std::optional<int>& height,
    const std::optional<std::string>& pix_fmt);

std::string get_audio_filter_description(
    const std::optional<int>& sample_rate,
    const std::optional<int>& num_channels,
    const std::optional<std::string>& sample_fmt);

// Convenient method for initializing folly, in case
// applications are not linked to folly directly.
//
// If your application uses folly, then use the proper
// folly::Init mechanism.
void init_folly(int* argc, char*** argv);

class TracingSession {
  void* sess = nullptr;

 public:
  explicit TracingSession(void* sess = nullptr);
  TracingSession(const TracingSession&) = delete;
  TracingSession& operator=(const TracingSession&) = delete;
  TracingSession(TracingSession&& other) noexcept;
  TracingSession& operator=(TracingSession&& other) noexcept;
  ~TracingSession();

  void init();
  void config(const std::string& process_name);
  void start(int fd);
  void stop();
};

std::unique_ptr<TracingSession> init_tracing();

} // namespace spdl::core
