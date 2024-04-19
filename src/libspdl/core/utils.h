#pragma once

#include <libspdl/core/types.h>

#include <memory>
#include <optional>
#include <string>
#include <tuple>

namespace spdl::core {

//////////////////////////////////////////////////////////////////////////////////
// Utilities for FFmpeg
//////////////////////////////////////////////////////////////////////////////////
int get_ffmpeg_log_level();

void set_ffmpeg_log_level(int);

//////////////////////////////////////////////////////////////////////////////////
// Utilities for Folly
//////////////////////////////////////////////////////////////////////////////////
// Convenient method for initializing folly, in case
// applications are not linked to folly directly.
//
// If your application uses folly, then use the proper
// folly::Init mechanism.
void init_folly(int* argc, char*** argv);

//////////////////////////////////////////////////////////////////////////////////
// Utilities for tracing
//////////////////////////////////////////////////////////////////////////////////
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
  void start(int fd, int buffer_size_in_kb);
  void stop();
};

std::unique_ptr<TracingSession> init_tracing();

void trace_default_demux_executor_queue_size();
void trace_default_decode_executor_queue_size();

// These utilities are for adding custom tracing on Python side
template <typename Number>
void trace_counter(int i, Number counter);
void trace_event_begin(const std::string& name);
void trace_event_end();

} // namespace spdl::core
