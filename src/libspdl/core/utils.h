/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/types.h>

#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace spdl::core {

//////////////////////////////////////////////////////////////////////////////////
// FFmpeg utilities
//////////////////////////////////////////////////////////////////////////////////

/// Get the current FFmpeg log level.
///
/// @return Current log level value.
int get_ffmpeg_log_level();

/// Set the FFmpeg log level.
///
/// @param level Log level to set (e.g., AV_LOG_QUIET, AV_LOG_INFO).
void set_ffmpeg_log_level(int level);

/// Register FFmpeg AVDevices.
///
/// Enables support for device input/output (e.g., webcam, microphone).
void register_avdevices();

/// Get list of available FFmpeg filters.
///
/// @return Vector of filter names.
std::vector<std::string> get_ffmpeg_filters();

/// Get FFmpeg library versions.
///
/// @return Map of library names to version tuples (major, minor, micro).
std::map<std::string, std::tuple<int64_t, int64_t, int64_t>>
get_ffmpeg_versions();

//////////////////////////////////////////////////////////////////////////////////
// Glog utilities
//////////////////////////////////////////////////////////////////////////////////

/// Initialize Google logging (glog).
///
/// @param name Program name for logging.
void init_glog(const char* name);

//////////////////////////////////////////////////////////////////////////////////
// Tracing utilities
//////////////////////////////////////////////////////////////////////////////////

/// Tracing session for performance profiling.
///
/// TracingSession manages Perfetto tracing for profiling media operations.
class TracingSession {
  void* sess_ = nullptr;

 public:
  /// Construct a tracing session.
  ///
  /// @param sess Optional session pointer.
  explicit TracingSession(void* sess = nullptr);

  /// Deleted copy constructor.
  TracingSession(const TracingSession&) = delete;

  /// Deleted copy assignment operator.
  TracingSession& operator=(const TracingSession&) = delete;

  /// Move constructor.
  TracingSession(TracingSession&& other) noexcept;

  /// Move assignment operator.
  TracingSession& operator=(TracingSession&& other) noexcept;

  /// Destructor.
  ~TracingSession();

  /// Initialize the tracing session.
  void init();

  /// Configure the tracing session.
  ///
  /// @param process_name Name of the process being traced.
  void config(const std::string& process_name);

  /// Start tracing to a file descriptor.
  ///
  /// @param fd File descriptor to write trace data.
  /// @param buffer_size_in_kb Buffer size in kilobytes.
  void start(int fd, int buffer_size_in_kb);

  /// Stop tracing.
  void stop();
};

/// Initialize a tracing session.
///
/// @return Unique pointer to a TracingSession.
std::unique_ptr<TracingSession> init_tracing();

/// Record a counter value for tracing.
///
/// @tparam Number Numeric type for the counter.
/// @param i Counter ID.
/// @param counter Counter value.
template <typename Number>
void trace_counter(int i, Number counter);

/// Begin a trace event.
///
/// @param name Event name.
void trace_event_begin(const std::string& name);

/// End the current trace event.
void trace_event_end();

} // namespace spdl::core
