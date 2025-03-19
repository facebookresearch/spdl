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
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace spdl::core {

constexpr bool built_with_cuda() {
  return
#ifdef SPDL_USE_CUDA
      true
#else
      false
#endif
      ;
}

constexpr bool built_with_nvcodec() {
  return
#ifdef SPDL_USE_NVCODEC
      true
#else
      false
#endif
      ;
}

constexpr bool built_with_nvjpeg() {
  return
#ifdef SPDL_USE_NVJPEG
      true
#else
      false
#endif
      ;
}

//////////////////////////////////////////////////////////////////////////////////
// Utilities for FFmpeg
//////////////////////////////////////////////////////////////////////////////////
int get_ffmpeg_log_level();

void set_ffmpeg_log_level(int);

void register_avdevices();

std::vector<std::string> get_ffmpeg_filters();

std::map<std::string, std::tuple<int64_t, int64_t, int64_t>>
get_ffmpeg_versions();

//////////////////////////////////////////////////////////////////////////////////
// Utilities for Glog
//////////////////////////////////////////////////////////////////////////////////
void init_glog(const char* name);

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

// These utilities are for adding custom tracing on Python side
template <typename Number>
void trace_counter(int i, Number counter);
void trace_event_begin(const std::string& name);
void trace_event_end();

} // namespace spdl::core
