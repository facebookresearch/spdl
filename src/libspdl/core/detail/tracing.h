/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifndef SPDL_USE_TRACING

#define TRACE_COUNTER(...)
#define TRACE_EVENT(...)
#define TRACE_EVENT_BEGIN(...)
#define TRACE_EVENT_END(...)
#define TRACE_EVENT_INSTANT(...)

#else

#include <perfetto.h>

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("demuxing").SetDescription("Demuxing events"),
    perfetto::Category("decoding").SetDescription("Decoding events"),
    perfetto::Category("nvdec").SetDescription("Decoding events using NVDEC"),
    perfetto::Category("other").SetDescription(
        "Other events used for testing"));

namespace spdl::core::detail {
void init_perfetto();
void configure_perfetto(const std::string& process_name);

std::unique_ptr<perfetto::TracingSession> start_tracing_session(
    int fd,
    int buffer_size_in_kb);
void stop_tracing_session(std::unique_ptr<perfetto::TracingSession> sess);

} // namespace spdl::core::detail

#ifdef SPDL_USE_TORCH_RECORD_FUNCTION

// Use PyTorch record_function for TRACE_EVENT so that the trace shows
// up on PyTorch profiler.
// Note: Currently `_ExperimentalConfig(profile_all_threads=True)` is required
// to show the traces in subthreads.

#include <c10/util/ScopeExit.h>
#include <torch/csrc/autograd/record_function_ops.h>

#define SPDL_CONCAT_IMPL(x, y) x##y
#define SPDL_CONCAT(x, y) SPDL_CONCAT_IMPL(x, y)

namespace spdl::core::detail {

thread_local inline c10::intrusive_ptr<
    torch::autograd::profiler::PythonRecordFunction>
    _REC;

}

#undef TRACE_EVENT
#define TRACE_EVENT(category, name, ...)                          \
  spdl::core::detail::_REC =                                      \
      torch::autograd::profiler::record_function_enter_new(name); \
  auto SPDL_CONCAT(pytorch_record_guard_, __COUNTER__) =          \
      c10::make_scope_exit([&]() {                                \
        if (spdl::core::detail::_REC) {                           \
          spdl::core::detail::_REC->record.end();                 \
          spdl::core::detail::_REC.reset();                       \
        }                                                         \
      });
#endif

#endif
