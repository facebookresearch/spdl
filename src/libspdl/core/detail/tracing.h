#pragma once

#ifndef SPDL_ENABLE_TRACING

#define TRACE_EVENT(...)
#define TRACE_EVENT_BEGIN(...)
#define TRACE_EVENT_END(...)
#define TRACE_EVENT_INSTANT(...)

#else

#include <perfetto.h>

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("demuxing").SetDescription("Demuxing events"),
    perfetto::Category("decoding").SetDescription("Decoding events"),
    perfetto::Category("nvdec").SetDescription("Decoding events using NVDEC"));

namespace spdl::core::detail {
void init_perfetto();
void configure_perfetto(const std::string& process_name);

std::unique_ptr<perfetto::TracingSession> start_tracing_session(int fd);
void stop_tracing_session(std::unique_ptr<perfetto::TracingSession> sess);
} // namespace spdl::core::detail

#endif
