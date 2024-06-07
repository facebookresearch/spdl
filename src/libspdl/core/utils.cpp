#include <libspdl/core/utils.h>

#include <libspdl/core/types.h>

#include "libspdl/core/detail/ffmpeg/ctx_utils.h"
#include "libspdl/core/detail/ffmpeg/filter_graph.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <fmt/core.h>
#include <glog/logging.h>

#include <cstdint>
#include <mutex>

extern "C" {
#include <libavdevice/avdevice.h>
#include <libavutil/log.h>
}

namespace spdl::core {

int get_ffmpeg_log_level() {
  return av_log_get_level();
}

void set_ffmpeg_log_level(int level) {
  av_log_set_level(level);
}

void register_avdevices() {
  avdevice_register_all();
}

////////////////////////////////////////////////////////////////////////////////
// Tracing
////////////////////////////////////////////////////////////////////////////////
TracingSession::TracingSession(void* s) : sess(s) {}
TracingSession::TracingSession(TracingSession&& other) noexcept {
  *this = std::move(other);
}
TracingSession& TracingSession::operator=(TracingSession&& other) noexcept {
  using std::swap;
  swap(sess, other.sess);
  return *this;
}
TracingSession::~TracingSession() {
  if (sess) {
    stop();
  }
}

void TracingSession::init() {
  static std::once_flag flag;
  std::call_once(flag, []() {
#ifdef SPDL_USE_TRACING
    detail::init_perfetto();
#else
    LOG(WARNING) << "Tracing is not enabled.";
#endif
  });
}

void TracingSession::config(const std::string& process_name) {
#ifdef SPDL_USE_TRACING
  detail::configure_perfetto(process_name);
#endif
}

void TracingSession::start(int fd, int buffer_size_in_kb) {
#ifdef SPDL_USE_TRACING
  if (sess) {
    SPDL_FAIL("Tracing session is avtive.");
  }
  sess =
      (void*)(detail::start_tracing_session(fd, buffer_size_in_kb).release());
#endif
}

void TracingSession::stop() {
#ifdef SPDL_USE_TRACING
  if (!sess) {
    SPDL_FAIL("Tracing session is not avtive.");
  }
  std::unique_ptr<perfetto::TracingSession> p;
  p.reset((perfetto::TracingSession*)sess);
  sess = nullptr;
  detail::stop_tracing_session(std::move(p));
#endif
}

std::unique_ptr<TracingSession> init_tracing() {
  return std::make_unique<TracingSession>();
}

template <typename Number>
void trace_counter(int i, Number val) {
#ifdef SPDL_USE_TRACING

#define _CASE(i)                                \
  case i: {                                     \
    TRACE_COUNTER("other", "Counter " #i, val); \
    return;                                     \
  }

  switch (i) {
    _CASE(0);
    _CASE(1);
    _CASE(2);
    _CASE(3);
    _CASE(4);
    _CASE(5);
    _CASE(6);
    _CASE(7);
    default:
      SPDL_FAIL(fmt::format(
          "Counter {} is not supported. The valid value range is [0, 7].", i));
  }
#undef _CASE

#endif
}

template void trace_counter<int>(int i, int counter);
template void trace_counter<double>(int i, double counter);

void trace_event_begin(const std::string& name) {
#ifdef SPDL_USE_TRACING
  TRACE_EVENT_BEGIN("other", perfetto::DynamicString{name});
#endif
}

void trace_event_end() {
#ifdef SPDL_USE_TRACING
  TRACE_EVENT_END("other");
#endif
}

} // namespace spdl::core
