#include <libspdl/core/utils.h>

#include <libspdl/core/types.h>

#include "libspdl/core/detail/ffmpeg/ctx_utils.h"
#include "libspdl/core/detail/ffmpeg/filter_graph.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#if defined(SPDL_USE_CUDA) || defined(SPDL_USE_NVDEC)
#include "libspdl/core/detail/cuda.h"
#endif

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <cstdint>
#include <mutex>

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

int get_cuda_device_index(unsigned long long ptr) {
#if defined(SPDL_USE_CUDA) || defined(SPDL_USE_NVDEC)
  return detail::get_cuda_device_index(ptr);
#else
  SPDL_FAIL("SPDL is not compiled with CUDA support.");
#endif
}

std::string get_video_filter_description(
    const std::optional<Rational>& frame_rate,
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

namespace {
folly::Init* FOLLY_INIT = nullptr;

void delete_folly_init() {
  delete FOLLY_INIT;
}
} // namespace

void init_folly(int* argc, char*** argv) {
  FOLLY_INIT = new folly::Init{argc, argv};
  std::atexit(delete_folly_init);
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
#ifdef SPDL_ENABLE_TRACING
    detail::init_perfetto();
#else
    XLOG(WARN) << "Tracing is not enabled.";
#endif
  });
}

void TracingSession::config(const std::string& process_name) {
#ifdef SPDL_ENABLE_TRACING
  detail::configure_perfetto(process_name);
#endif
}

void TracingSession::start(int fd) {
#ifdef SPDL_ENABLE_TRACING
  if (sess) {
    SPDL_FAIL("Tracing session is avtive.");
  }
  sess = (void*)(detail::start_tracing_session(fd).release());
#endif
}

void TracingSession::stop() {
#ifdef SPDL_ENABLE_TRACING
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

} // namespace spdl::core
