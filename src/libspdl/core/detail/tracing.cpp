#include "libspdl/core/detail/tracing.h"

#ifdef SPDL_ENABLE_TRACING

#include <folly/logging/xlog.h>

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

namespace spdl::core::detail {

void init_perfetto() {
  XLOG(INFO) << "Initializing Tracing";
  perfetto::TracingInitArgs args;
  args.backends = perfetto::kInProcessBackend;
  perfetto::Tracing::Initialize(args);
  perfetto::TrackEvent::Register();
}

void configure_perfetto(const std::string& name) {
  perfetto::ProcessTrack process_track = perfetto::ProcessTrack::Current();
  perfetto::protos::gen::TrackDescriptor desc = process_track.Serialize();
  desc.mutable_process()->set_process_name(name);
  perfetto::TrackEvent::SetTrackDescriptor(process_track, desc);
}

std::unique_ptr<perfetto::TracingSession> start_tracing_session(int fd) {
  XLOG(INFO) << "Starting tracing";
  // The trace config defines which types of data sources are enabled for
  // recording. In this example we just need the "track_event" data source,
  // which corresponds to the TRACE_EVENT trace points.
  perfetto::TraceConfig cfg;
  cfg.add_buffers()->set_size_kb(1024);
  auto* ds_cfg = cfg.add_data_sources()->mutable_config();
  ds_cfg->set_name("track_event");

  auto session = perfetto::Tracing::NewTrace();
  session->Setup(cfg, fd);
  session->StartBlocking();
  return session;
}

void stop_tracing_session(std::unique_ptr<perfetto::TracingSession> session) {
  XLOG(INFO) << "Stopping tracing";
  perfetto::TrackEvent::Flush();
  session->StopBlocking();
}

} // namespace spdl::core::detail

#endif
