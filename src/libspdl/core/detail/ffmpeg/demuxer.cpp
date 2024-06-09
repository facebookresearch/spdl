#include "libspdl/core/detail/ffmpeg/demuxer.h"
#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <fmt/format.h>

namespace spdl::core::detail {

Demuxer::Demuxer(AVFormatContext* fmt_ctx_) : fmt_ctx(fmt_ctx_) {}

Generator<AVPacketPtr> Demuxer::demux() {
  int errnum = 0;
  while (errnum >= 0) {
    AVPacketPtr packet{CHECK_AVALLOCATE(av_packet_alloc())};
    {
      TRACE_EVENT("demuxing", "av_read_frame");
      errnum = av_read_frame(fmt_ctx, packet.get());
    }
    if (errnum < 0 && errnum != AVERROR_EOF) {
      CHECK_AVERROR_NUM(
          errnum, fmt::format("Failed to read a packet. ({})", fmt_ctx->url));
    }
    if (errnum == AVERROR_EOF) {
      break;
    }
    co_yield std::move(packet);
  }
}

} // namespace spdl::core::detail
