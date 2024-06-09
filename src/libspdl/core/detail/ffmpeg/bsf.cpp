#include "libspdl/core/detail/ffmpeg/bsf.h"
#include "libspdl/core/detail/ffmpeg/logging.h"

#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

namespace spdl::core::detail {
namespace {
AVBSFContextPtr init_bsf(
    const std::string& name,
    AVCodecParameters* codec_par) {
  TRACE_EVENT("demuxing", "init_bsf");
  const AVBitStreamFilter* bsf = av_bsf_get_by_name(name.c_str());
  if (!bsf) {
    SPDL_FAIL(fmt::format("Bit stream filter ({}) is not available", name));
  }
  AVBSFContext* p = nullptr;
  CHECK_AVERROR(av_bsf_alloc(bsf, &p), "Failed to allocate AVBSFContext.");
  AVBSFContextPtr bsf_ctx{p};
  CHECK_AVERROR(
      avcodec_parameters_copy(p->par_in, codec_par),
      "Failed to copy codec parameter.");
  CHECK_AVERROR(av_bsf_init(p), "Failed to initialize AVBSFContext..");
  return bsf_ctx;
}

void send_packet(AVBSFContext* bsf_ctx, AVPacket* packet) {
  TRACE_EVENT("decoding", "av_bsf_send_packet");
  CHECK_AVERROR(
      av_bsf_send_packet(bsf_ctx, packet),
      "Failed to send packet to bit stream filter.");
}

int redeivce_paccket(AVBSFContext* bsf_ctx, AVPacket* packet) {
  int ret = av_bsf_receive_packet(bsf_ctx, packet);
  TRACE_EVENT("decoding", "av_bsf_receive_packet");
  if (ret < 0 && ret != AVERROR_EOF && ret != AVERROR(EAGAIN)) {
    CHECK_AVERROR_NUM(ret, "Failed to fetch packet from bit stream filter.");
  }
  return ret;
}
} // namespace

BitStreamFilter::BitStreamFilter(
    const std::string& name,
    AVCodecParameters* codec_par)
    : bsf_ctx(init_bsf(name, codec_par)){};

AVCodecParameters* BitStreamFilter::get_output_codec_par() {
  return bsf_ctx->par_out;
}

Generator<AVPacketPtr> BitStreamFilter::filter(AVPacket* packet) {
  send_packet(bsf_ctx.get(), packet);
  int errnum;
  do {
    AVPacketPtr ret{CHECK_AVALLOCATE(av_packet_alloc())};
    switch ((errnum = redeivce_paccket(bsf_ctx.get(), ret.get()))) {
      case AVERROR(EAGAIN):
        co_return;
      case AVERROR_EOF:
        co_return;
      default: {
        co_yield std::move(ret);
      }
    }
  } while (errnum >= 0);
}

} // namespace spdl::core::detail
