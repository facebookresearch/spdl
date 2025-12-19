/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libspdl/core/detail/ffmpeg/bsf.h"
#include "libspdl/core/detail/ffmpeg/logging.h"

#include "libspdl/common/logging.h"
#include "libspdl/common/tracing.h"

namespace spdl::core::detail {
namespace {
AVBSFContextPtr init_bsf(
    const std::string& name,
    const AVCodecParameters* codec_par) {
  TRACE_EVENT("demuxing", "init_bsf");
  AVBSFContext* p = nullptr;
  CHECK_AVERROR(
      av_bsf_list_parse_str(name.c_str(), &p), "Failed to create AVBSFContext.")
  AVBSFContextPtr bsf_ctx{p};
  CHECK_AVERROR(
      avcodec_parameters_copy(p->par_in, codec_par),
      "Failed to copy codec parameter.")
  CHECK_AVERROR(av_bsf_init(p), "Failed to initialize AVBSFContext..")
  return bsf_ctx;
}

void send_packet(AVBSFContext* bsf_ctx, AVPacket* packet) {
  TRACE_EVENT("decoding", "av_bsf_send_packet");
  CHECK_AVERROR(
      av_bsf_send_packet(bsf_ctx, packet),
      "Failed to send packet to bit stream filter.")
}

int redeivce_paccket(AVBSFContext* bsf_ctx, AVPacket* packet) {
  int ret = av_bsf_receive_packet(bsf_ctx, packet);
  TRACE_EVENT("decoding", "av_bsf_receive_packet");
  if (ret < 0 && ret != AVERROR_EOF && ret != AVERROR(EAGAIN)) {
    CHECK_AVERROR_NUM(ret, "Failed to fetch packet from bit stream filter.")
  }
  return ret;
}

Generator<AVPacket*> stream_packets(
    const std::vector<AVPacket*>& packets,
    bool flush) {
  for (auto& p : packets) {
    co_yield p;
  }
  if (flush) {
    co_yield nullptr;
  }
}

} // namespace

BSFImpl::BSFImpl(const std::string& name, const AVCodecParameters* codec_par)
    : bsf_ctx_(init_bsf(name, codec_par)) {}

AVCodecParameters* BSFImpl::get_output_codec_par() {
  return bsf_ctx_->par_out;
}

AVRational BSFImpl::get_output_time_base() {
  return bsf_ctx_->time_base_out;
}

Generator<AVPacketPtr> BSFImpl::filter(AVPacket* packet) {
  send_packet(bsf_ctx_.get(), packet);
  int errnum;
  do {
    AVPacketPtr ret{CHECK_AVALLOCATE(av_packet_alloc())};
    switch ((errnum = redeivce_paccket(bsf_ctx_.get(), ret.get()))) {
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

void BSFImpl::filter(
    const std::vector<AVPacket*>& packets,
    PacketSeries& out,
    bool flush) {
  auto packet_stream = stream_packets(packets, flush);
  while (packet_stream) {
    auto filtering = this->filter(packet_stream());
    while (filtering) {
      out.push(filtering().release());
    }
  }
}

void BSFImpl::flush(PacketSeries& out) {
  auto filtering = this->filter(nullptr);
  while (filtering) {
    out.push(filtering().release());
  }
}

} // namespace spdl::core::detail
