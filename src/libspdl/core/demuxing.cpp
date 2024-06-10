#include <libspdl/core/demuxing.h>

#include "libspdl/core/detail/ffmpeg/bsf.h"
#include "libspdl/core/detail/tracing.h"

extern "C" {
#include <libavformat/avformat.h>
}

namespace spdl::core {

// Implemented in core/detail/ffmpeg/demuxing.cpp
namespace detail {
AVStream* init_fmt_ctx(AVFormatContext* fmt_ctx, enum MediaType type_);

template <MediaType media_type>
PacketsPtr<media_type> demux_window(
    AVFormatContext* fmt_ctx,
    AVStream* stream,
    const std::optional<std::tuple<double, double>>& window = std::nullopt);

std::unique_ptr<DataInterface> get_interface(
    const std::string_view src,
    const SourceAdaptorPtr& adaptor,
    const std::optional<DemuxConfig>& dmx_cfg);

std::unique_ptr<DataInterface> get_in_memory_interface(
    const std::string_view data,
    const std::optional<DemuxConfig>& dmx_cfg);
} // namespace detail

////////////////////////////////////////////////////////////////////////////////
// Demuxer (audio/video)
////////////////////////////////////////////////////////////////////////////////

template <MediaType media_type>
Demuxer<media_type>::Demuxer(std::unique_ptr<DataInterface> di_)
    : di(std::move(di_)),
      fmt_ctx(di->get_fmt_ctx()),
      stream(detail::init_fmt_ctx(fmt_ctx, media_type)){};

template <MediaType media_type>
Demuxer<media_type>::~Demuxer() {
  TRACE_EVENT("demuxing", "Demuxer::~Demuxer");
  di.reset();
  // Techinically, this is not necessary, but doing it here puts
  // the destruction of AVFormatContext under ~StreamingDemuxe, which
  // makes the trace easier to interpret.
}

template <MediaType media_type>
PacketsPtr<media_type> Demuxer<media_type>::demux_window(
    const std::optional<std::tuple<double, double>>& window) {
  auto packets = detail::demux_window<media_type>(fmt_ctx, stream, window);
  if constexpr (media_type == MediaType::Video) {
    auto frame_rate = av_guess_frame_rate(fmt_ctx, stream, nullptr);
    packets->frame_rate = Rational{frame_rate.num, frame_rate.den};
  }
  return packets;
}

template class Demuxer<MediaType::Audio>;
template class Demuxer<MediaType::Video>;

template <MediaType media_type>
DemuxerPtr<media_type> make_demuxer(
    const std::string src,
    const SourceAdaptorPtr& adaptor,
    const std::optional<DemuxConfig>& dmx_cfg) {
  TRACE_EVENT("demuxing", "make_demuxer");
  return std::make_unique<Demuxer<media_type>>(
      detail::get_interface(src, adaptor, dmx_cfg));
}

template DemuxerPtr<MediaType::Audio> make_demuxer(
    const std::string,
    const SourceAdaptorPtr&,
    const std::optional<DemuxConfig>&);
template DemuxerPtr<MediaType::Video> make_demuxer(
    const std::string,
    const SourceAdaptorPtr&,
    const std::optional<DemuxConfig>&);

template <MediaType media_type>
DemuxerPtr<media_type> make_demuxer(
    const std::string_view data,
    const std::optional<DemuxConfig>& dmx_cfg) {
  TRACE_EVENT("demuxing", "make_demuxer");
  return std::make_unique<Demuxer<media_type>>(
      detail::get_in_memory_interface(data, dmx_cfg));
}

template DemuxerPtr<MediaType::Audio> make_demuxer(
    const std::string_view,
    const std::optional<DemuxConfig>&);
template DemuxerPtr<MediaType::Video> make_demuxer(
    const std::string_view,
    const std::optional<DemuxConfig>&);

////////////////////////////////////////////////////////////////////////////////
// Demuxing for Image
////////////////////////////////////////////////////////////////////////////////
ImagePacketsPtr demux_image(
    const std::string uri,
    const SourceAdaptorPtr adaptor,
    const std::optional<DemuxConfig>& dmx_cfg) {
  auto interface = detail::get_interface(uri, adaptor, dmx_cfg);
  auto fmt_ctx = interface->get_fmt_ctx();
  return detail::demux_window<MediaType::Image>(
      fmt_ctx, detail::init_fmt_ctx(fmt_ctx, MediaType::Video));
}

ImagePacketsPtr demux_image(
    const std::string_view data,
    const std::optional<DemuxConfig>& dmx_cfg) {
  thread_local SourceAdaptorPtr adaptor{new BytesAdaptor()};
  auto interface = detail::get_interface(data, adaptor, dmx_cfg);
  auto fmt_ctx = interface->get_fmt_ctx();
  auto result = detail::demux_window<MediaType::Image>(
      fmt_ctx, detail::init_fmt_ctx(fmt_ctx, MediaType::Video));
  return result;
}

////////////////////////////////////////////////////////////////////////////////
// Bit Stream Filtering for NVDEC
////////////////////////////////////////////////////////////////////////////////
VideoPacketsPtr apply_bsf(VideoPacketsPtr packets) {
  // Note
  // FFmpeg's implementation applies BSF to all H264/HEVC formats,
  //
  // https://github.com/FFmpeg/FFmpeg/blob/5e2b0862eb1d408625232b37b7a2420403cd498f/libavcodec/cuviddec.c#L1185-L1191
  //
  // while NVidia SDK samples exclude those with the following substrings in
  // long_name attribute
  //
  //  "QuickTime / MOV", "FLV (Flash Video)", "Matroska / WebM"
  const char* name;
  switch (packets->codecpar->codec_id) {
    case AV_CODEC_ID_H264:
      name = "h264_mp4toannexb";
      break;
    case AV_CODEC_ID_HEVC:
      name = "hevc_mp4toannexb";
      break;
    default:
      return packets;
  }

  TRACE_EVENT("demuxing", "apply_bsf");
  auto bsf = detail::BitStreamFilter{name, packets->codecpar};

  auto ret = std::make_unique<DemuxedPackets<MediaType::Video>>(
      packets->src, bsf.get_output_codec_par(), packets->time_base);
  ret->timestamp = packets->timestamp;
  ret->frame_rate = packets->frame_rate;
  for (auto& packet : packets->get_packets()) {
    auto filtering = bsf.filter(packet);
    while (filtering) {
      ret->push(filtering().release());
    }
  }
  return ret;
}

} // namespace spdl::core
