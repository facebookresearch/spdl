#pragma once

#include <libspdl/core/adaptor.h>
#include <libspdl/core/packets.h>
#include <libspdl/core/types.h>

#include <optional>
#include <string_view>
#include <tuple>

struct AVFormatContext;
struct AVStream;

namespace spdl::core {

template <MediaType media_type>
class StreamingDemuxer;

template <MediaType media_type>
using DemuxerPtr = std::unique_ptr<StreamingDemuxer<media_type>>;

template <MediaType media_type>
class StreamingDemuxer {
  std::unique_ptr<DataInterface> di;
  AVFormatContext* fmt_ctx;
  AVStream* stream;

 public:
  StreamingDemuxer(std::unique_ptr<DataInterface> di);

  ~StreamingDemuxer();

  PacketsPtr<media_type> demux_window(
      const std::optional<std::tuple<double, double>>& window = std::nullopt);
};

template <MediaType media_type>
DemuxerPtr<media_type> make_demuxer(
    const std::string src,
    const SourceAdaptorPtr& adaptor = nullptr,
    const std::optional<DemuxConfig>& dmx_cfg = std::nullopt);

template <MediaType media_type>
DemuxerPtr<media_type> make_demuxer(
    const std::string_view data,
    const std::optional<DemuxConfig>& dmx_cfg = std::nullopt);

// Demux a single image from the resource indicator
ImagePacketsPtr demux_image(
    const std::string src,
    const SourceAdaptorPtr adaptor,
    const std::optional<DemuxConfig>& dmx_cfg);

// Demux a single image from the memory
ImagePacketsPtr demux_image(
    const std::string_view data,
    const std::optional<DemuxConfig>& dmx_cfg);

// Apply bitstream filter for NVDEC video decoding
VideoPacketsPtr apply_bsf(VideoPacketsPtr packets);

} // namespace spdl::core
