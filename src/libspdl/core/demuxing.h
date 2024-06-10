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

class Demuxer;

using DemuxerPtr = std::unique_ptr<Demuxer>;

class Demuxer {
  std::unique_ptr<DataInterface> di;
  AVFormatContext* fmt_ctx;

 public:
  Demuxer(std::unique_ptr<DataInterface> di);

  ~Demuxer();

  template <MediaType media_type>
  PacketsPtr<media_type> demux_window(
      const std::optional<std::tuple<double, double>>& window = std::nullopt,
      const std::optional<std::string>& bsf = std::nullopt);
};

DemuxerPtr make_demuxer(
    const std::string src,
    const SourceAdaptorPtr& adaptor = nullptr,
    const std::optional<DemuxConfig>& dmx_cfg = std::nullopt);

DemuxerPtr make_demuxer(
    const std::string_view data,
    const std::optional<DemuxConfig>& dmx_cfg = std::nullopt);

// Apply bitstream filter for NVDEC video decoding
VideoPacketsPtr apply_bsf(VideoPacketsPtr packets);

} // namespace spdl::core
