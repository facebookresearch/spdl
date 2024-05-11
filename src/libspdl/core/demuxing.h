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
class StreamingDemuxer {
  std::unique_ptr<DataInterface> di;
  AVFormatContext* fmt_ctx;
  AVStream* stream;

 public:
  StreamingDemuxer(
      const std::string uri,
      const SourceAdaptorPtr& adaptor,
      const std::optional<DemuxConfig>& io_cfg);

  StreamingDemuxer(
      const std::string_view data,
      const std::optional<DemuxConfig>& io_cfg);

  PacketsPtr<media_type> demux_window(const std::tuple<double, double>& window);
};

// Demux a single image from the resource indicator
ImagePacketsPtr demux_image(
    const std::string uri,
    const SourceAdaptorPtr adaptor,
    const std::optional<DemuxConfig>& io_cfg);

// Demux a single image from the memory
// _zero_clear sets all the data to zero. This is only for testing.
ImagePacketsPtr demux_image(
    const std::string_view src,
    const std::optional<DemuxConfig>& io_cfg,
    bool _zero_clear = false);

// Apply bitstream filter for NVDEC video decoding
VideoPacketsPtr apply_bsf(VideoPacketsPtr packets);

} // namespace spdl::core
