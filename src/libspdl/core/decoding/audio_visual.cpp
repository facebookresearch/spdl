#include <libspdl/core/decoding.h>

#include <libspdl/core/detail/executor.h>
#include <libspdl/core/detail/ffmpeg/decoding.h>
#include <libspdl/core/detail/logging.h>
#include <libspdl/core/detail/result.h>

#ifdef SPDL_USE_NVDEC
#include <libspdl/core/detail/nvdec/decoding.h>
#endif

#include <fmt/core.h>
#include <folly/experimental/coro/Task.h>
#include <folly/logging/xlog.h>

#include <cstddef>
#include <cstdint>

namespace spdl::core {

using folly::SemiFuture;
using folly::coro::Task;

namespace {

template <MediaType media_type>
Task<std::vector<SemiFuture<std::unique_ptr<DecodedFrames>>>>
stream_decode_task(
    const std::string src,
    const std::vector<std::tuple<double, double>> timestamps,
    const std::shared_ptr<SourceAdoptor> adoptor,
    const IOConfig io_cfg,
    const DecodeConfig decode_cfg,
    const std::string filter_desc,
    std::shared_ptr<ThreadPoolExecutor> decode_executor) {
  std::vector<SemiFuture<std::unique_ptr<DecodedFrames>>> futures;
  {
    auto exec = detail::get_decode_executor(decode_executor);
    auto demuxer = detail::stream_demux(
        media_type, src, timestamps, std::move(adoptor), std::move(io_cfg));
    while (auto result = co_await demuxer.next()) {
      auto task = detail::decode_packets(
          *std::move(result), std::move(decode_cfg), std::move(filter_desc));
      futures.emplace_back(std::move(task).scheduleOn(exec).start());
    }
  }
  co_return std::move(futures);
}
} // namespace

template <MediaType media_type>
Results<std::unique_ptr<DecodedFrames>, media_type> decoding::async_decode(
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::shared_ptr<SourceAdoptor>& adoptor,
    const IOConfig& io_cfg,
    const DecodeConfig& decode_cfg,
    const std::string& filter_desc,
    std::shared_ptr<ThreadPoolExecutor> demux_executor,
    std::shared_ptr<ThreadPoolExecutor> decode_executor) {
  if (timestamps.size() == 0) {
    SPDL_FAIL("At least one timestamp must be provided.");
  }
  return Results<std::unique_ptr<DecodedFrames>, media_type>{
      new typename Results<std::unique_ptr<DecodedFrames>, media_type>::Impl{
          {src},
          timestamps,
          stream_decode_task<media_type>(
              src,
              timestamps,
              adoptor,
              io_cfg,
              decode_cfg,
              filter_desc,
              decode_executor)
              .scheduleOn(detail::get_demux_executor(demux_executor))
              .start()}};
}

// Explicit instantiations
template Results<std::unique_ptr<DecodedFrames>, MediaType::Video>
decoding::async_decode<MediaType::Video>(
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::shared_ptr<SourceAdoptor>& adoptor,
    const IOConfig& io_cfg,
    const DecodeConfig& decode_cfg,
    const std::string& filter_desc,
    std::shared_ptr<ThreadPoolExecutor> demux_executor,
    std::shared_ptr<ThreadPoolExecutor> decode_executor);

template Results<std::unique_ptr<DecodedFrames>, MediaType::Audio>
decoding::async_decode<MediaType::Audio>(
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::shared_ptr<SourceAdoptor>& adoptor,
    const IOConfig& io_cfg,
    const DecodeConfig& decode_cfg,
    const std::string& filter_desc,
    std::shared_ptr<ThreadPoolExecutor> demux_executor,
    std::shared_ptr<ThreadPoolExecutor> decode_executor);

} // namespace spdl::core
