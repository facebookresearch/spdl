#include <libspdl/core/decoding.h>

#include <libspdl/core/decoding/results.h>
#include <libspdl/core/detail/executor.h>
#include <libspdl/core/detail/ffmpeg/decoding.h>
#include <libspdl/core/detail/logging.h>

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
Task<std::vector<SemiFuture<Output>>> stream_decode_task(
    const enum MediaType type,
    const std::string src,
    const std::vector<std::tuple<double, double>> timestamps,
    const std::shared_ptr<SourceAdoptor> adoptor,
    const IOConfig io_cfg,
    const DecodeConfig decode_cfg,
    const std::string filter_desc) {
  std::vector<SemiFuture<Output>> futures;
  {
    auto exec = detail::get_default_decode_executor();
    auto demuxer = detail::stream_demux(
        type, src, timestamps, std::move(adoptor), std::move(io_cfg));
    while (auto result = co_await demuxer.next()) {
      auto task = detail::decode_packets(
          *std::move(result), std::move(decode_cfg), std::move(filter_desc));
      futures.emplace_back(std::move(task).scheduleOn(exec).start());
    }
  }
  co_return std::move(futures);
}
} // namespace

MultipleDecodingResult decoding::async_decode(
    const enum MediaType type,
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::shared_ptr<SourceAdoptor>& adoptor,
    const IOConfig& io_cfg,
    const DecodeConfig& decode_cfg,
    const std::string& filter_desc) {
  if (timestamps.size() == 0) {
    SPDL_FAIL("At least one timestamp must be provided.");
  }
  auto task = stream_decode_task(
      type, src, timestamps, adoptor, io_cfg, decode_cfg, filter_desc);
  return MultipleDecodingResult{new MultipleDecodingResult::Impl{
      type,
      {src},
      timestamps,
      std::move(task)
          .scheduleOn(detail::get_default_demux_executor())
          .start()}};
}

} // namespace spdl::core
