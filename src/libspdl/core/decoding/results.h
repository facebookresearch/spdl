#include <libspdl/core/decoding.h>

#include <folly/futures/Future.h>

namespace spdl::core {
using Output = std::unique_ptr<DecodedFrames>;

////////////////////////////////////////////////////////////////////////////////
// SingleDecodingResult::Impl
////////////////////////////////////////////////////////////////////////////////
struct SingleDecodingResult::Impl {
  bool fetched{false};
  folly::SemiFuture<Output> future;

  Impl(folly::SemiFuture<Output>&& future);

  Output get();
};

////////////////////////////////////////////////////////////////////////////////
// MultipleDecodingResult::Impl
////////////////////////////////////////////////////////////////////////////////
struct MultipleDecodingResult::Impl {
  enum MediaType type;
  std::vector<std::string> srcs;
  std::vector<std::tuple<double, double>> timestamps;

  folly::SemiFuture<std::vector<folly::SemiFuture<Output>>> future;

  bool fetched{false};

  Impl(
      const enum MediaType type,
      std::vector<std::string> srcs,
      std::vector<std::tuple<double, double>> timestamps,
      folly::SemiFuture<std::vector<folly::SemiFuture<Output>>>&& future);

  std::vector<Output> get(bool strict);
};

} // namespace spdl::core
