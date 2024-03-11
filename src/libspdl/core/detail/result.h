#include <libspdl/core/result.h>

#include <folly/futures/Future.h>

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// Result::Impl
////////////////////////////////////////////////////////////////////////////////
template <typename ResultType, MediaType media_type>
struct Result<ResultType, media_type>::Impl {
  bool fetched{false};
  folly::SemiFuture<ResultType> future;

  Impl(folly::SemiFuture<ResultType>&& future);

  ResultType get();
};

////////////////////////////////////////////////////////////////////////////////
// Results::Impl
////////////////////////////////////////////////////////////////////////////////
template <typename ResultType, MediaType media_type>
struct Results<ResultType, media_type>::Impl {
  std::vector<std::string> srcs;
  std::vector<std::tuple<double, double>> timestamps;

  folly::SemiFuture<std::vector<folly::SemiFuture<ResultType>>> future;

  bool fetched{false};

  Impl(
      std::vector<std::string> srcs,
      std::vector<std::tuple<double, double>> timestamps,
      folly::SemiFuture<std::vector<folly::SemiFuture<ResultType>>>&& future);

  std::vector<ResultType> get(bool strict);
};

} // namespace spdl::core
