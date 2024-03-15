#include <libspdl/core/result.h>

#include <folly/futures/Future.h>

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// Result::Impl
////////////////////////////////////////////////////////////////////////////////
template <typename ResultType>
struct Result<ResultType>::Impl {
  bool fetched{false};
  folly::SemiFuture<ResultType> future;

  Impl(folly::SemiFuture<ResultType>&& future);

  ResultType get();
};

////////////////////////////////////////////////////////////////////////////////
// Results::Impl
////////////////////////////////////////////////////////////////////////////////
template <typename ResultType>
struct Results<ResultType>::Impl {
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
