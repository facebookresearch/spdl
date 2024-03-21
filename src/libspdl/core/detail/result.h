#include <libspdl/core/result.h>

#include <folly/futures/Future.h>

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// Result::Impl
////////////////////////////////////////////////////////////////////////////////
template <MediaType media_type, template <MediaType> typename ResultType>
struct Result<media_type, ResultType>::Impl {
  bool fetched{false};
  folly::SemiFuture<ResultType<media_type>> future;

  Impl(folly::SemiFuture<ResultType<media_type>>&& future);

  ResultType<media_type> get();
};

////////////////////////////////////////////////////////////////////////////////
// Results::Impl
////////////////////////////////////////////////////////////////////////////////
template <MediaType media_type, template <MediaType> typename ResultType>
struct Results<media_type, ResultType>::Impl {
  std::vector<std::string> srcs;
  std::vector<std::tuple<double, double>> timestamps;

  folly::SemiFuture<std::vector<folly::SemiFuture<ResultType<media_type>>>>
      future;

  bool fetched{false};

  Impl(
      std::vector<std::string> srcs,
      std::vector<std::tuple<double, double>> timestamps,
      folly::SemiFuture<
          std::vector<folly::SemiFuture<ResultType<media_type>>>>&& future);

  std::vector<ResultType<media_type>> get(bool strict);
};

} // namespace spdl::core
