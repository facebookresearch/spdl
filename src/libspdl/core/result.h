#pragma once

#include <libspdl/core/types.h>

#include <vector>

namespace spdl::core {

struct decoding;

////////////////////////////////////////////////////////////////////////////////
// Future for single task
////////////////////////////////////////////////////////////////////////////////

/// Future-like object that holds the result of single asynchronous decoding
/// operation. Used for decoding images.
template <MediaType media_type, template <MediaType> typename ResultType>
class Result {
  struct Impl;

  Impl* pimpl = nullptr;

  Result(Impl* impl);

 public:
  Result() = delete;
  Result(const Result&) = delete;
  Result& operator=(const Result&) = delete;
  Result(Result&&) noexcept;
  Result& operator=(Result&&) noexcept;
  ~Result();

  /// Blocks until the decoding is completed and the frame data is ready.
  /// If the decoding operation fails, throws an exception.
  ResultType<media_type> get();

  friend decoding;
};

////////////////////////////////////////////////////////////////////////////////
// Future for multiple tasks
////////////////////////////////////////////////////////////////////////////////

/// Future-like object that holds the results of multiple asynchronous decoding
/// operation. Used for decoding audio and video clips.
template <MediaType media_type, template <MediaType> typename ResultType>
class Results {
  struct Impl;

  Impl* pimpl = nullptr;

  Results(Impl* impl);

 public:
  Results() = delete;
  Results(const Results&) = delete;
  Results& operator=(const Results&) = delete;
  Results(Results&&) noexcept;
  Results& operator=(Results&&) noexcept;
  ~Results();

  /// Blocks until all the decoding operations are completed and the frame
  /// data are ready.
  ///
  /// If a decoding operation fails, and ``strict==true``, then throws one of
  /// the exception thrown from the failed operation.
  ///
  /// If ``strict==false``, then exceptions are not propagated. However, if
  /// there is no decoding result to return, (all the decoding operations fail)
  /// it throws an exception.
  std::vector<ResultType<media_type>> get(bool strict = true);

  friend decoding;
};

} // namespace spdl::core
