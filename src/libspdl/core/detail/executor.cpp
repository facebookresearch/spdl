#include "libspdl/core/detail/executor.h"

#include "libspdl/core/detail/logging.h"

#include <folly/Singleton.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/logging/xlog.h>
#include <folly/portability/GFlags.h>
#include <folly/system/HardwareConcurrency.h>

using namespace folly;

FOLLY_GFLAGS_DEFINE_uint32(
    spdl_demuxer_executor_threads,
    0,
    "Number of threads global CPUThreadPoolExecutor will create");

FOLLY_GFLAGS_DEFINE_bool(
    spdl_demuxer_executor_use_throttled_lifo_sem,
    true,
    "Use ThrottledLifoSem in global CPUThreadPoolExecutor");

FOLLY_GFLAGS_DEFINE_uint32(
    spdl_demuxer_executor_wake_up_interval_us,
    0,
    "If --spdl_demuxer_executor_use_throttled_lifo_sem is true, use this "
    "wake-up interval (in microseconds) in ThrottledLifoSem");

FOLLY_GFLAGS_DEFINE_uint32(
    spdl_decoder_executor_threads,
    0,
    "Number of threads global CPUThreadPoolExecutor will create");

FOLLY_GFLAGS_DEFINE_bool(
    spdl_decoder_executor_use_throttled_lifo_sem,
    true,
    "Use ThrottledLifoSem in global CPUThreadPoolExecutor");

FOLLY_GFLAGS_DEFINE_uint32(
    spdl_decoder_executor_wake_up_interval_us,
    0,
    "If --spdl_decoder_executor_use_throttled_lifo_sem is true, use this "
    "wake-up interval (in microseconds) in ThrottledLifoSem");

namespace spdl::core {
namespace detail {
namespace {

class DemuxerTag {};
class DecoderTag {};

CPUThreadPoolExecutor* get_executor(
    size_t num_threads,
    int throttle_interval,
    const std::string& thread_name_prefix) {
  return new CPUThreadPoolExecutor(
      num_threads ? num_threads : hardware_concurrency(),
      throttle_interval >= 0 ? CPUThreadPoolExecutor::makeThrottledLifoSemQueue(
                                   std::chrono::microseconds{throttle_interval})
                             : CPUThreadPoolExecutor::makeDefaultQueue(),
      std::make_shared<NamedThreadFactory>(thread_name_prefix));
}

Singleton<std::shared_ptr<Executor>, DemuxerTag> DEMUX_EXECUTOR([] {
  return new std::shared_ptr<Executor>(get_executor(
      FLAGS_spdl_demuxer_executor_threads,
      FLAGS_spdl_demuxer_executor_use_throttled_lifo_sem
          ? FLAGS_spdl_demuxer_executor_wake_up_interval_us
          : -1,
      "DefaultDemuxThreadPoolExecutor"));
});

Singleton<std::shared_ptr<Executor>, DecoderTag> DECODE_EXECUTOR([] {
  return new std::shared_ptr<Executor>(get_executor(
      FLAGS_spdl_decoder_executor_threads,
      FLAGS_spdl_decoder_executor_use_throttled_lifo_sem
          ? FLAGS_spdl_decoder_executor_wake_up_interval_us
          : -1,
      "DefaultDecodeThreadPoolExecutor"));
});

} // namespace

Executor::KeepAlive<> get_default_demux_executor() {
  auto executorPtrPtr = DEMUX_EXECUTOR.try_get();
  if (!executorPtrPtr) {
    SPDL_FAIL("Requested Demuxer executor during shutdown.");
  }
  return getKeepAliveToken(executorPtrPtr->get());
}

Executor::KeepAlive<> get_default_decode_executor() {
  auto executorPtrPtr = DECODE_EXECUTOR.try_get();
  if (!executorPtrPtr) {
    SPDL_FAIL("Requested Demuxer executor during shutdown.");
  }
  return getKeepAliveToken(executorPtrPtr->get());
}

folly::Executor::KeepAlive<> get_demux_executor(
    std::shared_ptr<ThreadPoolExecutor>& exe) {
  return exe ? exe->impl->get() : get_default_demux_executor();
}

folly::Executor::KeepAlive<> get_decode_executor(
    std::shared_ptr<ThreadPoolExecutor>& exe) {
  return exe ? exe->impl->get() : get_default_decode_executor();
}

} // namespace detail

ThreadPoolExecutor::Impl::Impl(
    size_t num_threads,
    const std::string& thread_name_prefix,
    int throttle_interval)
    : exec(detail::get_executor(
          num_threads,
          throttle_interval,
          thread_name_prefix)) {}

folly::Executor::KeepAlive<> ThreadPoolExecutor::Impl::get() {
  return getKeepAliveToken(exec.get());
}

} // namespace spdl::core
