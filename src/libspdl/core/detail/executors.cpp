#include <libspdl/core/detail/executors.h>

#include <libspdl/core/logging.h>

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

namespace spdl::core::detail {
namespace {

class DemuxTag {};
class DecoderTag {};

using DemuxExecutor = CPUThreadPoolExecutor;
using DecoderExecutor = CPUThreadPoolExecutor;

Singleton<std::shared_ptr<Executor>, DemuxTag> DEMUX_EXECUTOR([] {
  size_t nthreads = FLAGS_spdl_demuxer_executor_threads;
  nthreads = nthreads ? nthreads : hardware_concurrency();
  XLOG(DBG9) << "Demux executor #threads: " << nthreads;
  return new std::shared_ptr<Executor>(new DemuxExecutor(
      nthreads,
      FLAGS_spdl_demuxer_executor_use_throttled_lifo_sem
          ? CPUThreadPoolExecutor::makeThrottledLifoSemQueue(
                std::chrono::microseconds{
                    FLAGS_spdl_demuxer_executor_wake_up_interval_us})
          : CPUThreadPoolExecutor::makeDefaultQueue(),
      std::make_shared<NamedThreadFactory>("DemuxerThreadPool")));
});

Singleton<std::shared_ptr<Executor>, DecoderTag> DECODER_EXECUTOR([] {
  size_t nthreads = FLAGS_spdl_decoder_executor_threads;
  nthreads = nthreads ? nthreads : hardware_concurrency();
  XLOG(DBG9) << "Decoder executor #threads: " << nthreads;
  return new std::shared_ptr<Executor>(new DemuxExecutor(
      nthreads,
      FLAGS_spdl_decoder_executor_use_throttled_lifo_sem
          ? CPUThreadPoolExecutor::makeThrottledLifoSemQueue(
                std::chrono::microseconds{
                    FLAGS_spdl_decoder_executor_wake_up_interval_us})
          : CPUThreadPoolExecutor::makeDefaultQueue(),
      std::make_shared<NamedThreadFactory>("DecoderThreadPool")));
});

} // namespace

Executor::KeepAlive<> getDemuxerThreadPoolExecutor() {
  auto executorPtrPtr = DEMUX_EXECUTOR.try_get();
  if (!executorPtrPtr) {
    SPDL_FAIL("Requested Demuxer executor during shutdown.");
  }
  return getKeepAliveToken(executorPtrPtr->get());
}

Executor::KeepAlive<> getDecoderThreadPoolExecutor() {
  auto executorPtrPtr = DECODER_EXECUTOR.try_get();
  if (!executorPtrPtr) {
    SPDL_FAIL("Requested Demuxer executor during shutdown.");
  }
  return getKeepAliveToken(executorPtrPtr->get());
}

} // namespace spdl::core::detail
