#include "libspdl/coro/detail/executor.h"

#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <folly/Singleton.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/executors/ExecutorWithPriority.h>
#include <folly/logging/xlog.h>
#include <folly/portability/GFlags.h>
#include <folly/system/HardwareConcurrency.h>

using namespace folly;

FOLLY_GFLAGS_DEFINE_uint32(
    spdl_demuxer_executor_threads,
    4,
    "The number of threads the default demux executor creates.");

FOLLY_GFLAGS_DEFINE_uint32(
    spdl_decoder_executor_threads,
    8,
    "The number of threads the default decode executor creates.");

namespace spdl::coro {
namespace detail {
namespace {

class DemuxerTag {};
class DecoderTag {};

CPUThreadPoolExecutor* get_executor(
    size_t num_threads,
    const std::string& thread_name_prefix,
    int num_priorities = 1) {
  return new CPUThreadPoolExecutor(
      num_threads ? num_threads : hardware_concurrency(),
      num_priorities == 1
          ? CPUThreadPoolExecutor::makeDefaultQueue()
          : CPUThreadPoolExecutor::makeDefaultPriorityQueue(num_priorities),
      std::make_shared<NamedThreadFactory>(thread_name_prefix));
}

Singleton<std::shared_ptr<CPUThreadPoolExecutor>, DemuxerTag> DEMUX_EXECUTOR(
    [] {
      return new std::shared_ptr<CPUThreadPoolExecutor>(get_executor(
          FLAGS_spdl_demuxer_executor_threads,
          "DefaultDemuxThreadPoolExecutor",
          2));
    });

Singleton<std::shared_ptr<CPUThreadPoolExecutor>, DecoderTag> DECODE_EXECUTOR(
    [] {
      return new std::shared_ptr<CPUThreadPoolExecutor>(get_executor(
          FLAGS_spdl_decoder_executor_threads,
          "DefaultDecodeThreadPoolExecutor"));
    });

Executor::KeepAlive<> get_default_demux_executor() {
  auto executorPtrPtr = DEMUX_EXECUTOR.try_get();
  if (!executorPtrPtr) {
    SPDL_FAIL("Requested executor during shutdown.");
  }
  return getKeepAliveToken(executorPtrPtr->get());
}

Executor::KeepAlive<> get_default_demux_executor_high_prio() {
  return ExecutorWithPriority::create(
      get_default_demux_executor(), Executor::HI_PRI);
}

Executor::KeepAlive<> get_default_demux_executor_low_prio() {
  return ExecutorWithPriority::create(
      get_default_demux_executor(), Executor::LO_PRI);
}

Executor::KeepAlive<> get_default_decode_executor() {
  auto executorPtrPtr = DECODE_EXECUTOR.try_get();
  if (!executorPtrPtr) {
    SPDL_FAIL("Requested executor during shutdown.");
  }
  return getKeepAliveToken(executorPtrPtr->get());
}

} // namespace

folly::Executor::KeepAlive<> get_demux_executor(ThreadPoolExecutorPtr& exe) {
  return exe ? exe->impl->get() : get_default_demux_executor_low_prio();
}

folly::Executor::KeepAlive<> get_demux_executor_high_prio(
    ThreadPoolExecutorPtr& exe) {
  return exe ? exe->impl->get() : get_default_demux_executor_high_prio();
}

folly::Executor::KeepAlive<> get_decode_executor(ThreadPoolExecutorPtr& exe) {
  return exe ? exe->impl->get() : get_default_decode_executor();
}

} // namespace detail

ThreadPoolExecutor::Impl::Impl(
    size_t num_threads,
    const std::string& thread_name_prefix)
    : exec(detail::get_executor(num_threads, thread_name_prefix)) {}

folly::Executor::KeepAlive<> ThreadPoolExecutor::Impl::get() {
  return getKeepAliveToken(exec.get());
}

size_t ThreadPoolExecutor::Impl::get_task_queue_size() const {
  return exec->getTaskQueueSize();
}

void trace_default_demux_executor_queue_size() {
  auto executorPtrPtr = detail::DEMUX_EXECUTOR.try_get();
  if (!executorPtrPtr) {
    SPDL_FAIL("Requested executor during shutdown.");
  }
  TRACE_COUNTER(
      "demuxing",
      "default_demux_executor_queue_size",
      executorPtrPtr->get()->getTaskQueueSize());
}

void trace_default_decode_executor_queue_size() {
  auto executorPtrPtr = detail::DECODE_EXECUTOR.try_get();
  if (!executorPtrPtr) {
    SPDL_FAIL("Requested executor during shutdown.");
  }
  TRACE_COUNTER(
      "decoding",
      "default_decode_executor_queue_size",
      executorPtrPtr->get()->getTaskQueueSize());
}

} // namespace spdl::coro
