message(STATUS "########################################")
message(STATUS "Configuring folly")
message(STATUS "########################################")

FetchContent_Declare(
  folly_src
  GIT_REPOSITORY https://github.com/facebook/folly
  GIT_TAG v2023.12.11.00
  PATCH_COMMAND patch -N -i "${CMAKE_CURRENT_SOURCE_DIR}/m1_mac_memalign.patch" || true
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
  SUBBUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/subbuild"
  )
FetchContent_Populate(folly_src)

#------------------------------------------------------------------------------
# Configure
#------------------------------------------------------------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-nullability-completeness")

set(
  defs
  FOLLY_HAS_COROUTINES
  FOLLY_HAVE_PTHREAD
  FOLLY_HAVE_LIBGFLAGS
  )

set(base_dir ${folly_src_SOURCE_DIR})
set(CMAKE_MODULE_PATH
  "${base_dir}/CMake"
  )
include(FollyConfigChecks)
configure_file(
  ${base_dir}/CMake/folly-config.h.cmake
  ${base_dir}/folly/folly-config.h
)

set(
  src
  "${base_dir}/folly/CancellationToken.cpp"
  "${base_dir}/folly/Conv.cpp"
  "${base_dir}/folly/Demangle.cpp"
  "${base_dir}/folly/dynamic.cpp"
  "${base_dir}/folly/Executor.cpp"
  "${base_dir}/folly/ExceptionString.cpp"
  "${base_dir}/folly/ExceptionWrapper.cpp"
  "${base_dir}/folly/File.cpp"
  "${base_dir}/folly/FileUtil.cpp"
  "${base_dir}/folly/Format.cpp"
  "${base_dir}/folly/ScopeGuard.cpp"
  "${base_dir}/folly/SharedMutex.cpp"
  "${base_dir}/folly/Singleton.cpp"
  "${base_dir}/folly/SingletonThreadLocal.cpp"
  "${base_dir}/folly/String.cpp"
  "${base_dir}/folly/Unicode.cpp"
  "${base_dir}/folly/json.cpp"
  "${base_dir}/folly/json_pointer.cpp"

  "${base_dir}/folly/concurrency/CacheLocality.cpp"
  "${base_dir}/folly/concurrency/ProcessLocalUniqueId.cpp"

  "${base_dir}/folly/container/detail/F14Table.cpp"

  "${base_dir}/folly/detail/AsyncTrace.cpp"
  "${base_dir}/folly/detail/FileUtilDetail.cpp"
  "${base_dir}/folly/detail/Futex.cpp"
  "${base_dir}/folly/detail/MemoryIdler.cpp"
  "${base_dir}/folly/detail/SplitStringSimd.cpp"
  "${base_dir}/folly/detail/StaticSingletonManager.cpp"
  "${base_dir}/folly/detail/UniqueInstance.cpp"
  "${base_dir}/folly/detail/ThreadLocalDetail.cpp"

  "${base_dir}/folly/fibers/Baton.cpp"
  "${base_dir}/folly/fibers/Fiber.cpp"
  "${base_dir}/folly/fibers/FiberManager.cpp"
  "${base_dir}/folly/fibers/GuardPageAllocator.cpp"
  "${base_dir}/folly/fibers/Semaphore.cpp"

  "${base_dir}/folly/futures/detail/Core.cpp"

  "${base_dir}/folly/executors/GlobalExecutor.cpp"
  "${base_dir}/folly/executors/GlobalThreadPoolList.cpp"
  "${base_dir}/folly/executors/IOThreadPoolExecutor.cpp"
  "${base_dir}/folly/executors/ThreadPoolExecutor.cpp"
  "${base_dir}/folly/executors/CPUThreadPoolExecutor.cpp"
  "${base_dir}/folly/executors/QueueObserver.cpp"
  "${base_dir}/folly/executors/QueuedImmediateExecutor.cpp"

  "${base_dir}/folly/hash/SpookyHashV2.cpp"

  "${base_dir}/folly/io/async/AsyncTimeout.cpp"
  "${base_dir}/folly/io/async/DelayedDestruction.cpp"
  "${base_dir}/folly/io/async/EventBase.cpp"
  "${base_dir}/folly/io/async/EventBaseBackendBase.cpp"
  "${base_dir}/folly/io/async/EventBaseLocal.cpp"
  "${base_dir}/folly/io/async/EventBaseManager.cpp"
  "${base_dir}/folly/io/async/EventHandler.cpp"
  "${base_dir}/folly/io/async/HHWheelTimer.cpp"
  "${base_dir}/folly/io/async/Request.cpp"
  "${base_dir}/folly/io/async/TimeoutManager.cpp"
  "${base_dir}/folly/io/async/VirtualEventBase.cpp"

  "${base_dir}/folly/init/Init.cpp"
  "${base_dir}/folly/init/Phase.cpp"

  "${base_dir}/folly/lang/CString.cpp"
  "${base_dir}/folly/lang/Exception.cpp"
  "${base_dir}/folly/lang/SafeAssert.cpp"
  "${base_dir}/folly/lang/ToAscii.cpp"
  "${base_dir}/folly/lang/UncaughtExceptions.cpp"

  "${base_dir}/folly/logging/AsyncFileWriter.cpp"
  "${base_dir}/folly/logging/AsyncLogWriter.cpp"
  "${base_dir}/folly/logging/CustomLogFormatter.cpp"
  "${base_dir}/folly/logging/ImmediateFileWriter.cpp"
  "${base_dir}/folly/logging/Init.cpp"
  "${base_dir}/folly/logging/InitWeak.cpp"
  "${base_dir}/folly/logging/FileWriterFactory.cpp"
  "${base_dir}/folly/logging/GlogStyleFormatter.cpp"
  "${base_dir}/folly/logging/LogCategory.cpp"
  "${base_dir}/folly/logging/LogCategoryConfig.cpp"
  "${base_dir}/folly/logging/LogConfig.cpp"
  "${base_dir}/folly/logging/LogConfigParser.cpp"
  "${base_dir}/folly/logging/LoggerDB.cpp"
  "${base_dir}/folly/logging/LogHandlerConfig.cpp"
  "${base_dir}/folly/logging/LogLevel.cpp"
  "${base_dir}/folly/logging/LogMessage.cpp"
  "${base_dir}/folly/logging/LogName.cpp"
  "${base_dir}/folly/logging/LogStream.cpp"
  "${base_dir}/folly/logging/LogStreamProcessor.cpp"
  "${base_dir}/folly/logging/RateLimiter.cpp"
  "${base_dir}/folly/logging/StandardLogHandler.cpp"
  "${base_dir}/folly/logging/StandardLogHandlerFactory.cpp"
  "${base_dir}/folly/logging/StreamHandlerFactory.cpp"
  "${base_dir}/folly/logging/xlog.cpp"

  "${base_dir}/folly/memory/MallctlHelper.cpp"
  "${base_dir}/folly/memory/ReentrantAllocator.cpp"
  "${base_dir}/folly/memory/detail/MallocImpl.cpp"

  "${base_dir}/folly/net/NetOps.cpp"

  "${base_dir}/folly/synchronization/AtomicNotification.cpp"
  "${base_dir}/folly/synchronization/DistributedMutex.cpp"
  "${base_dir}/folly/synchronization/Hazptr.cpp"
  "${base_dir}/folly/synchronization/HazptrDomain.cpp"
  "${base_dir}/folly/synchronization/HazptrThreadPoolExecutor.cpp"
  "${base_dir}/folly/synchronization/ParkingLot.cpp"

  "${base_dir}/folly/system/AtFork.cpp"
  "${base_dir}/folly/system/HardwareConcurrency.cpp"
  "${base_dir}/folly/system/Pid.cpp"
  "${base_dir}/folly/system/ThreadId.cpp"
  "${base_dir}/folly/system/ThreadName.cpp"

  "${base_dir}/folly/tracing/AsyncStack.cpp"

  "${base_dir}/folly/experimental/coro/Baton.cpp"
  "${base_dir}/folly/experimental/coro/detail/Malloc.cpp"
  "${base_dir}/folly/experimental/symbolizer/SignalHandler.cpp"
  )

add_library(folly "${src}")
target_include_directories(folly PUBLIC "${base_dir}")
target_compile_definitions(folly PRIVATE "${defs}")
target_link_libraries(
  folly
  PRIVATE
  c++abi
  PUBLIC
  event_core
  double-conversion
  fmt
  gflags::gflags
  glog::glog
  Boost::algorithm
  Boost::intrusive
  Boost::context
  Boost::preprocessor
  Boost::variant
  Boost::iterator
)
set_property(TARGET folly PROPERTY CXX_STANDARD 20)
