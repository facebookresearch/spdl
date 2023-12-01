cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(folly-subset)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

###############################################################################
# Third party libraries
###############################################################################
include(FetchContent)
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

# somehow `cmake_policy(SET .. NEW)` does not work for those...
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)

add_subdirectory(third_party/double-conversion)
add_subdirectory(third_party/fmt)
# add_subdirectory(third_party/gflags)
add_subdirectory(third_party/glog)
add_subdirectory(third_party/libevent)
add_subdirectory(third_party/boost)
add_subdirectory(third_party/folly)

###############################################################################
# Build test app
###############################################################################
# add_executable(test foo.cpp)
# target_link_libraries(test glog::glog)

add_executable(coro_test coro_test.cpp)
target_link_libraries(coro_test folly)
