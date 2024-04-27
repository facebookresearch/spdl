message(STATUS "########################################")
message(STATUS "# Configuring libevent")
message(STATUS "########################################")

set(EVENT__LIBRARY_TYPE "STATIC")
set(EVENT__DISABLE_TESTS ON)
set(EVENT__DISABLE_REGRESS ON)
set(EVENT__DISABLE_SAMPLES ON)
set(EVENT__DISABLE_BENCHMARK ON)
set(EVENT__DISABLE_OPENSSL ON)
set(EVENT__DISABLE_THREAD_SUPPORT ON)

FetchContent_Declare(
  libevent
  URL https://github.com/libevent/libevent/archive/refs/tags/release-2.1.12-stable.tar.gz
  URL_HASH SHA256=7180a979aaa7000e1264da484f712d403fcf7679b1e9212c4e3d09f5c93efc24
  PATCH_COMMAND patch < ${CMAKE_CURRENT_SOURCE_DIR}/libevent.patch
  DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  )
FetchContent_MakeAvailable(libevent)
