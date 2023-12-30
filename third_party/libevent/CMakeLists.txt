message(STATUS "########################################")
message(STATUS "# Configuring libevent")
message(STATUS "########################################")

set(EVENT__LIBRARY_TYPE "SHARED")
set(EVENT__DISABLE_TESTS ON)
set(EVENT__DISABLE_REGRESS ON)
set(EVENT__DISABLE_SAMPLES ON)
set(EVENT__DISABLE_BENCHMARK ON)
set(EVENT__DISABLE_OPENSSL ON)
set(EVENT__DISABLE_THREAD_SUPPORT ON)

FetchContent_Declare(
  libevent
  URL https://github.com/mthrok/libevent/archive/refs/tags/release-2.1.12-stable-patch.tar.gz
  URL_HASH SHA256=e705cf1969437b72dad60ab8eb3f63fb9d8822245fb412f1ec3dbd5091b90b9f
  DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  )
FetchContent_MakeAvailable(libevent)
